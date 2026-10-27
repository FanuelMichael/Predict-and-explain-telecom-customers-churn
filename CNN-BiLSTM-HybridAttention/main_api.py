import os, logging, warnings, gc, json, base64
from io import BytesIO

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prometheus_fastapi_instrumentator import Instrumentator

# Import your existing modules
from Data_Preprocessing import TelcoDataProcessor
from Hybrid_DeepLearning_Model import HybridModel
from SHAP_Explainability import SHAPExplainer
from LIME_Explainability import LIMEExplainer
from PDP_Explainability import PDP_Explainer

# Global state
ml_state = {}


# Setup JSON Logger for Loki
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "service": "cnn-bilstm-attention-api",
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


logger = logging.getLogger("churn_api")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(JsonFormatter())
logger.addHandler(console_handler)

try:
    from logging_loki import LokiHandler

    loki_handler = LokiHandler(
        url=os.getenv("LOKI_URL", "http://loki:3100/loki/api/v1/push"),
        tags={"application": "cnn-bilstm-attention-api", "environment": "docker"},
        version="1",
    )
    loki_handler.setFormatter(JsonFormatter())
    logger.addHandler(loki_handler)
except Exception as e:
    logger.warning(f"Loki handler not attached: {e}")


def fig_to_base64(fig):
    """Converts a Matplotlib figure to a Base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up... Loading Data and Training Model.")

    # 1. Load Data
    processor = TelcoDataProcessor('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_cleaned = processor.clean_data()
    y = df_cleaned['Churn'].values.astype(np.int32)
    X_df = df_cleaned.drop('Churn', axis=1)
    indices = df_cleaned.index.values

    X_train_df_raw, X_test_df_raw, y_train_full, y_test, _, indices_test = train_test_split(
        X_df, y, indices, test_size=0.2, random_state=42, stratify=y
    )

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlyCharge']

    # 2. Preprocess
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(X_train_df_raw, y_train_full, test_size=0.20, random_state=42,
                                                      stratify=y_train_full)
    state = processor.make_preprocessing_state()
    X_tr_df, state = processor.fit_transform_preprocessing(X_tr_raw, state)
    X_va_df = processor.transform_preprocessing(X_va_raw, state)
    X_te_df = processor.transform_preprocessing(X_test_df_raw, state)

    X_tr, X_va, X_te = X_tr_df.values.astype(np.float32), X_va_df.values.astype(np.float32), X_te_df.values.astype(
        np.float32)
    num_idx = [state["feature_names"].index(c) for c in numerical_cols if c in state["feature_names"]]

    scaler = StandardScaler()
    X_tr[:, num_idx] = scaler.fit_transform(X_tr[:, num_idx])
    X_va[:, num_idx] = scaler.transform(X_va[:, num_idx])
    X_te[:, num_idx] = scaler.transform(X_te[:, num_idx])

    X_tr_res, y_tr_res, _ = processor.apply_smote(X_tr, y_tr)

    # 3. Train Model (Using fewer epochs for faster API startup)
    logger.info("Training Hybrid Model...")
    deploy_model = HybridModel.build(input_shape=(X_tr_res.shape[1], 1), lr=0.0003)
    deploy_model.fit(X_tr_res, y_tr_res, epochs=5, batch_size=64,
                     validation_data=(X_va.reshape(-1, X_va.shape[1], 1), y_va), verbose=0)

    # 4. Setup Explainers (using a subset for speed)
    X_train_2d = X_tr.reshape(X_tr.shape[0], -1)
    X_test_2d = X_te

    # Wrap model for 2D input for SHAP/LIME
    inp_2d = tf.keras.Input(shape=(X_tr.shape[1],))
    shap_lime_model = tf.keras.Model(inputs=inp_2d,
                                     outputs=deploy_model(tf.keras.layers.Reshape((X_tr.shape[1], 1))(inp_2d)))

    shap_explainer = SHAPExplainer(shap_lime_model, state["feature_names"], X_train_2d, X_test_2d[:40],
                                   background_samples=50, algorithm="permutation")
    lime_explainer = LIMEExplainer(shap_lime_model, state["feature_names"], X_train_2d, X_test_2d)

    X_train_3d = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
    X_test_3d = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)
    pdp_explainer = PDP_Explainer(deploy_model, state["feature_names"], X_train_3d, X_test_3d)

    # Save to global state
    ml_state["model"] = deploy_model
    ml_state["processor"] = processor
    ml_state["indices_test"] = indices_test
    ml_state["X_test"] = X_test_2d
    ml_state["shap"] = shap_explainer
    ml_state["lime"] = lime_explainer
    ml_state["pdp"] = pdp_explainer
    ml_state["shap_top_features"] = shap_explainer.get_top_shap_features()

    logger.info("Startup complete. Model and Explainers ready.")
    yield
    ml_state.clear()
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan, title="CNN-BiLSTM Churn Prediction & XAI API")

# Instrument with Prometheus
Instrumentator().instrument(app).expose(app)


class CustomerRequest(BaseModel):
    customer_index: int


@app.get("/")
def read_root():
    return {"message": "Welcome to the CNN-BiLSTM Hybrid Attention API. Use POST /predict to get a prediction."}


@app.post("/predict")
def predict(data: CustomerRequest):
    if data.customer_index < 0 or data.customer_index >= len(ml_state["X_test"]):
        raise HTTPException(status_code=400, detail=f"Index must be between 0 and {len(ml_state['X_test']) - 1}")

    x_sample_2d = ml_state["X_test"][[data.customer_index]]
    x_sample_3d = x_sample_2d.reshape(1, x_sample_2d.shape[1], 1)

    proba = float(ml_state["model"].predict(x_sample_3d, verbose=0)[0][0])
    prediction = int(proba >= 0.5)

    logger.info(f"Prediction made for index {data.customer_index}: {prediction} (Prob: {proba:.4f})")
    return {"customer_index": data.customer_index, "churn_prediction": prediction, "churn_probability": proba,
            "status": "success"}


@app.get("/explain/shap/{customer_index}")
def get_shap_explanation(customer_index: int):
    if customer_index < 0 or customer_index >= 40:
        raise HTTPException(status_code=400, detail="SHAP is computed only for indices 0-39.")

    fig = ml_state["shap"].plot_waterfall(cust_actual_id=f"Index {customer_index}", sample_idx=customer_index)
    img_base64 = fig_to_base64(fig)
    html_content = f"<html><body><h2>SHAP Waterfall Plot for Customer {customer_index}</h2><img src='data:image/png;base64,{img_base64}'/></body></html>"
    return HTMLResponse(content=html_content)


@app.get("/explain/lime/{customer_index}")
def get_lime_explanation(customer_index: int):
    if customer_index < 0 or customer_index >= len(ml_state["X_test"]):
        raise HTTPException(status_code=400, detail="Invalid index.")

    fig = ml_state["lime"].lime_bar_chart_plot(cust_actual_id=f"Index {customer_index}", sample_idx=customer_index)
    img_base64 = fig_to_base64(fig)
    html_content = f"<html><body><h2>LIME Bar Chart for Customer {customer_index}</h2><img src='data:image/png;base64,{img_base64}'/></body></html>"
    return HTMLResponse(content=html_content)


@app.get("/explain/pdp")
def get_pdp_explanation():
    top_features = ml_state["shap_top_features"][:4]
    fig = ml_state["pdp"].plot_pdp_grid(top_features)
    img_base64 = fig_to_base64(fig)
    html_content = f"<html><body><h2>Partial Dependence Plots (Top 4 Features)</h2><img src='data:image/png;base64,{img_base64}'/></body></html>"
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_api:app", host="0.0.0.0", port=8000)