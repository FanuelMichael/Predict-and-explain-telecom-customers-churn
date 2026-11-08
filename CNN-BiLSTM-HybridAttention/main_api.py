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

if os.getenv("ENABLE_LOKI", "false").lower() == "true":
    try:
        from logging_loki import LokiHandler

        loki_handler = LokiHandler(
            url=os.getenv("LOKI_URL", "http://loki:3100/loki/api/v1/push"),
            tags={"application": "cnn-bilstm-attention-api", "environment": "docker"},
            version="1")
        loki_handler.setFormatter(JsonFormatter())
        logger.addHandler(loki_handler)
        print("Loki handler attached successfully.")
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

    # Calculate predictions for all test data so the UI can filter by Churn/No Churn
    test_probas = deploy_model.predict(X_test_3d, verbose=0).ravel()
    ml_state["y_pred_test"] = (test_probas >= 0.5).astype(int)
    ml_state["y_test"] = y_test
    # Save to global state
    ml_state["model"] = deploy_model

    logger.info("Startup complete. Model and Explainers ready.")
    yield
    ml_state.clear()
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan, title="CNN-BiLSTM Churn Prediction & XAI API")

# Instrument with Prometheus
Instrumentator().instrument(app).expose(app)


class CustomerRequest(BaseModel):
    customer_index: int


# ==========================================
# 1. The missing Table Generator Function
# ==========================================
def create_customer_profile_figure(customer_dict):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.axis("off")
    keys = list(customer_dict.keys())
    values = [str(v) for v in customer_dict.values()]
    table_data = list(zip(keys, values))

    table = ax.table(cellText=table_data, colLabels=["Feature", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=11)
            cell.set_facecolor('#1E293B')
            cell.set_edgecolor('#1E293B')
        else:
            cell.set_facecolor('#F8FAFC' if row % 2 == 0 else '#FFFFFF')
            cell.set_edgecolor('#E2E8F0')
            cell.set_text_props(color='#334155')
            if col == 0:
                cell.set_text_props(weight='bold', color='#0F172A')
    plt.close(fig)
    return fig


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
    if customer_index < 0 or customer_index >= len(ml_state["X_test"]):
        raise HTTPException(status_code=400, detail="Invalid index.")

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


@app.get("/customer/profile/{customer_index}")
def get_customer_profile(customer_index: int):
    if customer_index < 0 or customer_index >= len(ml_state["indices_test"]):
        raise HTTPException(status_code=400, detail="Invalid index.")

    # Generate the customer profile dynamically for this specific index
    raw_data = ml_state["processor"].get_customer_original_data(customer_index, ml_state["indices_test"])
    fig_cust = create_customer_profile_figure(raw_data)
    img_base64 = fig_to_base64(fig_cust)

    html_content = f"<html><body><h2>Customer Original Profile (Index {customer_index})</h2><img src='data:image/png;base64,{img_base64}'/></body></html>"
    return HTMLResponse(content=html_content)

@app.get("/api/customer-data/{customer_index}")
def get_customer_json_data(customer_index: int):
    if customer_index < 0 or customer_index >= len(ml_state["indices_test"]):
        raise HTTPException(status_code=400, detail="Invalid index.")

    raw_data = ml_state["processor"].get_customer_original_data(customer_index, ml_state["indices_test"])
    return raw_data

@app.get("/full-report/{customer_index}")
def get_full_report(customer_index: int):
    logger.info(f"Generating Full XAI Report for Customer {customer_index}")

    # 1. Generate all figures
    raw_data = ml_state["processor"].get_customer_original_data(customer_index, ml_state["indices_test"])
    fig_cust = create_customer_profile_figure(raw_data)

    fig_shap_summary = ml_state["shap"].plot_summary()
    fig_pdp = ml_state["pdp"].plot_pdp_grid(ml_state["shap_top_features"][:4])
    fig_shap_local = ml_state["shap"].plot_waterfall(cust_actual_id=f"Index {customer_index}",
                                                     sample_idx=customer_index)
    fig_lime = ml_state["lime"].lime_bar_chart_plot(cust_actual_id=f"Index {customer_index}", sample_idx=customer_index)

    # 2. Convert to Base64
    b64_cust = fig_to_base64(fig_cust)
    b64_shap_summary = fig_to_base64(fig_shap_summary)
    b64_pdp = fig_to_base64(fig_pdp)
    b64_shap_local = fig_to_base64(fig_shap_local)
    b64_lime = fig_to_base64(fig_lime)

    # 3. Build HTML with Custom Order and Styling
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 20px; margin: 0 auto; max-width: 1200px; }}
            h1 {{ text-align: center; color: #1e3a8a; border-bottom: 4px solid #1e3a8a; padding-bottom: 10px; }}
            h2 {{ color: #ffffff; background-color: #3b82f6; padding: 10px 15px; border-radius: 5px; margin-top: 50px; }}
            h3 {{ color: #1e3a8a; border-bottom: 2px solid #cbd5e1; padding-bottom: 5px; margin-top: 30px; }}
            .plot-container {{ text-align: center; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>CNN-BiLSTM-HybridAttention Explainability Results Report</h1>
        <h2  style="text-align: center;">1. Global Explainability</h2>
        <h3>1.1. SHAP Summary Plot</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{b64_shap_summary}" alt="SHAP Summary">
        </div>

        <h3>1.2. Partial Dependence Plots (PDP)</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{b64_pdp}" alt="PDP">
        </div>

        <h2  style="text-align: center;">2. Local Explainability (Customer Specific)</h2>

        <h3>Customer Profile (Context)</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{b64_cust}" alt="Customer Profile">
        </div>

        <h3>2.1. SHAP Waterfall Plot</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{b64_shap_local}" alt="SHAP Waterfall">
        </div>

        <h3>2.2. LIME Bar Chart</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{b64_lime}" alt="LIME Bar Chart">
        </div>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/customers-by-class/{churn_class}")
def get_customers_by_class(churn_class: int):
    if "y_test" not in ml_state:
        raise HTTPException(status_code=500, detail="Labels not ready")
    # Find indices that match the ACTUAL class (0 = No Churn, 1 = Churn)
    indices = np.where(ml_state["y_test"] == churn_class)[0]

    # Limit to 150 rows for UI performance
    results = []
    for idx in indices[:150]:
        raw = ml_state["processor"].get_customer_original_data(int(idx), ml_state["indices_test"])
        customer_data = {"index": int(idx)}
        customer_data.update(raw)
        results.append(customer_data)

    return results


@app.get("/custom-ui", response_class=HTMLResponse)
def custom_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telecom Churn XAI Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; padding: 40px; margin: 0; }
            .header { 
                text-align: center; 
                margin-bottom: 40px; 
                padding: 50px 20px; 
                border-radius: 12px; 
                /* Dark blue gradient representing data/telecom */
                background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); 
                box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.5);
                position: relative;
                overflow: hidden;
            }
            
            /* Subtle data grid pattern */
            .header::before {
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background-image: 
                    linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
                background-size: 30px 30px;
                z-index: 1;
            }
            
            .header h1 { 
                color: #ffffff !important; 
                font-size: 36px; 
                margin-bottom: 10px; 
                font-weight: 700; 
                letter-spacing: 0.5px;
                position: relative;
                z-index: 2;
            }
            
            .header p { 
                color: #93c5fd !important; 
                font-size: 18px; 
                margin-top: 0; 
                font-weight: 500;
                position: relative;
                z-index: 2;
            }

            .container { max-width: 95%; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            h2 { color: #1e3a8a; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-top: 0; }
            .input-group { margin-bottom: 20px; margin-top: 20px; display: flex; align-items: center; gap: 15px; }
            label { font-weight: bold; color: #334155; font-size: 16px; }
            select { padding: 10px; font-size: 16px; border: 1px solid #cbd5e1; border-radius: 6px; width: 200px; cursor: pointer; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; border: none; border-radius: 6px; font-weight: bold; color: white; background-color: #3b82f6; }
            button:hover { background-color: #2563eb; }

            /* Table Container for Horizontal Scrolling */
            .table-wrapper { width: 100%; overflow-x: auto; margin-top: 20px; display: none; border: 1px solid #e2e8f0; border-radius: 8px; }
            table { width: 100%; border-collapse: collapse; white-space: nowrap; font-size: 14px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
            th { background-color: #f8fafc; color: #1e293b; font-weight: bold; position: sticky; top: 0; }
            tr:nth-child(even) { background-color: #f8fafc; }
            tbody tr { cursor: pointer; transition: background 0.2s; }
            tbody tr:hover { background-color: #e0f2fe; }

            /* Modal Popup Styling */
            .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); }
            
            /* Make the modal a flex column with a max height */
            .modal-content {background-color: #f8f9fa; margin: 5% auto; padding: 20px; border-radius: 10px; 
                width: 80%; max-width: 800px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); 
                display: flex; flex-direction: column; max-height: 85vh;}
            
            .modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
            .close-btn { color: #aaa; font-size: 28px; font-weight: bold; cursor: pointer; }
            .close-btn:hover { color: black; }
            
            /* Make the table container scrollable */
            .modal-body-scroll {overflow-y: auto; flex-grow: 1; margin-bottom: 15px; padding-right: 5px;}
            
            .btn-predict { background-color: #10b981; float: right; }
            .btn-predict:hover { background-color: #059669; }
                        /* --- New Modal Table Enhancements --- */
            #modalTable th {
                background-color: #1e3a8a !important; 
                color: #ffffff !important;            
                font-size: 16px !important;           
                text-transform: uppercase;            
                letter-spacing: 0.5px;
                padding: 15px 12px !important;
                border-bottom: 2px solid #0f172a !important;
            }

            #modalTable td {
                font-size: 16px !important;           
                padding: 14px 12px !important;        
                color: #1e293b !important;            
            }

            /* Make the Feature Names bold and distinct */
            #modalTable td:first-child {
                font-weight: 600 !important;
                color: #0f172a !important;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Telecom Customer Churn Prediction and Explainability</h1>
            <p>Powered by 1D CNN-BiLSTM Hybrid Attention Neural Network</p>
        </div>

        <div class="container">
            <h2>Customer Selection</h2>

            <div class="input-group">
                <label>Select customer type:</label>
                <select id="classSelect">
                    <option value="1">Churn</option>
                    <option value="0">No Churn</option>
                </select>
                <button onclick="fetchCustomersByClass()">OK</button>
            </div>

            <!-- Scrollable Wrapper for Table -->
            <div class="table-wrapper" id="tableWrapper">
                <table id="customerTable">
                    <thead id="tableHead">
                        <!-- Headers generated dynamically -->
                    </thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
        </div>

        <!-- Modal Popup for Customer Details -->
        <div id="customerModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Selected customer original profile data</h2>
                    <span class="close-btn" onclick="closeModal()">&times;</span>
                </div>
                
                <!-- NEW WRAPPER FOR SCROLLING -->
                <div class="modal-body-scroll">
                    <table id="modalTable">
                        <thead>
                            <tr><th>Feature Name</th><th>Original Value</th></tr>
                        </thead>
                        <tbody id="modalTableBody"></tbody>
                    </table>
                </div>
                
                <button class="btn-predict" id="predictBtn" onclick="openReport()">Predict this customer</button>
                <div style="clear:both;"></div>
            </div>
        </div>

        <script>
            let currentSelectedIndex = 0;

            async function fetchCustomersByClass() {
                const churnClass = document.getElementById('classSelect').value;
                const wrapper = document.getElementById('tableWrapper');
                const thead = document.getElementById('tableHead');
                const tbody = document.getElementById('tableBody');

                tbody.innerHTML = '<tr><td colspan="22">Loading...</td></tr>';
                thead.innerHTML = '';
                wrapper.style.display = 'block';

                try {
                    const response = await fetch(`/api/customers-by-class/${churnClass}`);
                    const data = await response.json();

                    if (data.length > 0) {
                        // 1. Dynamically Generate Headers
                        const keys = Object.keys(data[0]);
                        const headerRow = document.createElement('tr');
                        keys.forEach(key => {
                            const th = document.createElement('th');
                            th.innerText = key;
                            headerRow.appendChild(th);
                        });
                        thead.appendChild(headerRow);

                        // 2. Dynamically Generate Rows
                        tbody.innerHTML = '';
                        data.forEach((cust, i) => {
                            const row = document.createElement('tr');
                            // Keep the actual index for the backend, but display the sequential counter 'i'
                            row.onclick = () => showCustomerDetails(cust.index);
                            keys.forEach(key => {
                                const td = document.createElement('td');
                                if (key === 'index') {
                                    td.innerText = i; // Display continuous 0, 1, 2...
                                } else {
                                    td.innerText = cust[key];
                                }
                                row.appendChild(td);
                            });
                            tbody.appendChild(row);
                        });
                        
                    } else {
                        tbody.innerHTML = '<tr><td colspan="22">No customers found for this class.</td></tr>';
                    }
                } catch (error) {
                    tbody.innerHTML = '<tr><td colspan="22">Error loading data.</td></tr>';
                }
            }

            async function showCustomerDetails(index) {
                currentSelectedIndex = index;
                document.getElementById('customerModal').style.display = "block";

                const tbody = document.getElementById('modalTableBody');
                tbody.innerHTML = '<tr><td colspan="2">Loading...</td></tr>';

                try {
                    const response = await fetch(`/api/customer-data/${index}`);
                    const data = await response.json();

                    tbody.innerHTML = '';
                    for (const [key, value] of Object.entries(data)) {
                        tbody.innerHTML += `<tr><td>${key}</td><td>${value}</td></tr>`;
                    }
                } catch (error) {
                    tbody.innerHTML = '<tr><td colspan="2">Error loading details.</td></tr>';
                }
            }

            function closeModal() {
                document.getElementById('customerModal').style.display = "none";
            }

            function openReport() {
                window.open(`/full-report/${currentSelectedIndex}`, '_blank');
            }

            // Close modal if clicking outside
            window.onclick = function(event) {
                const modal = document.getElementById('customerModal');
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_api:app", host="0.0.0.0", port=8000)