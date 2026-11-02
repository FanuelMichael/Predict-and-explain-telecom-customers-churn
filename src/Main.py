import os, logging, warnings, gc
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from Data_Preprocessing import TelcoDataProcessor
from Hybrid_DeepLearning_Model import HybridModel
from Model_Evaluation import plot_evaluation,plot_class_distribution
from Data_analysis import data_analysis
from Dashboard import export_plots_to_html

# =========================================================method calls=======================================================================================
def get_callbacks(patience=3):
    return [EarlyStopping(monitor='val_f1', mode='max', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_f1', mode='max', factor=0.5, patience=patience, min_lr=1e-6, verbose=0)]

def to_3d(X):
    return X.reshape(X.shape[0], X.shape[1], 1)

def prep_leakage_safe_data(X_train_raw, y_train, X_test_raw, val_size, processor, numerical_cols):
    """Handles split, fold-safe preprocessing, scaling, and SMOTE for baseline & deployment models."""
    # 1. Split
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(X_train_raw, y_train, test_size=val_size, random_state=42,
                                                      stratify=y_train)
    # 2. Preprocess (Fit on train, transform on val/test)
    state = processor.make_preprocessing_state()
    X_tr_df, state = processor.fit_transform_preprocessing(X_tr_raw, state)
    X_va_df = processor.transform_preprocessing(X_va_raw, state)
    X_te_df = processor.transform_preprocessing(X_test_raw, state)

    # 3. Convert to numpy arrays
    X_tr = X_tr_df.values.astype(np.float32)
    X_va = X_va_df.values.astype(np.float32)
    X_te = X_te_df.values.astype(np.float32)

    # 4. Scale
    num_idx = [state["feature_names"].index(c) for c in numerical_cols if c in state["feature_names"]]
    scaler = StandardScaler()

    X_tr[:, num_idx] = scaler.fit_transform(X_tr[:, num_idx])
    X_va[:, num_idx] = scaler.transform(X_va[:, num_idx])
    X_te[:, num_idx] = scaler.transform(X_te[:, num_idx])

    # 5. SMOTE (Train only)
    X_tr_res, y_tr_res, _ = processor.apply_smote(X_tr, y_tr)

    return X_tr_res, y_tr_res, X_va, y_va, X_te, state["feature_names"]


# ======================== Main Project Execution ========================================================================================================================
def main(data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv', random_state=42):

    print("===1. Telecom customer dataset analysis===:\n")
    data_analysis(file_path=data_path)

    all_figures = []

    # STEP 2: Data Loading & Row-wise Cleaning
    print("\n=== 2. Data Loading & Row-wise Cleaning ===")
    processor = TelcoDataProcessor(data_path)
    df_cleaned = processor.clean_data()
    y = df_cleaned['Churn'].values.astype(np.int32)
    X_df = df_cleaned.drop('Churn', axis=1)
    indices = df_cleaned.index.values

    X_train_df_raw, X_test_df_raw, y_train_full, y_test, _, indices_test = \
        train_test_split(X_df, y, indices, test_size=0.2, random_state=random_state, stratify=y)
    X_train_df_raw.to_csv('raw_train_split.csv', index=False)
    X_test_df_raw.to_csv('raw_test_split.csv', index=False)
    print(f"Raw training pool: {X_train_df_raw.shape}, Raw test set: {X_test_df_raw.shape}")

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlyCharge']

    # STEP 3: Class Distribution Visualisation
    print("\n=== 3. Class Distribution Visualisation ===")
    non_churn, churn = int((y_train_full == 0).sum()), int((y_train_full == 1).sum())
    before_sampling = {"Non-Churn": non_churn, "Churn": churn}
    after_sampling = {"Non-Churn": max(non_churn, churn), "Churn": max(non_churn, churn)}
    all_figures.append(plot_class_distribution(before_sampling, after_sampling))

    # ======================================Next-work begins from below============================================================

    # STEP 4: Stratified 5-Fold Cross-Validation
    print("\n=== 4. Stratified 5-Fold Cross-Validation Training ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_preds, test_preds_folds = np.zeros(len(X_train_df_raw)), np.zeros(len(X_test_df_raw))
    best_models, fold_states = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df_raw, y_train_full)):
        print(f"\n--- Training Fold {fold + 1} / 5 ---")
        X_tr_raw, X_va_raw = X_train_df_raw.iloc[train_idx].copy(), X_train_df_raw.iloc[val_idx].copy()
        y_tr, y_va = y_train_full[train_idx], y_train_full[val_idx]

        fold_state = processor.make_preprocessing_state()
        X_tr_df, fold_state = processor.fit_transform_preprocessing(X_tr_raw, fold_state)
        X_va_df = processor.transform_preprocessing(X_va_raw, fold_state)

        num_idx = [fold_state["feature_names"].index(c) for c in numerical_cols if c in fold_state["feature_names"]]
        X_tr, X_va = X_tr_df.values.astype(np.float32), X_va_df.values.astype(np.float32)

        scaler = StandardScaler()
        X_tr[:, num_idx], X_va[:, num_idx] = scaler.fit_transform(X_tr[:, num_idx]), scaler.transform(X_va[:, num_idx])
        X_tr_res, y_tr_res, _ = processor.apply_smote(X_tr, y_tr)

        model = HybridModel.build(input_shape=(X_tr_res.shape[1], 1), lr=0.0003)
        model.fit(X_tr_res, y_tr_res, epochs=30, batch_size=64, validation_data=(to_3d(X_va), y_va),
                  callbacks=get_callbacks(), verbose=1)

        oof_preds[val_idx] = model.predict(to_3d(X_va), verbose=0).ravel()

        X_te = processor.transform_preprocessing(X_test_df_raw, fold_state).values.astype(np.float32)
        X_te[:, num_idx] = scaler.transform(X_te[:, num_idx])
        test_preds_folds += model.predict(to_3d(X_te), verbose=0).ravel() / 5

        best_models.append(model)
        fold_states.append({"state": fold_state, "scaler": scaler, "feature_names": fold_state["feature_names"]})

    # STEP 5: Threshold tuning on OOF
    print("\n=== 5. Finding Optimal Threshold on OOF Predictions ===")
    thresh_scores = [(t, f1_score(y_train_full, (oof_preds >= t).astype(int))) for t in np.arange(0.1, 0.9, 0.01)]
    best_thresh, best_f1 = max(thresh_scores, key=lambda x: x[1])
    print(f"Optimal threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")

    # STEP 6: Evaluation on Test Set
    print("\n=== 6. Evaluating Ensemble Model on Untouched 20% Test Set ===")
    y_pred = (test_preds_folds >= best_thresh).astype(int)
    all_figures.append(plot_evaluation(y_test, test_preds_folds, y_pred))

    export_plots_to_html(all_figures)
    del all_figures
    K.clear_session(); gc.collect()


if __name__ == "__main__":
    main()
