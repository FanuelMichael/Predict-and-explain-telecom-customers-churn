import os, logging, warnings, gc
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from Data_Preprocessing import TelcoDataProcessor
from Model_Evaluation import plot_class_distribution
from Data_analysis import data_analysis
from Dashboard import export_plots_to_html


# ======================== Main Project Execution ========================
def main(data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv', random_state=42):

    print("===1. Telecom customer dataset analysis===:\n")
    data_analysis(file_path=data_path)

    all_figures = []

    # STEP 2: Data Loading & Row-wise Cleaning
    print("\n=== 2. Data preprocessing(Data Loading and Row-wise Cleaning) ===")
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

    
    # STEP 3: Class Distribution Visualisation
    print("\n=== 3. Class Distribution Visualisation ===")
    non_churn, churn = int((y_train_full == 0).sum()), int((y_train_full == 1).sum())
    before_sampling = {"Non-Churn": non_churn, "Churn": churn}
    after_sampling = {"Non-Churn": max(non_churn, churn), "Churn": max(non_churn, churn)}
    all_figures.append(plot_class_distribution(before_sampling, after_sampling))

    export_plots_to_html(all_figures)
    del all_figures
    K.clear_session(); gc.collect()


if __name__ == "__main__":
    main()
