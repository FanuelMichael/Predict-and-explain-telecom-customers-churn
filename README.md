# Six-Week Project Plan: Telco Customer Churn Prediction

## 1. Project Overview
This plan describes the implementation workflow for the Telcom Customer Churn prediction project. The six-week schedule below shows the seven Gantt-chart task names in sequential order.

## 2. Six-Week Gantt Chart

The solid color bar indicates the primary execution period for a core-task. A pale tint indicates continuation of tasks.
<img width="876" height="322" alt="image" src="https://github.com/user-attachments/assets/8df23c24-9d43-4773-94ce-c7585a5996a0" />

## 3. Week-by-Week Execution Plan
## Week 1: Data Foundations
**Focus:** Phase 1 and the first part of Phase 2

* Search open public telecom customer churn dataset.
* Load telecommunication customer dataset and verify expected columns and target labels.
* Inspect missing values and data types; convert numeric-features to numeric.
* Review class balance and feature distributions.
  
**Milestone:** Dataset will be loaded and data-quality checks will be also document; preprocessing pipeline will also ready.

## Week 2: Feature Engineering & Model Architecture
**Focus:** Finish Phase 2 and begin Phase 3
* Split rows into stratified training, and test partitions.
* Impute missing values (fit on training set only)
* Preporocess(Impute missing values,One-Hot Encode categorical variables).
* StandardScaler (fit and transform in train data).
* Apply SMOTE to the training partition only; keep validation/test data untouched.
* Prepare model tensors and build the parallel 1D CNN and BiLSTM branches with HybridAttention.
  
**Milestone:** Leakage-aware data partitions and a compiling hybrid architecture.

## Week 3: Model Training & Evaluation
**Focus:** Finish Phase 3 and execute Phase 4
* Train the hybrid model using the training partition.
* Tune model settings using validation performance.
* Select the classification threshold using validation predictions.
* Evaluate once on the untouched test-set and generate the confusion matrix, ROC curve, classification report.

**Milestone**: Trained model, and validation-selected threshold.

## Week 4: Baseline Comparison
**Focus**: Finish Phase 4 and execute Phase 5
* Train standalone CNN and BiLSTM baselines using the same training-only resampled data.
* Evaluate each baseline model on the same untouched test set.
* Compare Accuracy, Precision, Recall, F1-score, and ROC-AUC in a consistently formatted results table.

**Milestone:** Fair comparison of the hybrid model and baseline models.

## Week 5: Global & Local Explainability (XAI Part-1)
**Focus:** First part of Phase 6
* Initialize SHAP with a training reference sample and generate global feature-importance summaries supported by the implementation.
* Create PDP plots and show leading features.

**Milestone:** Global explanation figures using shap and pdp will be generated.

## Week 6: Local Explainability (XAI Part-2) & Dashboarding(result exporting)
**Focus:** Finish Phase 6 and execute Phase 7
* Generate local SHAP and LIME explanations for a selected customer.
* Collect explainable plots in the dashboard exporter.
* Apply consistent visual styling, run the pipeline end-to-end, and check output HTML, memory use, and error handling.

**Milestone:** End-to-end workflow and a styled HTML dashboard ready for demonstration.

## 4. Critical Path and Risk Management
<img width="967" height="218" alt="image" src="https://github.com/user-attachments/assets/83ccdd15-1788-4a77-800b-561942662b9d" />

