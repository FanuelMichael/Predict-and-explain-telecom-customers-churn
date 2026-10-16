#  Telcom Customer Churn Prediction - Project Plan
#  1. Project Overview
This plan describes the implementation workflow for the Telcom Customer Churn prediction project. The schedule below spans from October 16, 2026, to December 20, 2026, and outlines the seven Gantt-chart task names in sequential order.

#  2. Gantt Chart
The solid color bar (●) indicates the primary execution period for a core task. A pale tint (>) indicates continuation or overlapping of tasks.
<img width="1345" height="482" alt="image" src="https://github.com/user-attachments/assets/9920a291-d881-43a0-9d15-2e698817ba6e" />

#  3. Phase-by-Phase Execution Plan
#  Phase 1 & 2: Data Foundations (Oct 16 – Oct 23)
Focus Phase 1 and the first part of Phase 2
* Search open public telecom customer churn dataset.
* Load telecommunication customer dataset and verify expected columns and target labels.
* Inspect missing values and data types; convert numeric features to numeric.
* Review class balance and feature distributions.

Milestone: Dataset will be loaded and data-quality checks will be documented; preprocessing pipeline will also be ready.

#  Phase 2 & 3: Feature Engineering & Model Architecture (Oct 24 – Oct 30)
Focus: Finish Phase 2 and begin Phase 3
* Split rows into stratified training, and test partitions.
* Impute missing values (fit on training set only).
* Preprocess (Impute missing values, One-Hot Encode categorical variables).
* StandardScaler (fit and transform in train data).
* Apply SMOTE to the training partition only; keep validation/test data untouched.
* Prepare model tensors and build the parallel 1D CNN and BiLSTM branches with HybridAttention.

Milestone: Leakage-aware data partitions and a compiling hybrid architecture.

#  Phase 3 & 4: Model Training & Evaluation (Oct 31 – Nov 06)
Focus: Finish Phase 3 and execute Phase 4
* Train the hybrid model using the training partition.
* Tune model settings using validation performance.
* Select the classification threshold using validation predictions.
* Evaluate once on the untouched test set and generate the confusion matrix, ROC curve, classification report.

Milestone: Trained model, and validation-selected threshold.

#  Phase 4 & 5: Baseline Comparison (Nov 07 – Nov 13)
Focus: Finish Phase 4 and execute Phase 5
* Train standalone CNN and BiLSTM baselines using the same training-only resampled data.
* Evaluate each baseline model on the same untouched test set.
* Compare Accuracy, Precision, Recall, F1-score, and ROC-AUC in a consistently formatted results table.

Milestone: Fair comparison of the hybrid model and baseline models.

#  Phase 6 (Part 1): Global Explainability (Nov 14 – Nov 20)
Focus: First part of Phase 6
* Initialize SHAP with a training reference sample and generate global feature-importance summaries supported by the implementation.
* Create PDP plots and show leading features.

Milestone: Global explanation figures using shap and pdp will be generated.

#  Phase 6 (Part 2) & 7: Local Explainability & Dashboarding (Nov 21 – Dec 20)
Focus: Finish Phase 6 and execute Phase 7
* Generate local SHAP and LIME explanations for a selected customer.
* Collect explainable plots in the dashboard exporter.
* Apply consistent visual styling, run the pipeline end-to-end, and check output HTML, memory use, and error handling.

Milestone: End-to-end workflow and a styled HTML dashboard ready for demonstration.
