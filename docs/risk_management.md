# Telcom Customer Churn - Risk Management Plan

## 1. Methodology and Risk Appetite
This document outlines the risk management strategy for the Telco Customer Churn prediction project. Risks are evaluated using a standard **Probability (P) × Impact (I) = Score (S)** matrix, where P and I are scored from 1 to 5.

## Probability x Impact Matrix

<img width="1118" height="385" alt="image" src="https://github.com/user-attachments/assets/3ea8c97d-2a4a-4a4f-8f25-0422949c829b" />

*   🔴 **Critical (15–25):** Act now. Requires immediate action plans.
*   🟠 **High (9–14):** Plan a response.
*   🟡 **Medium (4–8):** Watch and monitor.
*   🟢 **Low (1–3):** Accept and note.

**Risk Appetite:** The team accepts a model that only matches baselines (as the primary product is the explainability), but does *not* accept invalid results (data leakage), missed deadlines, or unsupported claims.


## 2. Risk Register table
Categories: Tech, People, Data, Market, Legal, Dependencies. P = probability, I = impact, S = score.
Responses: Avoid, Reduce, Transfer, Accept.
<img width="1242" height="803" alt="image" src="https://github.com/user-attachments/assets/09d07ce0-7301-4a79-aa80-89c5251f5b0c" />

## 3. Detailed Risk & Mitigation Strategies

### R1 - Explainability Runtime
*   **Category:** Tech | **Score:** 12 (High) | **Response:** Reduce | **Owner:** XAI Lead
*   **Mitigation:** SHAP and LIME may be expensive for neural networks. Start with explanation samples, validate outputs, and increase sample sizes only if runtime permits.

### R2 - Data Leakage
*   **Category:** Data | **Score:** 12 (High) | **Response:** Reduce | **Owner:** Data Lead
*   **Mitigation:** Split first, fit transforms on training only, tune on validation, and reserve test data for final reporting.

### R3 - Class Imbalance
*   **Category:** Data | **Score:** 12 (High) | **Response:** Reduce | **Owner:** Data Lead
*   **Rationale:** The churn dataset is typically imbalanced (roughly 73/27). This biases the classifier toward the majority class and makes accuracy misleading.
*   **Mitigation:** Apply SMOTE only to training data and keep test class distributions unchanged.

### R4 - Memory Pressure
*   **Category:** Tech | **Score:** 9 (High) | **Response:** Reduce | **Owner:** Tech Lead
*   **Rationale:** Multiple plots and explainers are generated sequentially in a single run. A crash can lose a long run and delay analysis.
*   **Mitigation:** Close plots after saving, avoid clearing the active model session before explanations are finished, and run expensive explainers sequentially.

### R5 - Dashboard Consistency
*   **Category:** Tech | **Score:** 8 (Medium) | **Response:** Reduce | **Owner:** XAI Lead
*   **Rationale:** The export logic is simple, but list lengths can drift when a figure is skipped or a `None` is returned, leading to misinformation (mislabelled plots).
*   **Mitigation:** Keep the plots list and plot-name list synchronized. Use safe fallback names, and verify that every plot appears exactly once in the final exported report.

### R6 - Hybrid Gains Nothing Over Baseline
*   **Category:** Tech | **Score:** 16 (Critical) | **Response:** Reduce | **Owner:** Model Lead
*   **Rationale:** It is plausible that a hybrid 1D-CNN + BiLSTM + attention does not outperform standalone baselines. The hybrid's added value is the core claim of the thesis.
*   **Mitigation:** Run CNN and BiLSTM baselines under identical splits, scaling, and SMOTE. Add an ablation without attention, report effect size and confidence intervals; if no meaningful gain, present the hybrid as equivalent rather than superior.

### R7 - Deep Models Overfit / CV Looks Better Than Real Performance
*   **Category:** Tech | **Score:** 12 (High) | **Response:** Reduce | **Owner:** Model Lead
*   **Rationale:** Deep networks overfit small tabular datasets easily. Overfitting inflates CV scores but is detectable on the untouched test set.
*   **Mitigation:** Use EarlyStopping on `val_f1`, dropout, L2, and BatchNormalization. Evaluate only on the untouched test set, report the OOF-vs-test gap, avoid tuning on test data, and consider repeated stratified CV.

### R8 - Data Leakage (Fitting Before Split)
*   **Category:** Data | **Score:** 12 (High) | **Response:** Reduce | **Owner:** Data Lead
*   **Rationale:** Fitting transforms on the full dataset leaks information and inflates scores. This mistake is easy to make during preprocessing refactors.
*   **Mitigation:** Enforce split-first; fit imputer, encoder, and scaler inside each CV fold on fold-train only; apply SMOTE only to fold-train; keep the test set raw until final evaluation.

### R9 - SHAP Unreliable/Unstable Top Drivers
*   **Category:** Tech | **Score:** 12 (High) | **Response:** Reduce | **Owner:** XAI Lead
*   **Rationale:** Neural-network explainers can be sensitive to background samples and seeds. Unstable top drivers undermine the "top 3 reasons" used in the report.
*   **Mitigation:** Use permutation or gradient SHAP with a fixed background sample and fixed seed; cap explanation samples; run stability checks across seeds/subsamples; cross-check top drivers with LIME and PDP.

### R10 - Missing Plan / Deadline Risk
*   **Category:** People | **Score:** 16 (Critical) | **Response:** Reduce | **Owner:** Team Lead
*   **Rationale:** A missing plan is a common failure mode in deadline-driven projects. It threatens the report, the defence date, and deliverable completion.
*   **Mitigation:** Create a milestone plan with owners and dates; track Must/Should/Could items weekly.

### R11 - Compute Limits
*   **Category:** Tech | **Score:** 12 (High) | **Response:** Transfer | **Owner:** Model Lead
*   **Rationale:** Hybrid training plus tuning can exceed laptop/free-cloud budgets, blocking experiments and model selection.
*   **Mitigation:** Move training to free cloud/Colab; reduce hyperparameter search space; use callbacks early stopping ReduceLROnPlateau and a smaller epoch/batch; run expensive explainers sequentially.

### R12 - Scope Creep
*   **Category:** People | **Score:** 9 (High) | **Response:** Avoid | **Owner:** Team Lead
*   **Rationale:** Late feature requests are common in academic projects and can push out Must-have deliverables.
*   **Mitigation:** Maintain a MoSCoW list; Differentiate non-Must features to future work.

### R13 - Dataset Availability/Licence Issues
*   **Category:** Data | **Score:** 9 (High) | **Response:** Reduce | **Owner:** Data Lead
*   **Rationale:** The Telco churn dataset is well-known, but licences and structural suitability are not fully guaranteed. A mid-project loss would be severe.
*   **Mitigation:** Use other open public website dataset.

### R14 - Data Loss
*   **Category:** Tech | **Score:** 10 (High) | **Response:** Reduce | **Owner:** Deploy Lead
*   **Rationale:** Local failures happen and data loss would block training, evaluation, and reporting.
*   **Mitigation:** Back up dataset in multi-alternatives drives.

### R15 - Compliance & Privacy
*   **Category:** Legal | **Score:** 3 (Low) | **Response:** Accept | **Owner:** Data Lead
*   **Rationale:** The pipeline drops `customerID` and works with a public dataset, making the probability low. However, leaked personal identifiers would create a compliance issue.
*   **Mitigation:** Exclude sensitive customer details (full name, national ID, phone, etc.) from exported reports, or ensure the dataset itself contains no such details.
