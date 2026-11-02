# Telecom customers churn prediction and explainability results report

### ♦ Class distribution before and after SMOTE ♦

<img width="1408" height="698" alt="image" src="https://github.com/user-attachments/assets/f1b60abc-d260-4222-86a5-c1e744216a28" />
To address class imbalance without data leakage, the Synthetic Minority Over-sampling Technique (SMOTE) was applied exclusively to the training folds during cross-validation and deployment training, perfectly balancing the training distribution to a 50:50 ratio (4,139 non-churn and 4,139 churn samples).

### ♦ Performance metrics, confusion matrix and ROC curve of the hybrid model ♦
<img width="1408" height="652" alt="image" src="https://github.com/user-attachments/assets/4a00335c-ebd1-48b3-b71c-a633a9f8a6f7" />
<img width="1410" height="238" alt="image" src="https://github.com/user-attachments/assets/bb44451d-f0e3-417b-94a2-07128ce68630" />

### ♦ Classification Report and Confusion Matrix of baseline CNN ♦
<img width="1675" height="677" alt="image" src="https://github.com/user-attachments/assets/a21e72fc-fe9d-462f-87ca-454ca65f07c4" />

### ♦ Classification Report and Confusion Matrix of baseline BiLSTM ♦
<img width="1672" height="692" alt="image" src="https://github.com/user-attachments/assets/4a5fa25e-faae-49d6-ad12-9ed0fdea26d7" />

### ♦ Models performance comparison table ♦
<img width="1672" height="317" alt="image" src="https://github.com/user-attachments/assets/533eb9b8-388b-4317-a039-a376048d880a" />

The comparative evaluation metrics Table computed based on the confusion matrices of hybrid and baseline models, demonstrates that the hybrid architecture achieves superior overall classification performance: Accuracy 75.59%, Precision 52.68%, Recall 78.88%, F1-score 63.17%. The hybrid model outperforms both standalone CNN and BiLSTM baselines across primary predictive benchmarks, while maintaining equal top-tier discrimination power (AUC 0.84) alongside CNN.

<img width="1817" height="98" alt="image" src="https://github.com/user-attachments/assets/46008707-aa8a-478b-b457-73207dbee26a" />

### ♦ Global SHAP summary analysis ♦
<img width="937" height="737" alt="image" src="https://github.com/user-attachments/assets/66a359be-7c24-42b6-a6c8-02c8e886c56d" />
 The SHAP summary plot ranks features by global importance and illustrates their directional contributions to the model output. Tenure, MonthlyCharges, Contract_Month-to-month, Contract_Two year, and InternetService_Fiber optic emerge as top influential features. Low tenure values are associated with positive SHAP values, indicating increased predicted churn, whereas high tenure values generally contribute negatively. Similarly, high MonthlyCharges values and the presence of month-to-month contracts or fiber-optic internet service tend to push predictions toward churn. Additionally, lacking a two-year contract and having paperless billing push predictions toward churn. In contrast, the presence of a two-year contract, having dependents, and subscribing to online security or tech support generally contribute to lower predicted churn.

 ### ♦ Global Partial Dependence Plot(PDP) analysis ♦
<img width="1262" height="756" alt="image" src="https://github.com/user-attachments/assets/463933e3-a0cb-4c63-9f4a-48722517b297" />
<img width="1267" height="671" alt="image" src="https://github.com/user-attachments/assets/c39d8051-470c-4e76-b542-b4d3df0a259c" />
Longer tenure strongly reduces churn risk, whereas higher monthly charges increases attrition. Furthermore, being bound to a month-to-month contract consistently raises churn risk, while conversely, having a two-year contract significantly decreases the likelihood of churn. Overall, these marginal effects confirm that shorter tenure, higher costs, and flexible (month-to-month) contracts act as independent drivers that increase the probability of customer churn, whereas long-term commitments provide a protective effect.
<img width="1263" height="803" alt="image" src="https://github.com/user-attachments/assets/95d675d0-f695-4e5c-8200-e328f9cc8a32" />
<img width="1262" height="666" alt="image" src="https://github.com/user-attachments/assets/7b2b258d-2474-4d29-98cc-60841adca14f" />
Subscribing to fiber-optic internet and opting for paperless billing both increase the probability of churn. In contrast, having dependents strongly reduces churn risk. Furthermore, the effect of having multiple services is non-linear; churn risk peaks for customers with a few services (around 2 to 3) but steadily decreases as the total number of subscribed services grows toward 8. Overall, these marginal effects confirm that specific service choices like fiber-optic internet and paperless billing act as independent drivers that increase the probability of customer churn, whereas having dependents and subscribing to a high volume of services provide a protective, stabilizing effect.
<img width="1841" height="76" alt="image" src="https://github.com/user-attachments/assets/ce0f9189-40ac-4f09-b017-cb7809a28217" />

### ♦ Selected customer original file for local explainability ♦
<img width="1231" height="863" alt="image" src="https://github.com/user-attachments/assets/5b198d1c-e663-458d-9a45-4a21c2a68f1f" />

The table above details the raw profile of customer 2034-GDRCN, who was selected to demonstrate local (customer-level) explainability using SHAP waterfall and LIME bar chart plots. This customer represents a classic high-risk churn profile: a female with a very short tenure (2 months) on a month-to-month contract, paying a relatively high monthly charge ($90.4) for Fiber optic internet without any tech support or online security. We selected, the actual churned customer file. We will see next how SHAP, and LIME explainable tools dissect this specific customer model prediction, by highlighting exactly what features are pushed the hybrid model's decision toward a churn probability.

## ♦ Local SHAP waterfall plot explainability ♦
<img width="1240" height="925" alt="image" src="https://github.com/user-attachments/assets/856605c3-be76-4c17-8811-0c0f57b1296a" />

The SHAP waterfall plot illustrates the local explainability for customer 2034-GDRCN, detailing exactly how the hybrid model reached its final prediction. Starting from the model's base expected value of 0.475, the final predicted churn probability increases to 0.577, successfully crossing the decision threshold to classify this customer as a churner.

The red bars indicate features that increased the churn risk. The most significant driver was the customer's short tenure (-1.241), which alone added +0.03 to the probability. Secondary factors, including high MonthlyCharges (0.845), a Month-to-month contract, Fiber optic internet, and paying via Electronic check, each contributed +0.01 to the risk. The only feature that slightly reduced the prediction (blue bar) was the absence of MultipleLines.

The SHAP values confirm that the model is correctly identifying a classic high-risk profile: a new customer locked into a flexible contract with high monthly costs.

## ♦ SHAP LIME plot explainability ♦
<img width="1245" height="897" alt="image" src="https://github.com/user-attachments/assets/976c7693-168d-444e-b93a-d62a3277ee3c" />

