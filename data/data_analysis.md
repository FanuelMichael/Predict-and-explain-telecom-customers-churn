# Telcom Customer Churn - Exploratory Data Analysis

This page provides a summary of the exploratory data analysis (EDA) conducted on the Telcom Customer Churn dataset, highlighting key behavioral patterns across various features of the entire dataset.

<img width="1093" height="395" alt="image" src="https://github.com/user-attachments/assets/f405506c-7438-4862-b71d-6c28ce3ef9b7" />
Fig. 1. Overall Distributions and Correlations:

The dataset exhibits a significant class imbalance, with non-churned customers outnumbering churned ones, while senior citizens show a disproportionately higher churn rate. The correlation heatmap indicates a strong positive relationship between tenure and total charges, whereas monthly charges have a moderate correlation with total charges.

<img width="1093" height="395" alt="image" src="https://github.com/user-attachments/assets/25082b74-774e-461e-8061-119e77391eef" />
Fig. 2: Numerical Feature Densities by Churn:

Customers who churn tend to have significantly higher monthly charges but lower total charges and much shorter tenures, indicating that new customers with high bills are highly likely to leave. Retained customers, conversely, demonstrate longer tenures and accumulate higher total charges over time.

<img width="1093" height="390" alt="image" src="https://github.com/user-attachments/assets/4be547ba-d846-40a6-acba-f87c5937b057" />
Fig. 3: Demographic Distributions by Churn:

Demographic features like gender show minimal impact on churn, as the distribution is nearly equal across both categories. However, customers lacking a partner or dependents exhibit notably higher churn rates, suggesting that family commitments strongly influence customer retention.

<img width="1093" height="392" alt="image" src="https://github.com/user-attachments/assets/be32f3fb-11b8-48b4-b3cf-efb6816b3f8d" />
Fig. 4: Core Telecom Services by Churn:

Customers with Fiber optic internet service exhibit a significantly higher churn rate compared to those with DSL or no internet service at all. Having phone service or multiple lines does not appear to strongly deter churn, as a large portion of churned customers utilize these core services.

<img width="1093" height="399" alt="image" src="https://github.com/user-attachments/assets/e1d4102e-4650-4cfe-8f49-45619ec7d47b" />
Fig. 5: Security and Protection Services by Churn:

The absence of value-added security and protection services (Online Security, Online Backup, Device Protection) is strongly associated with higher customer churn. Subscribing to these specific services appears to significantly improve customer retention and loyalty.

<img width="1093" height="403" alt="image" src="https://github.com/user-attachments/assets/6b730a5d-f8c7-4ff6-adfc-84c2b4ff9aee" />
Fig. 6: Support and Streaming Services by Churn:

Similar to security services, the lack of Tech Support is a strong indicator of churn, with non-subscribers leaving at higher rates. Streaming services also show that non-subscribers churn more frequently, though the impact is less pronounced than that of technical support and security features.

<img width="1093" height="397" alt="image" src="https://github.com/user-attachments/assets/a6eabecd-3057-4814-a189-effec9b3d02d" />
Fig. 7: Contractual and Billing Payment Methods by Churn:

Customers on month-to-month contracts and those utilizing paperless billing exhibit the highest churn rates, whereas long-term contract holders (one or two years) are significantly more likely to stay. Regarding payment methods, customers using electronic checks churn at a notably higher rate compared to those using mailed checks, automatic bank transfers or credit cards.
