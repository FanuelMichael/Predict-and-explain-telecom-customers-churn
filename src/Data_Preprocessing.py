import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


class TelcoDataProcessor:
    def __init__(self, file_path):
        self.customer_details = pd.read_csv(file_path)
        self.df = self.customer_details.copy()
        self.feature_names = []


    # Call once per fold, and once more for the final full-train fit.
    @staticmethod
    def make_preprocessing_state():
        return {"imputer": SimpleImputer(strategy="median"),
                "encoder": OneHotEncoder(handle_unknown="ignore", sparse_output=False),"feature_names": []}

    def clean_data(self):
        self.df.drop('customerID', axis=1, inplace=True)
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df.replace(['No internet service', 'No phone service'], 'No', inplace=True)

        service_cols = ["PhoneService", "MultipleLines", "InternetService","OnlineSecurity", "OnlineBackup",
                        "DeviceProtection","TechSupport", "StreamingTV", "StreamingMovies"]
        existing_services = [c for c in service_cols if c in self.df.columns]
        self.df["HasMultipleServices"] = (self.df[existing_services] == "Yes").sum(axis=1)

        yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines','OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection','TechSupport', 'StreamingTV','StreamingMovies','PaperlessBilling', 'Churn']
        for col in yes_no_cols:
            self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})
        self.df['gender'] = self.df['gender'].map({'Female': 1, 'Male': 0})

        self.df.to_csv('after_data_cleaning.csv', index=False)
        print("Shape after data cleaning: " + str(self.df.shape))
        return self.df

    # fit imputer + encoder on X_train_df only; return (df, state)
    def fit_transform_preprocessing(self, X_train_df, state=None):
        if state is None:
            state = self.make_preprocessing_state()

        X_train_df = X_train_df.copy()
        # 1. Impute TotalCharges (fit on train)
        X_train_df['TotalCharges'] = state['imputer'].fit_transform(X_train_df[['TotalCharges']]).ravel()
        # 2. AverageMonthlyCharge
        X_train_df["AverageMonthlyCharge"] = (X_train_df["TotalCharges"] / (X_train_df["tenure"] + 1))
        # 3. One-Hot Encode (fit on train)
        categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
        encoded = state['encoder'].fit_transform(X_train_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded,columns=state['encoder'].get_feature_names_out(categorical_cols),
                                  index=X_train_df.index,)

        X_train_df = X_train_df.drop(columns=categorical_cols)
        X_train_df = pd.concat([X_train_df, encoded_df], axis=1)

        state['feature_names'] = list(X_train_df.columns)
        self.feature_names = state['feature_names']
        return X_train_df, state

    # only transform using the GIVEN state (on test-dataset only) (no fitting)
    def transform_preprocessing(self, X_test_df, state):
        X_test_df = X_test_df.copy()

        X_test_df['TotalCharges'] = state['imputer'].transform(X_test_df[['TotalCharges']]).ravel()
        X_test_df["AverageMonthlyCharge"] = (X_test_df["TotalCharges"] / (X_test_df["tenure"] + 1))

        categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
        encoded = state['encoder'].transform(X_test_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded,columns=state['encoder'].get_feature_names_out(categorical_cols),
                     index=X_test_df.index,)

        X_test_df = X_test_df.drop(columns=categorical_cols) #Because they are already encoded in the above line code
        X_test_df = pd.concat([X_test_df, encoded_df], axis=1)

        # Force column order to match the training state
        X_test_df = X_test_df[state['feature_names']]
        return X_test_df

    @staticmethod
    def apply_smote(X_train, y_train, random_state=42):
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        X_train_res = X_train_res.astype(np.float32)
        y_train_res = y_train_res.astype(np.int32)
        X_train_res = X_train_res.reshape(X_train_res.shape[0], X_train_res.shape[1], 1)
        after_sampling = {"Non-Churn": int((y_train_res == 0).sum()),"Churn": int((y_train_res == 1).sum())}
        return X_train_res, y_train_res, after_sampling

    def get_customer_original_data(self, sample_idx, test_indices):
        original_idx = test_indices[sample_idx]
        return self.customer_details.iloc[original_idx].to_dict()