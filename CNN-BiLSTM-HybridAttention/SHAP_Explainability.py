import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import shap
import numpy as np
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

#================================================================================================================
class SHAPExplainer:
    def __init__(self, model, feature_names, X_train, X_test,max_test_samples=40,background_samples=50,
                 algorithm=""):
        self.model = model
        self.feature_names = feature_names

        # Convert 3D → 2D
        self.X_train_2d = self._to_2d(X_train)
        self.X_test_2d = self._to_2d(X_test[:max_test_samples])

        # 2. Subsample background for Gradient/Kernel explainer
        n_background = min(background_samples, len(self.X_train_2d))
        background_idx = np.random.choice(len(self.X_train_2d), n_background, replace=False)
        self.X_background = self.X_train_2d[background_idx]

        # 1. Choose faster explainer explicitly
        self.algorithm = algorithm
        self.explainer = self._create_explainer()

        # Compute SHAP values only for the (smaller) test subset
        self.shap_exp = self.explainer(self.X_test_2d)
        self.shap_vals = self.shap_exp.values
        self.expected_value = self.shap_exp.base_values

    def _to_2d(self, arr):
        return arr.reshape(arr.shape[0], arr.shape[2]) if arr.ndim == 3 else arr

    def _predict_fn(self, x):
        if x.ndim == 2 and len(self.model.input_shape) == 3:
            x3 = x.reshape(x.shape[0], 1, x.shape[1])
            return self.model.predict(x3, verbose=0).ravel()
        return self.model.predict(x, verbose=0).ravel()

    def _create_explainer(self):
        if self.algorithm == "gradient":
            # GradientExplainer (uses background samples, much faster than Kernel)
            return shap.GradientExplainer(self.model,self.X_background,batch_size=256)
        elif self.algorithm == "permutation":
            # 4. Permutation explainer - very fast, approximate
            return shap.Explainer(self._predict_fn, self.X_background, algorithm="permutation")
        else:  # 'auto'
            return shap.Explainer(self._predict_fn, masker=shap.maskers.Independent(self.X_background),algorithm="auto")

    def get_top_shap_features(self, top_n=None):
        shap_abs_mean = np.mean(np.abs(self.shap_vals), axis=0)
        sorted_idx = np.argsort(shap_abs_mean)[::-1]
        ordered_features = [self.feature_names[i] for i in sorted_idx]
        if top_n is not None:
            ordered_features = ordered_features[:top_n]
        return ordered_features

    # ---------------- SHAP global explainability-------------------
    # 1. shap summary plot-------------------
    def plot_summary(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(self.shap_vals,self.X_test_2d,feature_names=self.feature_names,
            max_display=12,show=False,rng=rng)
        ax.set_title("SHAP Summary (Global Importance)")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # 2.---------------- SHAP local explainability-------------------
    # 1. SHAP Waterfall Plot------
    def plot_waterfall(self,cust_actual_id='', sample_idx=0):
        plt.figure(figsize=(10, 7))
        explanation_to_plot = self.shap_exp[sample_idx]
        explanation_to_plot.feature_names = self.feature_names
        shap.plots.waterfall(explanation_to_plot,show=False)
        # Center align the plot
        plt.gca().set_position([0.3, 0.1, 0.6, 0.8])  # [left, bottom, width, height]
        plt.title(f"SHAP Waterfall Plot (customer id: {cust_actual_id})")
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def get_local_shap_values(self, sample_idx):
        """Return SHAP values and feature values for a single sample customer."""
        shap_vals = self.shap_vals[sample_idx]
        feature_vals = self.X_test_2d[sample_idx]
        return shap_vals, feature_vals