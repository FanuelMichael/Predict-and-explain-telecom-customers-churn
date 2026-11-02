import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.inspection import PartialDependenceDisplay

#=========================================================================================================
class _ModelWrapper(ClassifierMixin,BaseEstimator):
    _estimator_type = "classifier"
    def __init__(self, keras_model, batch_size=256):
        self.keras_model = keras_model
        self.batch_size = batch_size

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        # Use the exact attribute name Scikit-Learn expects for fitted validation
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, attributes=['is_fitted_'])

        # Safely reshape 2D sklearn input back to 3D Keras input (samples, features, 1)
        if X.ndim == 2:
            X_keras = X.reshape((X.shape[0], X.shape[1], 1))
        else:
            X_keras = X

        # Utilize Keras's built-in batch processing
        p = self.keras_model.predict(X_keras, batch_size=self.batch_size, verbose=0)
        p = np.asarray(p).reshape(p.shape[0], -1)   # Flatten to (n, 1)
        p = p[:, -1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class PDP_Explainer:
    def __init__(self, model, feature_names, X_train, X_test):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.X_test = X_test

        # Convert (samples, 1, features) → (samples, features)
        self.X_train_2d = self._to_2d(X_train)
        self.X_test_2d = self._to_2d(X_test)

        # Wrap model for sklearn PDP
        self.model_wrapper = self._wrap_model()
        # Fit wrapper once with dummy labels
        dummy_y = np.random.randint(0, 2, len(self.X_train_2d))
        self.model_wrapper.fit(self.X_train_2d, dummy_y)

    # Utility: reshape 3D → 2D
    def _to_2d(self, arr):
        if arr.ndim == 3:
            if arr.shape[1] == 1:
                return arr.reshape(arr.shape[0], arr.shape[2])
            elif arr.shape[2] == 1:
                return arr.reshape(arr.shape[0], arr.shape[1])
            else:
                return arr.reshape(arr.shape[0], -1)
        return arr

    # Model Wrapper for sklearn PDP
    def _wrap_model(self, batch_size=256):
        return _ModelWrapper(self.model, batch_size)

    # Simple fallback PDP ------------------------------
    def _plot_simple_pdp(self, feature, ax):
        idx = self.feature_names.index(feature)
        X_sub = self.X_test_2d[:100]
        vals = np.unique(X_sub[:, idx])
        if len(vals) > 20:
            vals = np.percentile(vals, np.linspace(0, 100, 20))
        pdp_vals = []
        for v in vals:
            X_temp = X_sub.copy()
            X_temp[:, idx] = v
            X3 = X_temp.reshape(X_temp.shape[0], X_temp.shape[1], 1)
            preds = self.model.predict(X3, verbose=0).ravel()
            pdp_vals.append(np.mean(preds))
        ax.plot(vals, pdp_vals, 'b-', linewidth=2)
        ax.scatter(vals, pdp_vals, color='red', s=40)
        ax.set_xlabel(feature)
        ax.set_ylabel("Average prediction")
        ax.grid(True, alpha=0.3)

    # 1. GRID Partial Dependence Plot(PDP)--------------------------------------------------------------------
    def plot_pdp_grid(self, top_features):
        valid = [f for f in top_features if f in self.feature_names]
        if not valid:
            print("No valid features for partial dependence plot.")
            return

        n = len(valid)
        rows = int(np.ceil(n / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(9, 5 * rows))
        axes = axes.flatten()

        for i, feature in enumerate(valid):
            ax = axes[i]
            idx = self.feature_names.index(feature)

            try:
                PartialDependenceDisplay.from_estimator(self.model_wrapper, self.X_test_2d, [idx],
                            feature_names=self.feature_names, ax=ax, kind='average',subsample=100)
                ax.set_title(f"PDP: {feature}")
            except Exception as e:
                print(f"PDP failed for '{feature}': {e}")
                try:
                    self._plot_simple_pdp(feature, ax)
                    ax.set_title(f"PDP Analysis: {feature}")
                except:
                    ax.axis("off")
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.suptitle("Partial Dependence Plots", fontsize=12)
        plt.tight_layout()
        #plt.close(fig)
        return fig
