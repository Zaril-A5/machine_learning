"""
third_model.py

XGBoost-based traffic prediction model.

Author: Member 3
"""
import joblib
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("XGBoost not installed. Run: pip install xgboost")


class ThirdModel:
    def __init__(self):
        """
        Initialize model (created during training).
        """
        self.model = None

    def train(self, X_train, y_train):
        """
        Train the XGBoost model.

        Parameters:
        - X_train: feature data
        - y_train: target traffic flow
        """
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1,  # use all CPU cores
            verbosity=0
        )

        self.model.fit(X_train, y_train)

    def load_model(self, model_path=None):
        """
        Load trained XGBoost model from disk.
        """

        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "saved_models",
                    "xgboost_model.pkl"
                )
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        self.model = joblib.load(model_path)

        print("[OK] XGBoost model loaded successfully.")

    def predict(self, X):
        """
        Predict traffic flow.

        Parameters:
        - X: input features

        Returns:
        - predicted traffic flow
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Returns:
        - MAE, RMSE, MAPE
        """
        predictions = self.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Prevent division by zero
        y_test_safe = np.where(y_test == 0, 1e-6, y_test)
        mape = np.mean(np.abs((y_test - predictions) / y_test_safe)) * 100

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }