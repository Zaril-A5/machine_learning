"""
predictor.py

Provides a unified interface to predict traffic flow
using the trained XGBoost model.

Author: Member 3
"""
import pickle
import os
import numpy as np

from datetime import datetime
from ml_models.third_model import ThirdModel


class TrafficPredictor:
    def __init__(self):
        """
        Load trained XGBoost model.
        """

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

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.trained = True

        print("[OK] XGBoost model loaded successfully.")

    def train_model(self, X_train, y_train, scaler_path='data/processed/scaler.pkl'):
        """
        Train the model using dataset from Member 2.
        Also loads the scaler so _extract_features() can normalize correctly.

        Parameters:
        - X_train: feature data (shape: num_samples x timesteps)
        - y_train: target traffic flow values
        - scaler_path: path to saved MinMaxScaler from data processing
        """
        # Train the XGBoost model
        self.model.train(X_train, y_train)
        self.trained = True

        # Load the scaler (needed for _extract_features to normalize flow values)
        try:
            import joblib
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"Scaler not found at {scaler_path}")
            raise e

    def _extract_features(self, site_id, timestamp):
        """
        Convert site ID and timestamp into model features.

        Parameters:
        - site_id (int)
        - timestamp (str): format HH:MM

        Returns:
        - numpy array of features
        """

        from datetime import datetime
        import numpy as np

        # Convert timestamp string into datetime
        if isinstance(timestamp, str):
            timestamp = datetime.strptime(timestamp, "%H:%M")

        hour = timestamp.hour

        # Simple feature vector
        features = [
            hour,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            site_id % 100,
            1.0
        ]

        return np.array(features).reshape(1, -1)

    def predict_flow(self, site_id, timestamp):
        """
        Predict traffic flow for a site and time.
        """

        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        X = self._extract_features(site_id, timestamp)

        prediction = self.model.predict(X)

        return float(prediction[0])