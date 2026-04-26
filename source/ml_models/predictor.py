"""
predictor.py

Provides a unified interface to predict traffic flow
using the trained XGBoost model.

Author: Member 3
"""

import numpy as np
from datetime import datetime
from third_model import ThirdModel


class TrafficPredictor:
    def __init__(self):
        """
        Initialize predictor with XGBoost model.
        """
        self.model = ThirdModel()
        self.trained = False

    def train_model(self, X_train, y_train):
        """
        Train the model using dataset from Member 2.

        IMPORTANT (Member 2):
        - X_train must match the exact feature format used in prediction
        - Feature order MUST be consistent
        """
        self.model.train(X_train, y_train)
        self.trained = True

    def _extract_features(self, site_id, timestamp):
        """
        Convert input (site_id, timestamp) into model features.

        IMPORTANT!!! (Member 2 MUST UPDATE THIS):
        This function MUST match EXACTLY the features used in training.

        Current version is ONLY a placeholder.

        Member 2 needs to confirm:
        - What features are used? (e.g., hour, day, previous flow, etc.)
        - Feature order
        - Any scaling/normalization applied

        Example expected structure:
        features = [hour, day_of_week, prev_flow, site_id, ...]

        If this does not match training data, The Model predictions will be WRONG
        """

        # --- Convert timestamp ---
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(timestamp, "%H:%M")
            except ValueError:
                timestamp = datetime.fromisoformat(timestamp)

        hour = timestamp.hour

        # --- PLACEHOLDER FEATURES ---
        # REPLACE THIS WITH REAL FEATURES FROM DATASET
        features = [
            hour,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            site_id % 100,   # simple encoding (temporary)
            1.0              # placeholder value
        ]

        return np.array(features).reshape(1, -1)

    def predict_flow(self, site_id, timestamp):
        """
        Predict traffic flow for a given site and time.

        Parameters:
        - site_id (int)
        - timestamp (str or datetime)

        Returns:
        - predicted flow (float)
        """

        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        X = self._extract_features(site_id, timestamp)
        prediction = self.model.predict(X)

        return float(prediction[0])