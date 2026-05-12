"""
predictor.py

Provides a unified interface to predict traffic flow
using the trained XGBoost model.

Author: Member 3
"""

import pandas as pd
import joblib
import os
import numpy as np

from ml_models.third_model import ThirdModel


class TrafficPredictor:
    def __init__(self, timesteps=12):
        """
        Initialize predictor and load trained XGBoost model.
        """

        self.timesteps = timesteps
        self.trained = False

        self.model = ThirdModel()
        self.model.load_model()

        # Load scaler
        scaler_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "data",
                "processed",
                "scaler.pkl"
            )
        )

        self.scaler = joblib.load(scaler_path)

        # Historical traffic data
        self.historical_data = None

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

    def _extract_features(self, site_id, date, time, location):
        """
        Extract 12 previous normalized traffic flow values.
        """

        if self.historical_data is None:
            raise ValueError("Historical traffic data not loaded.")

        # Combine date + time
        target_datetime = f"{date} {time}:00"

        # Search matching key
        matching_key = None

        for key in self.historical_data.keys():
            scats_id, dt, loc = key

            if (
                scats_id == site_id and
                loc == location and
                str(dt) == target_datetime
            ):
                matching_key = key
                break

        if matching_key is None:
            raise ValueError(
                f"No matching traffic data for site {site_id}"
            )

        # Get flow sequence
        flow_values = self.historical_data[matching_key]

        # Ensure exactly 12 timesteps
        flow_values = flow_values[:self.timesteps]

        # Convert to numpy
        features = np.array(flow_values).reshape(1, -1)

        # Normalize
        features = self.scaler.transform(features)

        return features

    def predict_flow(self, site_id, time, date="2006-10-10"):
        """
        Predict traffic flow using historical sequence data.
        """

        if self.historical_data is None:
            raise ValueError("Historical traffic data not loaded.")

        # Find matching location from dictionary keys
        location = None

        for key in self.historical_data.keys():
            scats_id, dt, loc = key

            if scats_id == site_id:
                location = loc
                break

        if location is None:
            raise ValueError(f"No data found for site {site_id}")

        X = self._extract_features(
            site_id,
            date,
            time,
            location
        )

        prediction = self.model.predict(X)

        return float(prediction[0])