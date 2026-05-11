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
    def __init__(self, timesteps=12):
        """
        Initialize predictor with XGBoost model.

        Parameters:
        - timesteps (int): number of previous flow values as features (default: 12)
        """
        self.model = ThirdModel()
        self.trained = False
        self.timesteps = timesteps
        self.scaler = None
        self.historical_data = {}

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
        Convert input (site_id, date, time, location) into model features.

        Returns 12 previous normalized flow values (shape: 1 x 12),
        matching the training format from data_processor.py.

        Parameters:
        - site_id (int): SCATS site ID
        - date (str): target date in format 'YYYY-MM-DD'
        - time (str): target time in format 'HH:MM'
        - location (str): specific location to filter by

        Returns:
        - numpy array of shape (1, 12) with normalized flow values
        """
        #  Convert date and time to datetime
        timestamp = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")

        #  Get previous flow values for this site 
        prev_flows = []

        if self.historical_data:
            # Find all records for this site
            site_records = []
            for (sid, date_str, loc), flows in self.historical_data.items():
                if sid == site_id and location.upper() in loc.upper():
                    for time_str, flow in flows.items():
                        # Parse time
                        hour, minute = map(int, time_str.split(':'))
                        try:
                            if ' ' in date_str:
                                dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                            else:
                                dt = datetime.strptime(date_str, '%Y-%m-%d')
                            dt = dt.replace(hour=hour, minute=minute, second=0)
                            site_records.append((dt, flow, location))
                        except ValueError:
                            continue

            # Sort by time and get flows before target timestamp
            site_records.sort(key=lambda x: x[0])
            prev_flows = [flow for dt, flow, location in site_records if dt < timestamp]

        #  Pad if not enough historical data 
        if len(prev_flows) < self.timesteps:
            # Pad with average flow value
            pad_value = 30.0
            padding_needed = self.timesteps - len(prev_flows)
            prev_flows = [pad_value] * padding_needed + prev_flows
            print(f"Insufficient historical data for site {site_id}. Padded {padding_needed} values with {pad_value}.")

        # Take only the last timesteps values
        prev_flows = prev_flows[-self.timesteps:]

        #  Normalize using MinMaxScaler
        if self.scaler is not None:
            flows_array = np.array(prev_flows).reshape(-1, 1)
            normalized = self.scaler.transform(flows_array).flatten()
        else:
            raise ValueError("Scaler not loaded. Call train_model() first to load the scaler.")

        return normalized.reshape(1, -1)

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