"""
data_processor.py - Preprocess traffic flow data for LSTM/GRU/XGBoost.

Author: Member 2
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from source.data_processing.traffic_flow_loader import TrafficFlowLoader


class DataProcessor:
    def __init__(self, flow_loader: TrafficFlowLoader, timesteps: int = 12):
        self.loader = flow_loader
        self.timesteps = timesteps
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def clean_data(self) -> pd.DataFrame:
        """
        Convert the nested flow_data dictionary into a flat DataFrame,
        sort by time, and remove any NaN rows.
        """
        records = []
        for (site, date, loc), flows in self.loader.flow_data.items():
            dt = pd.to_datetime(date)
            for interval, flow in flows.items():
                hour, minute = map(int, interval.split(':'))
                timestamp = dt.replace(hour=hour, minute=minute, second=0)
                records.append({
                    'site_id': site,
                    'timestamp': timestamp,
                    'flow': flow,
                    'location': loc
                })
        df = pd.DataFrame(records)
        df = df[df['flow'] >= 0].dropna()
        df = df.sort_values(['site_id', 'timestamp']).reset_index(drop=True)
        return df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each site, create sliding window sequences: X = past timesteps flows,
        y = next flow.
        Returns (X, y) as numpy arrays.
        """
        X_list, y_list = [], []
        for site_id, group in df.groupby('site_id'):
            flows = group['flow_norm'].values
            if len(flows) <= self.timesteps:
                continue
            for i in range(len(flows) - self.timesteps):
                X_list.append(flows[i:i+self.timesteps])
                y_list.append(flows[i+self.timesteps])
        return np.array(X_list), np.array(y_list)

    def preprocess_and_save(self, save_dir: str = 'data/processed/'):
        """
        load raw data, clean, normalise, create sequences,
        split train/test (80/20%), save everything.
        """
        print("Cleaning data...")
        df = self.clean_data()
        print(f"  Cleaned data shape: {df.shape}")

        print("Normalising flows...")
        flows_reshaped = df['flow'].values.reshape(-1, 1)
        df['flow_norm'] = self.scaler.fit_transform(flows_reshaped)

        print(f"Creating sequences (timesteps={self.timesteps})...")
        X, y = self.create_sequences(df)
        print(f"  Number of sequences: {X.shape[0]}")

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

        import joblib
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))

        print(f"\nProcessed data saved to {save_dir}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples:     {len(X_test)}")
        print(f"  Input shape:      {X_train.shape[1]} timesteps")
        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    loader = TrafficFlowLoader('data/raw/Scats Data October 2006.xls')
    loader.load_flow_data()
    processor = DataProcessor(loader, timesteps=12)
    processor.preprocess_and_save()
