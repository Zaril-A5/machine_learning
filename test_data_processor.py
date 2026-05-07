from source.data_processing.traffic_flow_loader import TrafficFlowLoader
from source.data_processing.data_processor import DataProcessor
import numpy as np
import os


def test_data_processor():
    loader = TrafficFlowLoader("data/raw/Scats Data October 2006.xls")
    loader.load_flow_data()

    processor = DataProcessor(loader, timesteps=12)

    print("Testing clean_data()...")
    df = processor.clean_data()
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  First 3 rows:\n{df.head(3)}")

    print("\nTesting preprocess_and_save()...")
    X_train, y_train, X_test, y_test = processor.preprocess_and_save("data/processed/")

    print(f"\n  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    print(f"\n  X_train min: {X_train.min():.4f}, max: {X_train.max():.4f}")
    print(f"  y_train min: {y_train.min():.4f}, max: {y_train.max():.4f}")
    print(f"  Sample X_train[0]: {X_train[0]}")
    print(f"  Sample y_train[0]: {y_train[0]:.4f}")

    print("\nData processor test completed")


if __name__ == "__main__":
    test_data_processor()
