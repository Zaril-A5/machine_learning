import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pickle
import numpy as np

from third_model import ThirdModel


# -----------------------------
# LOAD TRAINING DATA
# -----------------------------

X_train = np.load("../data/processed/X_train.npy")
y_train = np.load("../data/processed/y_train.npy")

X_test = np.load("../data/processed/X_test.npy")
y_test = np.load("../data/processed/y_test.npy")

print("Training data loaded successfully.")

# -----------------------------
# TRAIN MODEL
# -----------------------------

model = ThirdModel()

print("\nTraining XGBoost model...\n")

model.train(X_train, y_train)

print("Training completed.")

# -----------------------------
# EVALUATE MODEL
# -----------------------------

metrics = model.evaluate(X_test, y_test)

print("\nEvaluation Results:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------

os.makedirs("saved_models", exist_ok=True)

MODEL_PATH = "saved_models/xgboost_model.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")