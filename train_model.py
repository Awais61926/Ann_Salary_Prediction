"""
Train the ANN Salary Regression model.

Run this file after placing Churn_Modelling.csv beside this script:

    python train_model.py

It creates:
    salary_model.keras
    encoder.pkl
    scaler.pkl
    feature_order.pkl
    metadata.pkl
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember"
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
TARGET = "EstimatedSalary"
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Churn_Modelling.csv was not found. Put the assignment dataset "
            "in the same folder as train_model.py."
        )

    df = pd.read_csv(DATA_PATH)

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # The assignment specifically says not to let identifiers or the target
    # leak into X.
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

    X_raw = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(float).copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Fit preprocessing only on training data.
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )
    encoder.fit(X_train_raw[CATEGORICAL_FEATURES])

    scaler = StandardScaler()
    scaler.fit(X_train_raw[NUMERIC_FEATURES])

    def transform(frame):
        numeric = scaler.transform(frame[NUMERIC_FEATURES])
        categorical = encoder.transform(frame[CATEGORICAL_FEATURES])
        return np.hstack([numeric, categorical]).astype("float32")

    X_train = transform(X_train_raw)
    X_test = transform(X_test_raw)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    predictions = model.predict(X_test, verbose=0).ravel()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nFinal test-set results")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")

    model.save(BASE_DIR / "salary_model.keras")

    with open(BASE_DIR / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open(BASE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    encoded_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_order = NUMERIC_FEATURES + encoded_names

    with open(BASE_DIR / "feature_order.pkl", "wb") as f:
        pickle.dump(feature_order, f)

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoded_features": encoded_names,
        "feature_order": feature_order,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
    }

    with open(BASE_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("\nArtifacts saved. The Streamlit app can now use the new model.")


if __name__ == "__main__":
    main()
