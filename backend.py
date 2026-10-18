"""Training and prediction utilities for the Flat Price Prediction project."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Housing.csv"
MODEL_PATH = ROOT / "linear_regression_model.pkl"

CATEGORICAL_COLUMNS = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]

def load_data(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            "Housing.csv was not found. Download the Housing Price Prediction "
            "dataset from Kaggle, place Housing.csv beside backend.py, then run again."
        )
    return pd.read_csv(path)

def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = load_data(data_path)
    if "price" not in df.columns:
        raise ValueError("The dataset must contain a 'price' target column.")

    X = df.drop(columns=["price"])
    y = df["price"]

    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=50
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
    }
    joblib.dump(model, model_path)
    return model, metrics, list(X.columns)

def predict_price(values, model_path=MODEL_PATH):
    if not Path(model_path).exists():
        train_model(model_path=model_path)
    model = joblib.load(model_path)
    return float(model.predict(pd.DataFrame([values]))[0])
