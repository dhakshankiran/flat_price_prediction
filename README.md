# Flat Price Prediction

A machine-learning project based on the Housing Price Prediction dataset. The original Colab notebook explores housing features and trains regression models. This organized version provides a reusable training module and a simple Streamlit web interface.

## Project files

- `backend.py` — loads the dataset, preprocesses features, trains a Linear Regression pipeline, evaluates it, and saves the model.
- `frontend.py` — web form for entering property details and predicting a price.
- `run_app.py` — starts the web application.
- `requirements.txt` — Python dependencies.
- `Housing.csv` — dataset file you download separately.
- `linear_regression_model.pkl` — generated automatically after training.

## Dataset setup

1. Download the Housing Price Prediction dataset from Kaggle:  
   https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction
2. Extract the download and place `Housing.csv` in the same folder as `backend.py`.
3. Do not upload private credentials or unrelated personal data.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python run_app.py
```

The app opens in your browser. On first launch, it trains the model and saves `linear_regression_model.pkl`.

## Model evaluation

The training function reports R², MAE, and MSE on a held-out test set. Values depend on the dataset version and installed library versions.

## Notes

- The model is for educational demonstration and is not a professional property valuation.
- The saved model file is generated locally. It is not included in this starter package.
- The dataset is not included; download it from Kaggle and check its license/terms before redistribution.
