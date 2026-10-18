"""Streamlit user interface for Flat Price Prediction."""
import streamlit as st
from backend import train_model, predict_price, MODEL_PATH, DATA_PATH

st.set_page_config(page_title="Flat Price Prediction", page_icon="🏠")
st.title("🏠 Flat Price Prediction")
st.write("Estimate a house price using the Housing Price Prediction dataset.")

if not DATA_PATH.exists():
    st.warning(
        "Before using the app, place the dataset file `Housing.csv` in this project folder. "
        "See README.md for the download instructions."
    )
    st.stop()

if not MODEL_PATH.exists():
    with st.spinner("Training the prediction model for the first time..."):
        _, metrics, _ = train_model()
    st.success("Model trained and saved.")
else:
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        st.caption("Saved model loaded.")
    except Exception:
        with st.spinner("Preparing the model..."):
            _, metrics, _ = train_model()
        st.success("Model trained and saved.")

df_columns = [
    "area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom",
    "basement", "hotwaterheating", "airconditioning", "parking",
    "prefarea", "furnishingstatus"
]
with st.form("prediction_form"):
    st.subheader("Enter property details")
    area = st.number_input("Area (square feet)", min_value=100, value=3000, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    stories = st.number_input("Stories", min_value=1, max_value=10, value=2)
    mainroad = st.selectbox("Main road access", ["yes", "no"])
    guestroom = st.selectbox("Guest room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot water heating", ["yes", "no"])
    airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
    parking = st.number_input("Parking spaces", min_value=0, max_value=10, value=1)
    prefarea = st.selectbox("Preferred area", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "Furnishing status", ["furnished", "semi-furnished", "unfurnished"]
    )
    submitted = st.form_submit_button("Predict price")

if submitted:
    property_data = {
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "stories": stories, "mainroad": mainroad, "guestroom": guestroom,
        "basement": basement, "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning, "parking": parking,
        "prefarea": prefarea, "furnishingstatus": furnishingstatus,
    }
    try:
        prediction = predict_price(property_data)
        st.success(f"Estimated price: {prediction:,.2f}")
        st.caption("This is a model estimate, not a formal property valuation.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
