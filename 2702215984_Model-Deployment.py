import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# =====================================
# 🩹 Patch DecisionTreeClassifier Class
# =====================================
# Add missing attribute if it doesn't exist
if not hasattr(DecisionTreeClassifier, "monotonic_cst"):
    setattr(DecisionTreeClassifier, "monotonic_cst", None)

# =====================================
# 🚀 Model Loader
# =====================================
class ModelInference:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

        # Defensive patching (older sklearn may fail during predict_proba)
        for estimator in getattr(self.model, 'estimators_', []):
            if not hasattr(estimator, "monotonic_cst"):
                estimator.monotonic_cst = None
            if not hasattr(estimator, "_support_missing_values"):
                estimator._support_missing_values = lambda X: False

    def predict(self, input_data):
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        if input_data.isnull().values.any():
            st.warning("Missing values detected. Filling with 0s.")
            input_data = input_data.fillna(0)

        for col in input_data.columns:
            if not pd.api.types.is_numeric_dtype(input_data[col]):
                raise ValueError(f"Feature '{col}' must be numeric.")

        return self.model.predict(input_data)

# =====================================
# 🧠 Streamlit App
# =====================================
def main():
    st.title("🏨 Hotel Booking Cancellation Predictor")

    features = {
        "no_of_adults": st.number_input("Number of Adults", min_value=0, value=2),
        "no_of_children": st.number_input("Number of Children", min_value=0, value=0),
        "no_of_guests": st.number_input("Total Guests", min_value=1, value=2),
        "no_of_weekend_nights": st.number_input("Weekend Nights", min_value=0, value=1),
        "no_of_week_nights": st.number_input("Week Nights", min_value=0, value=2),
        "type_of_meal": st.number_input("Meal Plan", min_value=1, max_value=5, value=1),
        "required_car_parking_space": st.number_input("Car Parking", min_value=0, max_value=1, value=0),
        "room_type": st.number_input("Room Type", min_value=1, max_value=7, value=1),
        "lead_time": st.number_input("Lead Time", min_value=0, value=224),
        "arrival_year": st.number_input("Arrival Year", min_value=2000, value=2017),
        "arrival_month": st.number_input("Arrival Month", min_value=1, max_value=12, value=10),
        "arrival_date": st.number_input("Arrival Date", min_value=1, max_value=31, value=2),
        "market_segment": st.number_input("Market Segment", min_value=1, max_value=5, value=2),
        "repeated_guest": st.number_input("Is Repeated Guest", min_value=0, max_value=1, value=0),
        "no_of_previous_cancellations": st.number_input("Previous Cancellations", min_value=0, value=0),
        "no_of_previous_bookings_not_canceled": st.number_input("Previous Bookings Not Canceled", min_value=0, value=0),
        "avg_price": st.number_input("Average Price", min_value=0.0, value=65.0),
        "no_of_special_requests": st.number_input("Special Requests", min_value=0, value=0),
    }

    if st.button("🧠 Predict Cancellation"):
        try:
            model_inference = ModelInference("rf_model.pkl")
            prediction = model_inference.predict(features)
            st.success(f"Prediction: {'❌ Canceled' if prediction[0] == 1 else '✅ Not Canceled'}")
        except Exception as e:
            st.error(f"🔥 Prediction Failed: {e}")

if __name__ == "__main__":
    main()
