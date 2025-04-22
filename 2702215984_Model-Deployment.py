import streamlit as st
import joblib
import pandas as pd
import numpy as np

class ModelInference:
    def __init__(self, model_path):
        # Load the model
        self.model = joblib.load(model_path)  # Loading the model
        if not hasattr(self.model, 'predict'):
            raise ValueError("The loaded object is not a valid model with a 'predict' method.")
        
    def predict(self, input_data):
        # Ensure the input data is in a DataFrame format
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])  # Convert dict to DataFrame
        
        if isinstance(input_data, pd.DataFrame):
            # Ensure the DataFrame is passed to predict
            prediction = self.model.predict(input_data)  # Make prediction with the model
            return prediction
        else:
            raise ValueError("Input data must be a DataFrame or dict.")


# Load the trained model with ModelInference
model_inference = ModelInference("rf_model.pkl")

def main():
    st.title("Hotel Booking Cancellation Prediction")

    # Inputs
    no_of_adults = st.number_input("Number of Adults", min_value=0, value=2)
    no_of_children = st.number_input("Number of Children", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
    type_of_meal = st.selectbox("Meal Plan", [0, 1, 2, 3])  # Replace with encoder values
    required_car_parking_space = st.selectbox("Car Parking", [0, 1])
    room_type = st.selectbox("Room Type", [1, 2, 3, 4, 5, 6, 7])
    lead_time = st.number_input("Lead Time", min_value=0, value=10)
    arrival_month = st.slider("Arrival Month", 1, 12, 6)
    arrival_date = st.slider("Arrival Date", 1, 31, 15)
    market_segment = st.selectbox("Market Segment", [0, 1, 2, 3, 4, 5, 6])
    repeated_guest = st.selectbox("Is Repeated Guest", [0, 1])
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Prev Bookings Not Canceled", min_value=0, value=0)
    avg_price = st.number_input("Average Price", min_value=0.0, value=100.0)
    no_of_special_requests = st.number_input("Special Requests", min_value=0, value=0)
    no_of_guests = st.number_input("Total Guests", min_value=1, value=2)

    # Prediction
    if st.button("Predict Cancellation"):
        features = {
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal": type_of_meal,
            "required_car_parking_space": required_car_parking_space,
            "room_type": room_type,
            "lead_time": lead_time,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment": market_segment,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price": avg_price,
            "no_of_special_requests": no_of_special_requests,
            "no_of_guests": no_of_guests
        }

        # Make the prediction using the ModelInference class
        prediction = model_inference.predict(features)

        # Display the result
        st.success(f"Prediction: {'Canceled' if prediction[0] == 1 else 'Not Canceled'}")

if __name__ == "__main__":
    main()
