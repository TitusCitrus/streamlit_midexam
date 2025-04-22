import streamlit as st
import numpy as np

# Load the machine learning model
model = joblib.load("rf_model.pkl")

def main():
    st.title("Hotel Booking Cancellation Predictor")

    # Input all features
    no_of_adults = st.number_input("Number of Adults", min_value=1, value=2)
    no_of_children = st.number_input("Number of Children", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
    type_of_meal_plan = st.number_input("Meal Plan (1-3)", min_value=1, max_value=3, value=1)
    required_car_parking_space = st.number_input("Car Parking (0 or 1)", min_value=0, max_value=1, value=0)
    room_type_reserved = st.number_input("Room Type (1-4)", min_value=1, max_value=4, value=1)
    lead_time = st.number_input("Lead Time", min_value=0, value=30)
    arrival_year = st.number_input("Arrival Year", min_value=2022, max_value=2025, value=2024)
    arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=6)
    arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=15)
    market_segment_type = st.number_input("Market Segment (1-3)", min_value=1, max_value=3, value=1)
    repeated_guest = st.number_input("Repeated Guest (0 or 1)", min_value=0, max_value=1, value=0)
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
    avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, value=100.0)
    no_of_special_requests = st.number_input("Special Requests", min_value=0, value=0)
    no_of_guests = no_of_adults + no_of_children

    if st.button("Make Prediction"):
        features = np.array([[
            no_of_adults,
            no_of_children,
            no_of_weekend_nights,
            no_of_week_nights,
            type_of_meal_plan,
            required_car_parking_space,
            room_type_reserved,
            lead_time,
            arrival_year,
            arrival_month,
            arrival_date,
            market_segment_type,
            repeated_guest,
            no_of_previous_cancellations,
            avg_price_per_room,
            no_of_special_requests,
            no_of_guests
        ]])
        prediction = model.predict(features)
        st.success(f"Prediction: {'Canceled' if prediction[0] == 1 else 'Not Canceled'}")

    if st.button("Test Case 1"):
        test_1 = np.array([[2, 1, 2, 3, 1, 1, 1, 20, 2024, 5, 10, 1, 0, 0, 150.0, 1, 3]])
        prediction = model.predict(test_1)
        st.info(f"Test Case 1: {'Canceled' if prediction[0] == 1 else 'Not Canceled'}")

    if st.button("Test Case 2"):
        test_2 = np.array([[4, 0, 1, 5, 2, 0, 2, 50, 2024, 7, 15, 2, 1, 1, 200.0, 2, 4]])
        prediction = model.predict(test_2)
        st.info(f"Test Case 2: {'Canceled' if prediction[0] == 1 else 'Not Canceled'}")

if __name__ == "__main__":
    main()
