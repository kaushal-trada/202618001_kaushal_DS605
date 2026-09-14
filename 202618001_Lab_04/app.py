import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "airbnb_xgboost_model.joblib"


# Load final XGBoost model
model = joblib.load(MODEL_PATH)


# Page configuration
st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# Title
st.title("🏠 Airbnb Price Predictor")
st.write("Enter the Airbnb listing details to estimate the nightly price.")


# -----------------------------
# User Inputs
# -----------------------------

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
)

neighbourhood = st.text_input(
    "Neighbourhood",
    "Harlem"
)

room_type = st.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"]
)

latitude = st.number_input(
    "Latitude",
    value=40.7580,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    value=-73.9855,
    format="%.6f"
)

minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    value=2
)

number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    value=10
)

reviews_per_month = st.number_input(
    "Reviews Per Month",
    min_value=0.0,
    value=1.0
)

calculated_host_listings_count = st.number_input(
    "Host Listings Count",
    min_value=1,
    value=1
)

availability_365 = st.number_input(
    "Availability (days/year)",
    min_value=0,
    max_value=365,
    value=200
)

last_review = st.date_input(
    "Last Review Date"
)


# -----------------------------
# Feature Engineering
# -----------------------------

midtown_lat = 40.7580
midtown_lon = -73.9855

distance_to_midtown = np.sqrt(
    (latitude - midtown_lat) ** 2 +
    (longitude - midtown_lon) ** 2
)

has_reviews = int(number_of_reviews > 0)

log_number_of_reviews = np.log1p(number_of_reviews)

is_commercial_host = int(
    calculated_host_listings_count > 1
)

reference_date = pd.Timestamp("2019-12-31")

last_review_timestamp = pd.Timestamp(last_review)

days_since_last_review = (
    reference_date - last_review_timestamp
).days


# -----------------------------
# Create Input DataFrame
# -----------------------------

input_data = pd.DataFrame([{
    "neighbourhood_group": neighbourhood_group,
    "neighbourhood": neighbourhood,
    "latitude": latitude,
    "longitude": longitude,
    "distance_to_midtown": distance_to_midtown,
    "room_type": room_type,
    "minimum_nights": minimum_nights,
    "number_of_reviews": number_of_reviews,
    "reviews_per_month": reviews_per_month,
    "calculated_host_listings_count": calculated_host_listings_count,
    "availability_365": availability_365,
    "has_reviews": has_reviews,
    "log_number_of_reviews": log_number_of_reviews,
    "days_since_last_review": days_since_last_review,
    "is_commercial_host": is_commercial_host
}])


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Airbnb Price"):

    prediction_log = model.predict(input_data)

    predicted_price = np.expm1(prediction_log[0])

    st.success(
        f"Estimated Airbnb Price: ${predicted_price:,.2f} per night"
    )