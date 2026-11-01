"""Streamlit frontend for the Wellness Tourism Package purchase model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


@st.cache_resource
def load_artifacts():
    """Load the model committed by the training job."""
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


def build_input_frame(values, feature_columns):
    """Store the form values in a one-row dataframe the model can score."""
    return pd.DataFrame([values], columns=feature_columns)


def main():
    st.set_page_config(page_title="Wellness Tourism Package", layout="wide")
    st.title("Wellness Tourism Package")
    st.write(
        "Score a customer before the sales team makes contact. "
        "The model estimates whether this customer will purchase the package."
    )

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("The trained model is not in models/. Run the training job first.")
        return

    model, metadata = load_artifacts()
    feature_columns = metadata["feature_columns"]

    with st.form("customer_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", min_value=18, max_value=80, value=36)
            typeof_contact = st.selectbox(
                "Type of contact",
                ["Self Enquiry", "Company Invited"],
            )
            city_tier = st.selectbox("City tier", [1, 2, 3], index=0)
            duration = st.number_input("Duration of pitch (minutes)", min_value=1.0, max_value=60.0, value=14.0)
            occupation = st.selectbox(
                "Occupation",
                ["Salaried", "Small Business", "Large Business", "Free Lancer"],
            )
            gender = st.selectbox("Gender", ["Male", "Female"])
            persons = st.number_input("Number of people visiting", min_value=1, max_value=5, value=3)
            followups = st.number_input("Number of follow-ups", min_value=1.0, max_value=6.0, value=4.0)
        with right:
            product = st.selectbox(
                "Product pitched",
                ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
            )
            property_star = st.selectbox("Preferred property star", [3.0, 4.0, 5.0], index=0)
            marital_status = st.selectbox(
                "Marital status",
                ["Single", "Unmarried", "Married", "Divorced"],
            )
            trips = st.number_input("Number of trips per year", min_value=1.0, max_value=10.0, value=3.0)
            passport = st.selectbox("Passport", [("No", 0), ("Yes", 1)], format_func=lambda item: item[0])
            pitch_score = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)
            own_car = st.selectbox("Owns a car", [("No", 0), ("Yes", 1)], format_func=lambda item: item[0])
            children = st.number_input("Children under 5 visiting", min_value=0.0, max_value=3.0, value=1.0)
            income = st.number_input("Monthly income", min_value=1000.0, max_value=100000.0, value=22418.0, step=100.0)
        submitted = st.form_submit_button("Predict purchase")

    if not submitted:
        st.info("Enter the customer details and run the prediction.")
        return

    values = {
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": int(city_tier),
        "DurationOfPitch": float(duration),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": int(persons),
        "NumberOfFollowups": float(followups),
        "ProductPitched": product,
        "PreferredPropertyStar": float(property_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(trips),
        "Passport": int(passport[1]),
        "PitchSatisfactionScore": int(pitch_score),
        "OwnCar": int(own_car[1]),
        "NumberOfChildrenVisiting": float(children),
        "MonthlyIncome": float(income),
    }
    input_frame = build_input_frame(values, feature_columns)
    probability = float(model.predict_proba(input_frame)[0, 1])
    prediction = int(model.predict(input_frame)[0])

    st.subheader("Customer inputs")
    st.dataframe(input_frame, use_container_width=True)
    st.subheader("Prediction")
    if prediction == 1:
        st.success(f"Likely to purchase. Estimated probability: {probability:.1%}.")
    else:
        st.warning(f"Unlikely to purchase. Estimated probability: {probability:.1%}.")
    st.caption(
        f"Model: {metadata['best_model']}. "
        f"Cross-validated ROC-AUC: {metadata['cv_roc_auc']:.3f}."
    )


if __name__ == "__main__":
    main()
