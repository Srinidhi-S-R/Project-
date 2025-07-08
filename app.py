import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Load trained model and scaler ----------
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ---------- Manual encoding maps (same as training) ----------
substance_map = {
    'carbon dioxide': 0,
    'methane': 1,
    'nitrous oxide': 2,
    'other GHGs': 3
}

unit_map = {
    'kg/2018 USD, purchaser price': 0,
    'kg CO2e/2018 USD, purchaser price': 1
}

source_map = {
    'Commodity': 0,
    'Industry': 1
}

# ---------- Streamlit App Title ----------
st.title("Supply Chain Emissions Prediction")

st.markdown("""
This app predicts **Supply Chain Emission Factors with Margins** based on DQ metrics and other parameters.
""")

# ---------- Form Inputs ----------
with st.form("prediction_form"):
    substance = st.selectbox("Substance", list(substance_map.keys()))
    unit = st.selectbox("Unit", list(unit_map.keys()))
    source = st.selectbox("Source", list(source_map.keys()))
    supply_wo_margin = st.number_input("Supply Chain Emission Factors without Margins", min_value=0.0)
    margin = st.number_input("Margins of Supply Chain Emission Factors", min_value=0.0)
    dq_reliability = st.slider("DQ Reliability", 0.0, 1.0)
    dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 1.0)
    dq_geo = st.slider("DQ Geographical Correlation", 0.0, 1.0)
    dq_tech = st.slider("DQ Technological Correlation", 0.0, 1.0)
    dq_data = st.slider("DQ Data Collection", 0.0, 1.0)

    submit = st.form_submit_button("Predict")

# ---------- Prediction Logic ----------
if submit:
    try:
        # Apply manual encoding
        input_data = {
            'Substance': substance_map[substance],
            'Unit': unit_map[unit],
            'Supply Chain Emission Factors without Margins': supply_wo_margin,
            'Margins of Supply Chain Emission Factors': margin,
            'DQ ReliabilityScore of Factors without Margins': dq_reliability,
            'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
            'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
            'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
            'DQ DataCollection of Factors without Margins': dq_data,
            'Source': source_map[source]
        }

        input_df = pd.DataFrame([input_data])

        # Scale features
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)

        # Show result
        st.success(f"✅ Predicted Supply Chain Emission Factor with Margin: **{prediction[0]:.4f}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.dataframe(input_df)
