import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

from roboflow import Roboflow
from scoring_engine import score_sidewalk_segments

st.set_page_config(page_title="Sidewalk AI Scoring Tool", layout="wide")

st.title("Sidewalk AI Scoring Tool")
st.write("Upload a CSV file of sidewalk segment data to compute ADA, risk, and priority scores.")

st.markdown("---")

st.header("AI Sidewalk Image Classification")
st.write("Upload a sidewalk image to classify sidewalk conditions using AI.")

rf_api_key = st.secrets.get("ROBOFLOW_API_KEY")

if not rf_api_key:
    st.warning("Roboflow API key not configured.")
else:
    try:
        rf = Roboflow(api_key=rf_api_key)

        workspace_id = "sidewalk-qzu8g"
        project_id = "classification-of-sidewalk"
        version_number = 1

        project = rf.workspace(workspace_id).project(project_id)
        model = project.version(version_number).model

        uploaded_image = st.file_uploader(
            "Upload sidewalk image",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_image.read())
                temp_path = temp_file.name

            prediction = model.predict(temp_path, confidence=0).json()

            st.subheader("Prediction Results")

            result = prediction["predictions"][0]

            predicted_class = result.get("top", "Unknown")
            confidence = result.get("confidence", 0)
            
            if predicted_class == "":
                predicted_class = "Unknown"
            st.metric("Predicted Class", predicted_class)
            st.metric("Confidence", f"{confidence * 100:.1f}%")

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        scored_df = score_sidewalk_segments(df)

        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Segments", len(scored_df))
        col2.metric("Avg ADA Score", f"{scored_df['ada_score'].mean():.1f}")
        col3.metric("Avg Risk Score", f"{scored_df['risk_score'].mean():.1f}")
        col4.metric("Avg Priority Score", f"{scored_df['priority_score'].mean():.1f}")

        st.subheader("Uploaded Data")
        st.dataframe(scored_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin.")