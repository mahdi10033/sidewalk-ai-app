import streamlit as st
import pandas as pd
from scoring_engine import score_sidewalk_segments

st.set_page_config(page_title="Sidewalk AI Scoring Tool", layout="wide")

st.title("Sidewalk AI Scoring Tool")
st.write("Upload a CSV file of sidewalk segment data to compute ADA, risk, and priority scores.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df, use_container_width=True)

        scored_df = score_sidewalk_segments(df)

        st.subheader("Scored Results")
        st.dataframe(scored_df, use_container_width=True)

        csv = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download scored results as CSV",
            data=csv,
            file_name="scored_sidewalk_segments.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin.")