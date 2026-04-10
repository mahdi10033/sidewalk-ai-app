import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scoring_engine import score_sidewalk_segments

st.set_page_config(page_title="Sidewalk AI Scoring Tool", layout="wide")

st.title("Sidewalk AI Scoring Tool")
st.write("Upload a CSV file of sidewalk segment data to compute ADA, risk, and priority scores.")

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

        st.subheader("Filter Results")
        priority_filter = st.multiselect(
            "Priority Label",
            options=sorted(scored_df["priority_label"].dropna().unique()),
            default=sorted(scored_df["priority_label"].dropna().unique())
        )

        filtered_df = scored_df[scored_df["priority_label"].isin(priority_filter)].copy()

        st.subheader("Uploaded Data")
        st.dataframe(df, use_container_width=True)

        st.subheader("Scored Results")
        display_cols = [
            "segment_id",
            "ada_score", "ada_label",
            "risk_score", "risk_label",
            "priority_score", "priority_label",
            "estimated_repair_cost",
            "estimated_width_in",
            "surface_crack_severity",
            "vertical_displacement_est_in",
            "obstruction_present",
            "curb_ramp_present",
            "detectable_warning_present",
            "near_school",
            "near_hospital",
            "equity_priority_area",
        ]
        existing_display_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[existing_display_cols], use_container_width=True)

        st.subheader("Priority Distribution")
        priority_counts = filtered_df["priority_label"].value_counts()

        fig, ax = plt.subplots()
        priority_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Priority Label")
        ax.set_ylabel("Count")
        ax.set_title("Count of Segments by Priority")
        st.pyplot(fig)

        csv = filtered_df.to_csv(index=False).encode("utf-8")
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