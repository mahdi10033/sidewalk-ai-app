import pandas as pd
import numpy as np


def label_ada(score: float) -> str:
    if score >= 85:
        return "Compliant / Minor Issue"
    elif score >= 60:
        return "Moderate Concern"
    return "Non-Compliant"


def label_risk(score: float) -> str:
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Moderate"
    return "Low"


def label_priority(score: float) -> str:
    if score >= 85:
        return "Urgent Priority"
    elif score >= 70:
        return "High Priority"
    elif score >= 40:
        return "Medium Priority"
    return "Low Priority"


def normalize_bool(series: pd.Series) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
    }

    def convert_value(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, bool):
            return x
        return mapping.get(str(x).strip().lower(), np.nan)

    return series.apply(convert_value)


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().replace({"nan": np.nan})


EXPECTED_COLUMNS = [
    "segment_id",
    "estimated_width_in",
    "obstruction_present",
    "vegetation_encroachment",
    "surface_crack_severity",
    "vertical_displacement_est_in",
    "trip_hazard_present",
    "curb_ramp_present",
    "detectable_warning_present",
    "cross_slope_est_pct",
    "pedestrian_volume_class",
    "near_school",
    "near_hospital",
    "crash_history_flag",
    "complaint_history_flag",
    "equity_priority_area",
    "estimated_repair_cost",
]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "estimated_width_in",
        "vertical_displacement_est_in",
        "cross_slope_est_pct",
        "estimated_repair_cost",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = [
        "obstruction_present",
        "vegetation_encroachment",
        "trip_hazard_present",
        "curb_ramp_present",
        "detectable_warning_present",
        "near_school",
        "near_hospital",
        "crash_history_flag",
        "complaint_history_flag",
        "equity_priority_area",
    ]
    for col in bool_cols:
        df[col] = normalize_bool(df[col])

    df["surface_crack_severity"] = normalize_text(df["surface_crack_severity"])
    df["pedestrian_volume_class"] = normalize_text(df["pedestrian_volume_class"])

    return df


def compute_ada_score_df(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(100, index=df.index, dtype=float)
    score -= np.where(df["estimated_width_in"] < 36, 20, 0)
    score -= np.where(df["obstruction_present"] == True, 15, 0)
    score -= np.where(df["vertical_displacement_est_in"] > 0.5, 20, 0)
    score -= np.where(df["curb_ramp_present"] == False, 15, 0)
    score -= np.where(df["detectable_warning_present"] == False, 10, 0)
    score -= np.where(df["vegetation_encroachment"] == True, 10, 0)
    score -= np.where(df["cross_slope_est_pct"] > 2, 10, 0)
    return score.clip(lower=0, upper=100).round(0)


def compute_risk_score_df(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=df.index, dtype=float)
    score += np.where(df["trip_hazard_present"] == True, 20, 0)
    score += np.where(df["vertical_displacement_est_in"] > 0.5, 20, 0)
    score += np.where(df["obstruction_present"] == True, 15, 0)
    score += np.where(df["surface_crack_severity"] == "severe", 15, 0)
    score += np.where(df["surface_crack_severity"] == "moderate", 10, 0)
    score += np.where(df["surface_crack_severity"] == "minor", 5, 0)
    score += np.where(df["curb_ramp_present"] == False, 10, 0)
    score += np.where(df["pedestrian_volume_class"] == "high", 10, 0)
    score += np.where(df["pedestrian_volume_class"] == "medium", 5, 0)
    score += np.where(df["near_school"] == True, 10, 0)
    score += np.where(df["near_hospital"] == True, 10, 0)
    score += np.where(df["crash_history_flag"] == True, 5, 0)
    score += np.where(df["complaint_history_flag"] == True, 5, 0)
    return score.clip(lower=0, upper=100).round(0)


def compute_equity_score_df(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["equity_priority_area"] == True, 100, 50), index=df.index, dtype=float)


def compute_cost_effectiveness_score_df(df: pd.DataFrame) -> pd.Series:
    cost = df["estimated_repair_cost"]
    score = np.select(
        [
            cost < 2000,
            (cost >= 2000) & (cost < 5000),
            (cost >= 5000) & (cost < 10000),
            cost >= 10000,
        ],
        [100, 75, 50, 25],
        default=50,
    )
    return pd.Series(score, index=df.index, dtype=float)


def compute_priority_score_df(df: pd.DataFrame) -> pd.Series:
    ada_score = compute_ada_score_df(df)
    risk_score = compute_risk_score_df(df)
    equity_score = compute_equity_score_df(df)
    cost_effectiveness_score = compute_cost_effectiveness_score_df(df)

    ada_deficiency_score = 100 - ada_score

    priority_score = (
        0.45 * risk_score
        + 0.30 * ada_deficiency_score
        + 0.15 * equity_score
        + 0.10 * cost_effectiveness_score
    )

    return priority_score.clip(lower=0, upper=100).round(0)


def score_sidewalk_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_dataframe(df)

    df["ada_score"] = compute_ada_score_df(df)
    df["risk_score"] = compute_risk_score_df(df)
    df["equity_score"] = compute_equity_score_df(df)
    df["cost_effectiveness_score"] = compute_cost_effectiveness_score_df(df)
    df["priority_score"] = compute_priority_score_df(df)

    df["ada_label"] = df["ada_score"].apply(label_ada)
    df["risk_label"] = df["risk_score"].apply(label_risk)
    df["priority_label"] = df["priority_score"].apply(label_priority)

    return df.sort_values(by=["priority_score", "risk_score"], ascending=[False, False])