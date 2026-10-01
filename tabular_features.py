#!/usr/bin/env python
# Formerly test_eskd_gen_v6.py (original in _original/test_eskd_gen_v6.ipynb).
"""Build the encounter-level tabular dataset for the XGBoost baseline.

Features follow the core feature set of the baseline study
(https://pmc.ncbi.nlm.nih.gov/articles/PMC10898824/):

  * Cohort: patients with at least one CKD ICD-10 code N18.x whose maximum
    recorded stage is >= 3 (N18.1-N18.5 -> 1-5, N18.6 (ESRD) -> 6,
    N18.9 (unspecified) -> 0).
  * Grain: one row per encounter (PatientID, META_1), dated by the encounter's
    first event date; ``EventMonth`` = that date's year-month.
  * Lab features: for labs whose name contains CREATININE, GFR, GFREST,
    ALBUMIN/CREATININE RATIO, PROTEIN/CREATININE RATIO, BUN or PTH, the monthly
    mean / min / max / std / count per distinct lab name, as columns
    ``lab_<stat>_<LAB NAME>``. Every encounter in a patient-month receives that
    month's aggregates.
  * AKI feature: ``AKI_ICD_Total`` -- number of acute kidney injury diagnoses
    (ICD-10 N17.0/.1/.2/.8/.9) in the patient-month (0 if none).
  * Stage: ``CKD_stage_numeric`` -- highest N18 stage coded during the
    encounter (null if none), plus the patient-level ``max_stage``.

Output: ``<output_dir>/processed_tab_eskd_v6.csv``, read by xgboost_baseline.py.
"""

import polars as pl
import os
import logging

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
output_fname = "processed_tab_eskd_v6.csv"
output_dir = "/opt/data/workingdir/ldiao/ckd_project/tabular_full"
event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v3.rpt.parquet"

os.makedirs(output_dir, exist_ok=True)
print(f"Processing started. Output directory: {output_dir}")

log_file_path = os.path.join(output_dir, "tab_gen_m.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Processing started. Output directory: {output_dir}")

# ----------------------------------------------------------------------------
# Load events
# ----------------------------------------------------------------------------
print(event_file)
df = pl.read_parquet(event_file)

logger.info(f"Initial DataFrame schema: {df.schema}")
print(df.shape)

# ----------------------------------------------------------------------------
# Cohort filter: patients whose max CKD stage (from N18.x codes) is >= 3
# ----------------------------------------------------------------------------
custom_map = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,  # ESRD
    9: 0,  # CKD, unspecified stage
}

print("Building Filter")
ckd_icd_df = (
    df.filter(pl.col("DataCategory").str.contains("N18"))
    .with_columns(
      pl.col("DataCategory")
        .str.extract(r"N18\.([1-9])", 1)
        .cast(pl.Int64)
        .replace(custom_map, default=None)
        .alias("CKD_stage_numeric")
    )
    .select([pl.col("PatientID"), pl.col("EventTimeStamp"), pl.col("META_1"), pl.col("CKD_stage_numeric")])
    .with_columns(
        pl.col("CKD_stage_numeric")
            .max()
            .over("PatientID")
            .alias("max_stage")
    )
    .filter(pl.col("max_stage") >= 3)
    .unique()
)

# Collapse ckd_icd_df from event grain to one row per encounter (PatientID, META_1)
ckd_stage_by_encounter = (
    ckd_icd_df
    .group_by(["PatientID", "META_1"])
    .agg([
        pl.col("CKD_stage_numeric").max().alias("CKD_stage_numeric"),  # worst stage coded in this encounter
        pl.col("max_stage").max().alias("max_stage"),  # patient-level max (constant per patient)
    ])
)

# Must be unique per encounter, otherwise the final join fans out rows.
n_rows = ckd_stage_by_encounter.shape[0]
n_keys = ckd_stage_by_encounter.select(["PatientID", "META_1"]).unique().shape[0]
assert n_rows == n_keys, f"ckd_stage_by_encounter is not unique per encounter: {n_rows} rows vs {n_keys} unique (PatientID, META_1) keys"
print(f"ckd_stage_by_encounter: {n_rows} rows, one per encounter (confirmed unique)")

print("Filtering and converting")
df = (
    df
    .join(
        ckd_icd_df.select("PatientID").unique(),
        on="PatientID",
        how='inner')
    .drop_nulls(subset=["DataNumeric"])
    .with_columns([
        pl.col("EventTimeStamp").str.strptime(pl.Datetime("us")),
        pl.col("DataCategory").fill_null(pl.col("META_2"))
    ])
    .with_columns(
        pl.col("EventTimeStamp").dt.date().alias("EventDate")
    )
)
print(len(df['PatientID'].unique()))

# ----------------------------------------------------------------------------
# Encounter-level skeleton: one row per (PatientID, META_1).
# EventMonth is the key for broadcasting monthly aggregates onto encounters.
# ----------------------------------------------------------------------------
enc_grouped_df = (
    df.group_by(["PatientID", "META_1"])
    .agg(pl.col("EventDate").min().alias("EventDate"))
    .with_columns(
        pl.col("EventDate").dt.strftime("%Y-%m").alias("EventMonth")
    )
    .sort(["PatientID", "META_1"])
)

# ----------------------------------------------------------------------------
# Monthly AKI counts (ICD-10 N17.x)
# ----------------------------------------------------------------------------
aki_icd_codes = ["N17.0", "N17.1", "N17.2", "N17.8", "N17.9"]
aki_events = df.filter(
    (pl.col("DataType") == "Diagnosis") &
    (pl.col("DataCategory").is_in(aki_icd_codes))
).with_columns(
    pl.col("EventDate").dt.strftime("%Y-%m").alias("EventMonth")
)

aki_count = aki_events.group_by(["PatientID", "EventMonth"]).agg([
    pl.len().alias("AKI_ICD_Total"),
])

# Every encounter in a patient-month gets that month's AKI count (0 if none).
enc_grouped_df = enc_grouped_df.join(
    aki_count, on=["PatientID", "EventMonth"], how="left", join_nulls=True
).with_columns(
    pl.col("AKI_ICD_Total").fill_null(0)
)

# ----------------------------------------------------------------------------
# Monthly lab aggregates for the baseline paper's top lab features
# ----------------------------------------------------------------------------
top_lab_features = [
    "CREATININE", "GFR", "GFREST", "ALBUMIN/CREATININE RATIO",
    "PROTEIN/CREATININE RATIO", "BUN", "PTH"
]

lab_df = df.filter(
    (pl.col("DataType") == "Labs") &
    (pl.col("DataCategory").cast(pl.Utf8).str.to_uppercase().str.contains("|".join(top_lab_features)))
).with_columns([
    pl.col("DataCategory").cast(pl.Utf8).str.to_uppercase().alias("LabCategory"),
    pl.col("EventDate").dt.strftime("%Y-%m").alias("EventMonth"),
    pl.col("DataNumeric").cast(pl.Float64, strict=False).alias("DataNumeric"),
])

# mean, min, max, std, count per PatientID + EventMonth + LabCategory
lab_monthly = lab_df.group_by(["PatientID", "EventMonth", "LabCategory"]).agg([
    pl.col("DataNumeric").mean().alias("mean"),
    pl.col("DataNumeric").min().alias("min"),
    pl.col("DataNumeric").max().alias("max"),
    pl.col("DataNumeric").std().alias("std"),
    pl.col("DataNumeric").count().alias("count"),
])

# Pivot wide: one column per (stat, LabCategory), e.g. "mean_CREATININE" -> "lab_mean_CREATININE"
lab_pivot = lab_monthly.pivot(
    index=["PatientID", "EventMonth"],
    on="LabCategory",
    values=["mean", "min", "max", "std", "count"],
)
rename_dict = {c: f"lab_{c}" for c in lab_pivot.columns if c not in ("PatientID", "EventMonth")}
lab_pivot = lab_pivot.rename(rename_dict)

enc_grouped_df = enc_grouped_df.join(lab_pivot, on=["PatientID", "EventMonth"], how="left", join_nulls=True)

agg_cols = [c for c in enc_grouped_df.columns if c.startswith("lab_") or c == "AKI_ICD_Total"]
enc_grouped_df = enc_grouped_df.with_columns([
    pl.col(c).cast(pl.Float64) for c in agg_cols
])

# ----------------------------------------------------------------------------
# Attach encounter-level CKD stage (one row per encounter, so no fan-out)
# ----------------------------------------------------------------------------
pre_join_rows = enc_grouped_df.shape[0]

enc_grouped_df = enc_grouped_df.join(
    ckd_stage_by_encounter,
    on=["PatientID", "META_1"], how='left'
)

assert enc_grouped_df.shape[0] == pre_join_rows, (
    f"CKD join changed row count: {pre_join_rows} -> {enc_grouped_df.shape[0]}. "
    "ckd_stage_by_encounter is no longer unique per (PatientID, META_1) -- check upstream."
)
print(f"enc_grouped_df: {enc_grouped_df.shape[0]} rows after CKD join (unchanged from {pre_join_rows})")

# ----------------------------------------------------------------------------
# Write output
# ----------------------------------------------------------------------------
base_df = enc_grouped_df

logger.info(f"[INFO] Final tabular shape: {base_df.shape}")
logger.info(f"[INFO] Sample features:\n{base_df.head()}")
logger.info(f"[INFO] CKD stage counts:\n{base_df['CKD_stage_numeric'].value_counts(sort=True)}")
base_df_path = os.path.join(output_dir, output_fname)
logger.info(f"Writing final DataFrame of shape {base_df.shape} to {base_df_path}")
base_df.write_csv(base_df_path)
logger.info("End of Tabular Generation")

# ----------------------------------------------------------------------------
# Cohort summary
# ----------------------------------------------------------------------------
df = pl.read_csv(base_df_path)

print(f"Rows: {df.shape[0]}, Patients: {df['PatientID'].n_unique()}")

# Encounter-level CKD stage prevalence
print(df["CKD_stage_numeric"].value_counts(sort=True).with_columns(
    (pl.col("count") / df.shape[0] * 100).round(2).alias("pct_rows")
))

# Patient-level prevalence: each patient's max (worst) CKD stage
patient_stage = df.group_by("PatientID").agg(pl.col("CKD_stage_numeric").max())
print(patient_stage["CKD_stage_numeric"].value_counts(sort=True).with_columns(
    (pl.col("count") / patient_stage.shape[0] * 100).round(2).alias("pct_patients")
))
