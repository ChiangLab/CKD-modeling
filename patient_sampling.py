#!/usr/bin/env python
# Formerly patient_level_sample.py (original in _original/patient_level_sample.ipynb).
"""Patient-level sampling of per-encounter test outputs.

Reduces a model's per-encounter test output file (from
ckd_prediction.py or xgboost_baseline.py) to one encounter per
test patient, so that patient-level metrics are not dominated by patients with
many encounters:

  * patients whose CKD stage increases at some point: the encounter immediately
    before their *last* stage increase;
  * patients whose stage never increases: one encounter chosen at random.

Select the run with ``preset_modifier`` below. The first file of the preset is
sampled and written to ``<training output dir>_patient_level/<same file name>``,
which the ``*_patient_full`` presets in eval_classification / eval_tte
read.
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Training-output directories
# ----------------------------------------------------------------------------
ckd_event_full   = "365day_future_prediction_outputs_50_full_stage_filter_v9"  # matches version in ckd_prediction.py
eskd_event_full  = "365day_future_prediction_outputs_full_stage_filter_eskd_v6"  # matches version in xgboost_baseline.py

# ----------------------------------------------------------------------------
# Per-model output files
# ----------------------------------------------------------------------------
ckd_clf_dirs = [
    "/LSTM_365DayFutureTarget_detailed_outputs.csv",
    "/MLP_365DayFutureTarget_detailed_outputs.csv",
    "/RNN_365DayFutureTarget_detailed_outputs.csv",
    "/TCN_365DayFutureTarget_detailed_outputs.csv",
    "/Transformer_365DayFutureTarget_detailed_outputs.csv",
]
ckd_surv_dirs = [
    "/DeepSurv_LSTM_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_MLP_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_RNN_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_TCN_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_Transformer_365DayFutureTarget_detailed_outputs.csv",
]
eskd_event_dirs = ["/XGBoost_365DayFuture_Classifier_detailed_outputs_classification.csv"]


def paths(base, dirs):
    return [f"./{base}{d}" for d in dirs]


# ---- naming convention: <ckd|eskd|ckd_eskd>_<clf|surv>_event_<full|subset> ----
presets = {
    # main models
    "ckd_event_full": {
            "fp": f"./{ckd_event_full}",
            "filepaths": paths(ckd_event_full, ckd_clf_dirs + ckd_surv_dirs),
        },
    
    # baseline
    "eskd_clf_event_full": {
        "fp": f"./{eskd_event_full}", "modifier": "classification",
        "filepaths": paths(eskd_event_full, eskd_event_dirs),
    },

    
    
}

# ==== EDIT TO SWITCH RUNS ====
# preset_modifier = "ckd_event_full"
preset_modifier = "eskd_clf_event_full"

RANDOM_SEED = 42  # for the random encounter picked for non-progressing patients
np.random.seed(RANDOM_SEED)

preset = presets[preset_modifier]
fp = preset["fp"]
filepaths = preset["filepaths"]

print("preset: ", preset_modifier)
print("fp:      ", fp)
print("filepaths:")
for p in filepaths:
    print("  ", p)


def select_encounters(df):
    """Select a single encounter row for each patient.

    - Patients whose CKD stage increases (CKD_stage_numeric at the next
      encounter is higher): the encounter immediately before the last increase.
    - Patients whose CKD stage never increases: a random encounter.
    """
    df = df.copy()
    date_column = "date"

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=['PatientID', date_column]).reset_index(drop=True)
    df['next_stage'] = df.groupby('PatientID')['CKD_stage_numeric'].shift(-1)
    df['last_encounter'] = df['next_stage'] > df['CKD_stage_numeric']

    selected_indices = []

    for patient, group in df.groupby('PatientID'):
        progressions = group[group['last_encounter']]

        if not progressions.empty:
            # patient progresses: last encounter before a stage increase
            selected_indices.append(progressions.index.max())
        else:
            # patient does not progress: random encounter
            selected_indices.append(np.random.choice(group.index))

    result_df = df.loc[selected_indices].drop(columns=['next_stage', 'last_encounter']).reset_index(drop=True)
    return result_df


# ----------------------------------------------------------------------------
# Save to <training output dir>_patient_level/<same file name>
# ----------------------------------------------------------------------------
output_dir = os.path.dirname(filepaths[0]) + "_patient_level"
os.makedirs(output_dir, exist_ok=True)


read_paths = {os.path.abspath(p) for p in filepaths}

for path in filepaths:
    # same filename as the input, just in the patient-level folder
    new_path = os.path.join(output_dir, os.path.basename(path))
    assert os.path.abspath(new_path) not in read_paths, f"would overwrite an input file: {new_path}"

    df = pd.read_csv(path).rename(columns={'EventDate': 'date'})
    np.random.seed(42)
    df = select_encounters(df)
    df.to_csv(new_path)
    print(path, "->", new_path, f"({len(df)} patients)")
