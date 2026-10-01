#!/usr/bin/env bash
# Run the CKD progression pipeline step by step.
#
#   bash run_pipeline.sh <step>
#
# Steps (in order):
#   embed        ClinicalBERT pseudo-note embeddings        embeddings.py
#   tab          tabular baseline features                  tabular_features.py
#   train_embed  embedding sequence models + DeepSurv       ckd_prediction.py
#   train_tab    tabular XGBoost baseline                   xgboost_baseline.py
#   patient      patient-level sampling of test outputs     patient_sampling.py
#   eval_clf     bootstrap classification metrics           eval_classification.py
#   eval_surv    bootstrap survival metrics                 eval_tte.py
#
# Input/output paths for embed, tab, patient, eval_clf and eval_surv are set in
# the configuration block at the top of each script (preset_modifier for the
# evaluation scripts). The two training scripts take the CLI flags below.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p log_files pt_files joblib_files results_logs

# ---- data locations ----
EMBEDDING_ROOT="/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/ckd_embedding_full_v3_icd_stage_filter"
TABULAR_FILE="/opt/data/workingdir/ldiao/ckd_project/tabular_full/processed_tab_eskd_v6.csv"

step="${1:-}"
case "$step" in
  embed)
    python embeddings.py
    ;;

  tab)
    python tabular_features.py
    ;;

  train_embed)
    # Trains RNN, LSTM, Transformer, MLP, TCN and their DeepSurv variants.
    # Values shown are the script defaults; edit to match the reported run.
    python ckd_prediction.py \
      --embedding-root "$EMBEDDING_ROOT" \
      --metadata-file meta_v3.csv \
      --prediction-horizon-days 365 \
      --window-size 365 \
      --embed-dim 768 \
      --epochs 50 \
      --batch-size 3072 \
      --lr 5e-5 \
      --patience 5 \
      --scheduler-patience 2 \
      --hidden-dim 128 \
      --num-layers 4 \
      --rnn-dropout 0.2 \
      --transformer-nhead 4 \
      --transformer-dim-feedforward 256 \
      --transformer-dropout 0.2 \
      --cox-loss-weight 1.0 \
      --random-seed 42
    ;;

  train_tab)
    python xgboost_baseline.py \
      --tabular-data-file "$TABULAR_FILE" \
      --prediction-horizon-days 365 \
      --window-size 365 \
      --xgb-n-estimators 100 \
      --xgb-max-depth 6 \
      --xgb-learning-rate 0.1 \
      --random-seed 42
    ;;

  patient)
    python patient_sampling.py
    ;;

  eval_clf)
    python eval_classification.py
    ;;

  eval_surv)
    python eval_tte.py
    ;;

  *)
    sed -n '2,17p' "$0"
    exit 1
    ;;
esac
