# CKD Progression Modeling (WIP)

This repository provides a full pipeline for simulating longitudinal patient-note data from tabular EHR events, extracting contextual embeddings from the resulting clinical notes, and training various deep learning architectures to predict chronic kidney disease (CKD) progression to stage 4+ within one year. 

The embedding models are compared against a tabular XGBoost baseline that follows the core features of [the baseline study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10898824/). 

The pipeline consists of the following main scripts:

---

### `embeddings.py`: Pseudonote (tabular to text) Generation + Embedding Extraction

This script reads tabular event data and ICD mappings to generate synthetic, per-patient, per-day clinical notes. These notes are constructed by integrating demographic attributes, diagnoses (ICD-10 codes), medications, and procedures. The script maps GFR readings to CKD stages (1--5) using forward-filled values while enforcing monotonic progression. The resulting notes are encoded into dense representations (CLS embeddings) using a pretrained transformer model (e.g., ClinicalBERT).

Each embedding is stored in a per-patient directory with corresponding metadata (GFR, CKD stage, text). Metadata is saved to a CSV file for downstream modeling.

**Usage:**
```bash
bash run_pipeline.sh embed
```
---

### `tabular_features.py`: Tabular Baseline Features

This script builds the features for the tabular baseline from the same cohort. It produces one row per encounter, with features aggregated over the calendar month of the encounter

**Usage:**
```bash
bash run_pipeline.sh tab
```

---

### `ckd_prediction.py`: Longitudinal CKD Classification Models

This script reads the embeddings and associated metadata and builds sequences of patient-note embeddings. The task is to predict whether the patient will progress to stage 4 or higher CKD at the next time step. Labels are derived from cleaned CKD stages, converted into a binary label indicating whether stage 4+ is reached.

It supports training and evaluation of the following sequence models:
- RNN and LSTM with optional bidirectionality
- Transformer encoder with positional encoding
- MLP over flattened embedding windows
- Temporal Convolutional Network (TCN)

Each model is trained using early stopping and learning rate scheduling. Test performance includes Accuracy, F1, Precision, Recall, AUROC, and AUPRC. An optional label-switch analysis tracks whether models anticipate CKD progression earlier than the ground truth.

**Usage:**
```bash
bash run_pipeline.sh train_embed
```

---


### `xgboost_baseline.py`: Tabular XGBoost Baseline

This script trains the tabular baseline using the same labels, split and windowing as `ckd_prediction.py`. 

It supports training and evaluation of the following model:
 - XGBoost gradient-boosted tree classifier

**Usage:**
```bash
bash run_pipeline.sh train_tab
```


---

### `patient_sampling.py`: Patient-Level Test Sets

This script reduces the per-encounter test outputs to one test encounter per patient. For patients whose stage increases, it keeps the encounter just before their last increase; for all other patients, it keeps a random encounter (seed 42, `RANDOM_SEED`). 

**Usage:**
```bash
bash run_pipeline.sh patient
```

---

### (placeholder) `patient_similarity.py`: Patient-Like-Me kNN and Patient Similarity Output

This script builds one representation per patient from the stage 3 encounter embeddings produced by `embeddings.py` and estimates progression risk from each patient's most similar patients.

It supports training and evaluation of the following models:
- Cosine k-nearest-neighbor (kNN) retrieval over mean and mean-max pooled embeddings, with and without NNMF
- Random forest classifier as a supervised reference


**Usage:**
```bash
bash run_pipeline.sh similarity
```
---


### Metadata Format and Embedding Structure

The script `embeddings.py` produces a metadata CSV file named `meta_v3.csv`. Each row in the file corresponds to a single patient-encounter pair and contains the following fields:


```markdown
| PatientID | META_1 | date       | enc_id | emb_id | CKD_stage_numeric | max_stage |
|-----------|--------|------------|--------|--------|-------------------|-----------|
| P0001     | E1001  | 2022-07-14 | E1001  | 0      | 3                 | 4         |
| P0001     | E1002  | 2022-08-03 | E1002  | 1      | 3                 | 4         |
| P0001     | E1003  | 2022-10-21 | E1003  | 2      | 3                 | 4         |
| P0001     | E1004  | 2023-01-09 | E1004  | 3      | 4                 | 4         |
| P0002     | E2001  | 2021-03-02 | E2001  | 0      | 3                 | 3         |
| P0002     | E2002  | 2021-05-17 | E2002  | 1      | 3                 | 3         |
```

`meta_v3_all.csv` has the same data with an additional `enc_summary` column for the encounter's pseudonote/pre-embedding text:

```markdown
| PatientID | META_1 | date       | enc_id | emb_id | CKD_stage_numeric | max_stage | enc_summary                                                        |
|-----------|--------|------------|--------|--------|-------------------|-----------|--------------------------------------------------------------------|
| P0001     | E1001  | 2022-07-14 | E1001  | 0      | 3                 | 4         | Encounter E1001 starting 2022-07-14. Patient P0001, born 1956-0... |
| P0001     | E1002  | 2022-08-03 | E1002  | 1      | 3                 | 4         | Encounter E1002 starting 2022-08-03. Patient P0001, born 1956-0... |
| P0002     | E2001  | 2021-03-02 | E2001  | 0      | 3                 | 3         | Encounter E2001 starting 2021-03-02. Patient P0002, born 1948-1... |
```

A pseudonote is not a clinician-written note. It is generated from the structured events of one encounter: a header with the encounter ID and start date, a demographic sentence, then one line per day listing that day's diagnoses, medications, procedures and labs.


Each embedding is stored in a single `.npy` array (one row per encounter, indexed by `emb_id`) and follows the folder structure:
```
{output_dir}/{PatientID}/{PatientID}.npy
```
For instance, embeddings for patients `P0001` and `P0002` would be located at:
```
ckd_embedding_full_v3_icd_stage_filter/P0001/P0001.npy
ckd_embedding_full_v3_icd_stage_filter/P0002/P0002.npy
...
```

This design allows efficient access of per-patient longitudinal embeddings for modeling.


---

### Shell Scripts

**`run_pipeline.sh train_embed`**
```bash
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
```

**`run_pipeline.sh train_tab`**
```bash
python xgboost_baseline.py \
  --tabular-data-file "$TABULAR_FILE" \
  --prediction-horizon-days 365 \
  --window-size 365 \
  --xgb-n-estimators 100 \
  --xgb-max-depth 6 \
  --xgb-learning-rate 0.1 \
  --random-seed 42
```

Make sure to `chmod +x run_pipeline.sh` before executing the script.

#### Acknowledgments

This work utilized resources provided by the [UCLA Department of Computational Medicine](https://compmed.ucla.edu/).

This work was also supported by OptumLabs.

---

For further inquiries or collaboration, please contact:
- Simon A. Lee: [simonlee711@g.ucla.edu](mailto:simonlee711@g.ucla.edu)
- Jeffrey N. Chiang: [njchiang@g.ucla.edu](mailto:njchiang@g.ucla.edu)
- Laura Diao: ldiao@g.ucla.edu

