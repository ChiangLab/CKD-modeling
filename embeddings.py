#!/usr/bin/env python
# Formerly embedding_gen_pl.py (original in _original/embedding_gen_pl.py).
"""Generate encounter-level clinical pseudo-notes and ClinicalBERT embeddings.

Steps:
  1. Cohort: patients with at least one CKD ICD-10 code N18.x whose maximum
     recorded stage is >= 3 (N18.1-N18.5 -> 1-5, N18.6 (ESRD) -> 6,
     N18.9 (unspecified) -> 0). Events with a null ``DataNumeric`` are dropped.
  2. Each event becomes a clause:
       Diagnosis   -> " - ICD-10 code N18.3: <ICD long title>"
                      (codes without an ICD-10 title, mostly ICD-9, are dropped)
       Medications -> " - Medication administered: <name>"
       Procedure   -> " - Procedure performed: <name>"
       Labs        -> " - <lab name>: <value>"
     Demographics/Encounter rows are not turned into clauses.
  3. Clauses are joined per day ("On <date>, the patient had the following
     records: ..."), and days are joined per encounter (META_1) into one
     pseudo-note prefixed with "Encounter <id> starting <date>." and a
     demographic sentence ("Patient <id>, born <DOB> is a <race> <sex>.").
  4. Each pseudo-note is encoded separately with ClinicalBERT (truncated to 512
     tokens); the CLS vector of the last hidden layer is the embedding.

Outputs in ``output_dir``:
  * ``meta_v3_all.csv`` -- '$'-separated metadata incl. the note text (enc_summary).
  * ``meta_v3.csv`` -- same without the note text; read by ckd_prediction.py.
    Columns: PatientID, META_1, date, enc_id, emb_id, CKD_stage_numeric, max_stage.
  * ``<PatientID>/<PatientID>.npy`` -- one row per metadata row of the patient,
    ordered by emb_id (encounter date order).
"""

import os
import pandas as pd
import polars as pl
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
cuda_num = 2
print(f"cuda:{str(cuda_num)}")

icd_file = "/opt/data/commonfilesharePHI/ldiao/ckd_project/icd_mapping.csv"
demographic_file = '/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-supplemental_pull.rpt'

output_dir = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/ckd_embedding_full_v3"
event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v3.rpt.parquet"

output_dir += "_icd"  # notes include ICD-10 long titles

filter_ckd_stage = True  # only tags the output dir; the stage >= 3 filter below always applies
if filter_ckd_stage:
    output_dir += "_stage_filter"

print(output_dir)

csv = event_file
icd = icd_file
model_name = "/opt/data/commonfilesharePHI/slee/MEME/clinicalBERT-emily"  # local copy of ClinicalBERT
embed_dim = 768

os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# ICD-10 code -> long title (codes stored without the dot)
# ----------------------------------------------------------------------------
icd_df = pd.read_csv(icd)
icd_df["icd_code"] = icd_df["icd_code"].astype(str).str.replace(".", "", regex=False)
icd_map = dict(zip(icd_df["icd_code"], icd_df["long_title"]))

# ----------------------------------------------------------------------------
# Cohort: patients whose max CKD stage (from N18.x codes) is >= 3
# ----------------------------------------------------------------------------
df = pl.read_parquet(csv)

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
print("Filtering and converting")

# Keep cohort events; parse timestamps; fall back to META_2 for missing categories.
df = (
    df
    .join(
        ckd_icd_df.select("PatientID").unique(),
        on="PatientID",
        how='inner')
    .drop_nulls(subset=["DataNumeric"])
    .with_columns([
        pl.col("EventTimeStamp").str.strptime(pl.Datetime("us"))
        , pl.col("DataCategory").fill_null(pl.col("META_2"))
    ])
    .with_columns(
        pl.col("EventTimeStamp").dt.date().alias("EventDate")
    )
)

# ----------------------------------------------------------------------------
# Demographic sentence per patient
# ----------------------------------------------------------------------------
demographic_df = (
    pl.read_csv(demographic_file, separator='$')
    .with_columns(
        pl.format(
            "Patient {}, born {} is a {} {}."
            , pl.col("PatientID")
            , pl.col("DOB")
            , pl.col("EthnoRacialCategory")
                .replace("*No Usable Values", "Unknown Race")
                .replace("Multiple Ethnoracial Categories", "Multiracial")
            , pl.col("Sex")
        ).alias("sentence")
    )
)
demographic_map = dict(zip(demographic_df["PatientID"], demographic_df["sentence"]))

# ----------------------------------------------------------------------------
# Clinical events -> clauses
# ----------------------------------------------------------------------------
sentences_df = (
    df.filter(~pl.col("DataType").is_in(["Demographics", "Encounter"]))
    .with_columns(
        pl.when(pl.col("DataType") == "Diagnosis").then(
            pl.col("DataCategory")
                .str.replace(".", "", literal=True)
                .replace(icd_map, default="Unknown")
        ).otherwise(pl.format("NA"))
        .alias("long_title")
        , pl.when((pl.col("DataType") == "Encounter") &
                (pl.col("DataCategory").str.contains("//"))).then(
            pl.col("DataCategory").str.replace("//", ": ")
        ).otherwise(pl.format("NA"))
        .alias("EncounterType")
    )
    .filter(pl.col("long_title") != "Unknown")  # diagnoses without an ICD-10 title (mostly ICD-9)
    .with_columns(
        pl.when(pl.col("DataType") == "Diagnosis").then(
            pl.format(" - ICD-10 code {}: {}", pl.col("DataCategory"), pl.col("long_title"))
        )
        .when(pl.col("DataType") == "Medications").then(
            pl.format(" - Medication administered: {}", pl.col("DataCategory"))
        )
        .when(pl.col("DataType") == "Procedure").then(
            pl.format(" - Procedure performed: {}", pl.col("DataCategory"))
        )
        .when(pl.col("DataType") == "Labs").then(
            pl.format(" - {}: {}", pl.col("DataCategory"), pl.col("DataNumeric"))
        )
        .when(pl.col("DataType") == "Encounter").then(
            pl.when(pl.col("EncounterType") != "NA").then(
                pl.format(" - {}", pl.col("DataCategory"))
            ).otherwise(
                pl.format(" - {}: {}", pl.col("DataCategory"), pl.col("DataNumeric"))
            )
        )
        .otherwise(pl.format("Not parsed."))
        .alias("sentence")
    )
    .drop(pl.col("META_2"))
    .unique()
)

# ----------------------------------------------------------------------------
# Daily summaries: "On <date>, the patient had the following records: ..."
# ----------------------------------------------------------------------------
day_grouped_df = (
    sentences_df.group_by("PatientID", "META_1", "EventDate")
      .agg([
        pl.col("sentence").str.concat(" ").alias("day_summary")
        , pl.col("EventDate").first().alias("date")
      ])
      .with_columns(
        pl.format("On {}, the patient had the following records: {}", pl.col("date"), pl.col("day_summary"))
        .alias("day_summary")
      )
)

# ----------------------------------------------------------------------------
# Encounter pseudo-notes (one per PatientID, META_1); emb_id = order within patient
# ----------------------------------------------------------------------------
enc_grouped_df = (
    day_grouped_df.group_by("PatientID", "META_1")
        .agg([
            pl.col("day_summary").str.concat("\n").alias("enc_summary")
            , pl.col("EventDate").min().alias("date")
            , pl.col("META_1").first().alias("enc_id")
            , pl.col("PatientID").first().replace(demographic_map, default="Demographics not available.").alias("demographic_string")
        ])
        .with_columns(
            pl.format("Encounter {} starting {}. {} {}", pl.col("enc_id"), pl.col("date"), pl.col("demographic_string"), pl.col("enc_summary")).alias("enc_summary")
        )
    .drop("demographic_string")
    .sort("PatientID", "date")
    .with_columns(pl.int_range(pl.len()).over("PatientID").alias("emb_id"))
)

# ----------------------------------------------------------------------------
# Metadata export. ckd_icd_df is at event grain, so an encounter with several
# N18 events gets several rows here (and several embeddings below).
# ----------------------------------------------------------------------------
enc_grouped_df = enc_grouped_df.join(
    ckd_icd_df.select(["PatientID", "META_1", "CKD_stage_numeric", "max_stage"])
    , on=["PatientID", "META_1"], how='left'
)
enc_grouped_df.write_csv(os.path.join(output_dir, "meta_v3_all.csv"), separator="$")
print("Saved meta df")
enc_grouped_df.drop("enc_summary").write_csv(os.path.join(output_dir, "meta_v3.csv"), separator="$")
print("Saved meta df clean")

# ----------------------------------------------------------------------------
# ClinicalBERT CLS embeddings, saved per patient as <PatientID>/<PatientID>.npy
# ----------------------------------------------------------------------------
print(f"[INFO] Loading model from: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = f"cuda:{cuda_num}"
model.to(device)
model.eval()


def get_cls_embeddings(texts, tokenizer, model, device, embed_dim):
    """CLS vector of the last hidden layer, truncated/zero-padded to embed_dim."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    if cls_emb.size(1) > embed_dim:
        cls_emb = cls_emb[:, :embed_dim]
    else:
        pad = embed_dim - cls_emb.size(1)
        cls_emb = torch.nn.functional.pad(cls_emb, (0, pad), value=0)
    return cls_emb.cpu().numpy()


patient_grouped_df = (
    enc_grouped_df.sort(["PatientID", "emb_id"])
        .group_by("PatientID")
        .agg(pl.col("enc_summary").alias("pseudonotes"))
)
grouped_dict = dict(zip(patient_grouped_df["PatientID"], patient_grouped_df["pseudonotes"]))
print("Generating embeddings")
for pid, texts in tqdm(grouped_dict.items()):
    patient_folder = os.path.join(output_dir, str(pid))
    os.makedirs(patient_folder, exist_ok=True)

    embeddings = [get_cls_embeddings(t, tokenizer, model, device, embed_dim) for t in texts]
    fname = f"{pid}"
    fpath = os.path.join(patient_folder, fname)
    np.save(fpath, np.array(embeddings).squeeze())
