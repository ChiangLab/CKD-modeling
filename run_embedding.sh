#!/bin/bash
python embeddings.py \
  --csv patients_subset_100.csv \
  --icd icd_mapping.csv \
  --output_dir ckd_embeddings_100 \
  --model_name [HUGGINGFACE_PATH] \
  --embed_dim 768 \
  --batch_size 128
