#!/usr/bin/env bash
# Reproduces docs/source/NPC_Train_Impute_count_HIPT.ipynb (Sections 0–5).
#
# All figures, tables, OrderData, and SaveData are written under DATA_ROOT
# (default: FineST_tutorial_data/{Figures,OrderData,SaveData}).
#
# Usage (from FineST/FineST/ package root):
#   bash run_NPC_tutorial_HIPT.sh
#
# Optional environment variables:
#   PYTHON              Python interpreter (default: python)
#   SYSTEM_PATH         Package root (default: ./)
#   DATA_ROOT           Tutorial data folder (default: FineST_tutorial_data)
#   WEIGHT_SAVE_PATH    Pretrained weights dir; skip training when set
#   RUN_STEP0/1/...     Set to 1 to enable a section (see below)
#   RUN_EVAL            Section 3 eval vs measured spots (default: 1)
#   GENES               Marker genes for eval (default: CD70 CD27)

set -euo pipefail

PYTHON="${PYTHON:-python}"
SYSTEM_PATH="${SYSTEM_PATH:-./}"
DATA_ROOT="${DATA_ROOT:-FineST_tutorial_data}"

# Pretrained checkpoint (notebook Section 2 skip). Download from Google Drive:
# https://drive.google.com/drive/folders/1w6hbMd0eUPJ4tFUft0O796NEbOrCeBxp
# Example: WEIGHT_SAVE_PATH="${DATA_ROOT}/logging/20250621001835815284"
# Leave empty to train a new model (weights saved under ${DATA_ROOT}/Figures/weights<TIMESTAMP>/).
WEIGHT_SAVE_PATH="${WEIGHT_SAVE_PATH:-}"

# Toggle pipeline sections (1 = run, 0 = skip). Precomputed assets ship with tutorial data.
RUN_STEP0="${RUN_STEP0:-0}"          # Section 0: HIPT within-spot features (~1–3 h)
RUN_STEP1="${RUN_STEP1:-1}"          # Sections 1 + 3: load, infer, impute within-spot
RUN_EVAL="${RUN_EVAL:-1}"            # Section 3: eval vs measured input spots
RUN_STEP2_INTERP="${RUN_STEP2_INTERP:-0}"  # Section 4.0: between-spot interpolation
RUN_STEP2A="${RUN_STEP2A:-0}"        # Section 4.1: HIPT between-spot features (~2–4 h)
RUN_STEP2B="${RUN_STEP2B:-0}"        # Section 5: sub-spot imputation (needs weights)

GENES="${GENES:-CD70 CD27}"

echo "=== FineST NPC tutorial (HIPT) ==="
echo "SYSTEM_PATH=${SYSTEM_PATH}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "WEIGHT_SAVE_PATH=${WEIGHT_SAVE_PATH:-<train new model>}"

########################################################
# Section 0: Extract within-spot image features (HIPT)
########################################################
if [[ "${RUN_STEP0}" == "1" ]]; then
  echo "--- Step 0: HIPT within-spot feature extraction ---"
  "${PYTHON}" -m FineST.image_feature_extraction \
    --position_path "${DATA_ROOT}/spatial/tissue_positions_list.csv" \
    --rawimage_path "${DATA_ROOT}/20210809-C-AH4199551.tif" \
    --dataset_class Visium \
    --STfactor_path "${DATA_ROOT}/spatial/scalefactors_json.json" \
    --is_05umperpix True \
    --hist_model HIPT \
    --patch_size 64 \
    --data_save_dir "${DATA_ROOT}"
fi

########################################################
# Sections 1 + 3: Train (optional) and within-spot inference
########################################################
if [[ "${RUN_STEP1}" == "1" ]]; then
  echo "--- Step 1: within-spot train / infer ---"
  STEP1_ARGS=(
    --system_path "${SYSTEM_PATH}"
    --data_root "${DATA_ROOT}"
    --parame_path parameter/parameters_NPC_HIPT.json
    --dataset_class Visium16
    --hist_model HIPT
    --gene_selected CD70
    --visium_path "${DATA_ROOT}/spatial/tissue_positions_list.csv"
    --image_embed_path "${DATA_ROOT}/ImgEmbeddings/HIPT/pth_64_16"
    --patch_size 64
    --do_scale True
    --weight_w 0.5
  )
  if [[ -n "${WEIGHT_SAVE_PATH}" ]]; then
    STEP1_ARGS+=(--weight_save_path "${WEIGHT_SAVE_PATH}")
  fi
  "${PYTHON}" -m FineST.step1_FineST_train_infer "${STEP1_ARGS[@]}"
fi

########################################################
# Section 3: Evaluate infer / impute vs measured spots
########################################################
if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ ! -f "${DATA_ROOT}/SaveData/adata_imput_spot.h5ad" ]]; then
    echo "ERROR: RUN_EVAL=1 requires Step 1 outputs under ${DATA_ROOT}/SaveData/." >&2
    echo "       Run with RUN_STEP1=1 first, or set RUN_EVAL=0." >&2
    exit 1
  fi
  echo "--- Eval: vs measured input spots ---"
  read -r -a GENE_ARGS <<< "${GENES}"
  "${PYTHON}" -m FineST.evaluation \
    --platform visium \
    --system_path "${SYSTEM_PATH}" \
    --data_root "${DATA_ROOT}" \
    --genes "${GENE_ARGS[@]}"
fi

########################################################
# Section 4–5: Between-spot interpolation and imputation
########################################################
if [[ "${RUN_STEP2_INTERP}" == "1" ]]; then
  echo "--- Step 2.0: between-spot interpolation ---"
  "${PYTHON}" -m FineST.spot_interpolation \
    --position_path "${DATA_ROOT}/spatial/tissue_positions_list.csv"
fi

if [[ "${RUN_STEP2A}" == "1" ]]; then
  echo "--- Step 2A: HIPT between-spot feature extraction ---"
  "${PYTHON}" -m FineST.image_feature_extraction \
    --position_path "${DATA_ROOT}/spatial/tissue_positions_list_add.csv" \
    --rawimage_path "${DATA_ROOT}/20210809-C-AH4199551.tif" \
    --dataset_class Visium \
    --STfactor_path "${DATA_ROOT}/spatial/scalefactors_json.json" \
    --is_05umperpix True \
    --hist_model HIPT \
    --patch_size 64 \
    --data_save_dir "${DATA_ROOT}" \
    --output_name NEW_pth_64_16
fi

if [[ "${RUN_STEP2B}" == "1" ]]; then
  if [[ -z "${WEIGHT_SAVE_PATH}" ]]; then
    echo "ERROR: RUN_STEP2B=1 requires WEIGHT_SAVE_PATH (Step 1 weights or downloaded checkpoint)." >&2
    exit 1
  fi
  echo "--- Step 2B: sub-spot imputation (within + between) ---"
  "${PYTHON}" -m FineST.step2_High_resolution_impute \
    --system_path "${SYSTEM_PATH}" \
    --data_root "${DATA_ROOT}" \
    --hist_model HIPT \
    --parame_path parameter/parameters_NPC_HIPT.json \
    --dataset_class Visium16 \
    --gene_selected CD70 \
    --visium_path "${DATA_ROOT}/spatial/tissue_positions_list.csv" \
    --imag_within_path "${DATA_ROOT}/ImgEmbeddings/HIPT/pth_64_16" \
    --imag_betwen_path "${DATA_ROOT}/ImgEmbeddings/HIPT/NEW_pth_64_16" \
    --weight_save_path "${WEIGHT_SAVE_PATH}"
fi

echo "=== Done. Outputs under ${DATA_ROOT}/Figures, OrderData/, SaveData/ ==="
