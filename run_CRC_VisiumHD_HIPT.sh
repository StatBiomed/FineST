#!/usr/bin/env bash
# Reproduces docs/source/CRC16_Train_Impute_count_HIPT.ipynb (Sections 0–4).
#
# Uses FineST_tutorial_data_VisiumHD/ as the single data root, replacing
# FineST_local/Dataset/CRC16um/ and ContrastCRC16geneLR/ from the notebook.
#
# Expected layout under DATA_ROOT:
#   Visium_HD_Human_Colon_Cancer_tissue_image.btf
#   square_016um/tissue_positions.parquet
#   square_016um/scalefactors_json.json
#   HIPT/HD_CRC_16um_pth_32_16/          (Step 0 output or precomputed)
#   OrderData/                           (Step 1: position_order.csv, matrix_order.npy)
#   SaveData/                            (Step 1 + eval: adata_*.h5ad)
#   Figures/                             (Step 1 + eval: plots)
#
# Usage (from FineST/FineST/ package root):
#   bash run_CRC_VisiumHD_HIPT.sh
#
# Optional environment variables:
#   PYTHON, SYSTEM_PATH, DATA_ROOT, WEIGHT_SAVE_PATH
#   RUN_STEP0, RUN_STEP1, RUN_EVAL
#   GENES (space-separated marker genes for eval, default: SPP1 COL1A1)

set -euo pipefail

PYTHON="${PYTHON:-python}"
SYSTEM_PATH="${SYSTEM_PATH:-./}"
DATA_ROOT="${DATA_ROOT:-FineST_tutorial_data_VisiumHD}"
GENES="${GENES:-SPP1 COL1A1}"

# Pretrained CRC16 HIPT checkpoint (notebook Section 2 skip).
# Bundled under package finetune/; copy to ${DATA_ROOT}/logging/ to keep all assets in DATA_ROOT:
#   cp -r finetune/20260801162414255436 "${DATA_ROOT}/logging/"
WEIGHT_SAVE_PATH="${WEIGHT_SAVE_PATH:-finetune/20260801162414255436}"

RUN_STEP0="${RUN_STEP0:-0}"   # Section 0: HIPT 16 µm bin features (~2–6 h)
RUN_STEP1="${RUN_STEP1:-1}"   # Sections 1 + 3: load, infer, impute within-bin
RUN_EVAL="${RUN_EVAL:-1}"     # Sections 3 + 3.6 + 4: eval vs input 16 µm + native 8 µm

SQUARE_DIR="${DATA_ROOT}/square_016um"
EMBED_DIR="${DATA_ROOT}/HIPT/HD_CRC_16um_pth_32_16"
EMBED_IMG_DIR="${DATA_ROOT}/HIPT/HD_CRC_16um_pth_32_16_image"

echo "=== FineST CRC16 Visium HD tutorial (HIPT) ==="
echo "SYSTEM_PATH=${SYSTEM_PATH}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "WEIGHT_SAVE_PATH=${WEIGHT_SAVE_PATH}"

########################################################
# Section 0: HE image feature extraction (HIPT, 16 µm)
########################################################
if [[ "${RUN_STEP0}" == "1" ]]; then
  echo "--- Step 0: HIPT Visium HD feature extraction ---"
  "${PYTHON}" -m FineST.image_feature_extraction \
    --dataset_class VisiumHD \
    --position_path "${SQUARE_DIR}/tissue_positions.parquet" \
    --rawimage_path "${DATA_ROOT}/Visium_HD_Human_Colon_Cancer_tissue_image.btf" \
    --STfactor_path "${SQUARE_DIR}/scalefactors_json.json" \
    --is_05umperpix True \
    --hist_model HIPT \
    --patch_size 32 \
    --output_pth "${EMBED_DIR}" \
    --output_img "${EMBED_IMG_DIR}"
fi

########################################################
# Sections 1 + 3: Load ST, align, infer & impute within-bin
########################################################
if [[ "${RUN_STEP1}" == "1" ]]; then
  echo "--- Step 1: Visium HD within-bin train / infer ---"
  STEP1_ARGS=(
    --system_path "${SYSTEM_PATH}"
    --data_root "${DATA_ROOT}"
    --parame_path parameter/parameters_CRC16_HIPT.json
    --dataset_class VisiumHD
    --hist_model HIPT
    --gene_selected SPP1
    --visium_path "${SQUARE_DIR}/tissue_positions.parquet"
    --image_embed_path "${EMBED_DIR}"
    --patch_size 32
    --do_scale True
    --weight_w 0.5
  )
  if [[ -n "${WEIGHT_SAVE_PATH}" ]]; then
    STEP1_ARGS+=(--weight_save_path "${WEIGHT_SAVE_PATH}")
  fi
  "${PYTHON}" -m FineST.step1_FineST_train_infer "${STEP1_ARGS[@]}"
fi

########################################################
# Sections 3 + 3.6 + 4: Evaluate vs input 16 µm and native 8 µm bins
########################################################
if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ ! -f "${DATA_ROOT}/SaveData/adata_imput_spot.h5ad" ]]; then
    echo "ERROR: RUN_EVAL=1 requires Step 1 outputs under ${DATA_ROOT}/SaveData/." >&2
    echo "       Run with RUN_STEP1=1 first, or set RUN_EVAL=0." >&2
    exit 1
  fi
  echo "--- Eval: vs input 16 µm bins + native 8 µm bins ---"
  read -r -a GENE_ARGS <<< "${GENES}"
  "${PYTHON}" -m FineST.evaluation \
    --platform visiumhd \
    --system_path "${SYSTEM_PATH}" \
    --data_root "${DATA_ROOT}" \
    --genes "${GENE_ARGS[@]}"
fi

echo "=== Done. Outputs under ${DATA_ROOT}/{Figures,OrderData,SaveData}/ ==="
