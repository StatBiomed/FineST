## 2024.01.24 copy from HIPT_image_feature.py
##            Adjust the code "for point in coordinates:", don't leave two spaces blank
## 2024.01.24 copy from HIPT_image_feature_NPC1new.py
##            Used to CRC
## 2024.10.05 copy from HIPT_image_feature_NPC1.py
##            Omit Image, only save Embedding
## 2024.10.30 Copy from HIPT_image_feature_CRC16um_FineST.py
##            write demo  
## 2024.11.14 Update input .csv for Visium, 
##            repeatable, reference FineST/FineST_local/FHIPT_test.ipynb
## 2025.01.10 use Virchow2, https://huggingface.co/paige-ai/Virchow2
## 2025.01.23 setting HIPT and Virchow2 selcetion, using original .parquet 
## 2025.02.06 add 'sys.path.append("./FineST/FineST")' for use 'HIPT' independently
##            omit Line130-Line135, for sing-nuclei file, dont need rename the colnums
##            this problem is from see the path image from 'sc_Patient1_pth_14_14_image'
##            there are some 'blank' patches. And compare the '_spot.csv' and 'all_spot_sc.csv'
## 2025.06.20 using 'FineST_demo'
## 2026.02.03 LLY make the final clean version
## 2026.04.14 LLY add the pixel_size_raw and pixel_size to get "scale" for the image
## 2026.04.14 LLY add UNI method
## 2026.05.20 Xenium Select4 — VUHD113 (CSV: X_pix_HE/Y_pix_HE from 0.2125 um/px)
##            Target 0.5 um/px so patch_size=16 -> ~8 um: scale_image=True, scale=0.2125/0.5=0.425
## 2026.07.06 LLY add the CODEX_HCC HE cell feature extraction from given cell_data.csv and HE image(.tif) with X Y
## 2026.08.04 LLY transfer this .py from demo to FineST/FineST, add the default_logging_folder for the log files
## 2026.08.04 LLY simplify the code, remove the dataset_class and STfactor_path, only use the default_output_paths

import os
import sys
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.transform import rescale
import numpy as np
import pandas as pd
from torchvision import transforms
from datetime import datetime
import random
import time
import argparse
import logging
import json
from typing import Tuple, List, Optional

#################################################################
## For Virchow2 
## Note: timm is only imported when hist_model='Virchow2' is used
## If you plan to use Virchow2, install it with: pip install timm
## if use transforms in Virchow2
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform  
#################################################################


## Constants
DEFAULT_SCALE = 0.5
# Standard Visium spot diameter in microns (used to derive µm/px from spot_diameter_fullres)
VISIUM_SPOT_DIAMETER_UM = 55.0
# 10x Xenium registered HE / morphology (µm per pixel at full resolution)
XENIUM_UM_PER_PX = 0.2125
TARGET_UM_PER_PX = 0.5  # FineST target resolution for patch extraction
XENIUM_SCALE_TO_TARGET_UM = XENIUM_UM_PER_PX / TARGET_UM_PER_PX  # 0.425
DEFAULT_SEED = 666
LARGE_DATASET_THRESHOLD = 50000
LARGE_DATASET_STEP = 1000
SMALL_DATASET_STEP = 100

# dataset_class groups for scale-factor rules
_VISIUM_CLASSES = {'Visium'}
_VISIUMHD_CLASSES = {'VisiumHD'}

# Token/tile grid size implied by each histology foundation model
HIST_MODEL_TILE_SIZE = {
    'HIPT': 16,
    'Virchow2': 14,
    'UNI': 16,
}

## Set logging
logging.getLogger().setLevel(logging.INFO)


def setup_logger(model_save_folder: str, method: str = "HIPT") -> logging.Logger:
    """Setup logger with file and console handlers."""
    level = logging.INFO
    method_tag = str(method).strip() if method is not None else "HIPT"
    log_name = f'{method_tag}_image_feature_extract.log'
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(model_save_folder + log_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(
        os.path.join(model_save_folder, log_name), mode='a'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def setup_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logging_step(total_items: int) -> int:
    """Get logging step based on dataset size."""
    return LARGE_DATASET_STEP if total_items > LARGE_DATASET_THRESHOLD else SMALL_DATASET_STEP

##########################################
## 2026.07.06 LLY add the default_logging_folder for the log files
##########################################
def default_logging_folder(output_pth: str) -> str:
    """
    Derive the log directory from ``output_pth``.

    Logs are written under ``<parent_of_output_pth>/Logging/``, e.g.::

        output_pth = 'FineST_tutorial_data/ImgEmbeddings/HIPT/pth_64_16'
        -> 'FineST_tutorial_data/ImgEmbeddings/HIPT/Logging/'

        output_pth = 'Dataset/CRC16um/ImgEmbeddings/HIPT/pth_32_16'
        -> 'Dataset/CRC16um/ImgEmbeddings/HIPT/Logging/'
    """
    parent = os.path.dirname(str(output_pth).rstrip('/\\'))
    if not parent:
        parent = '.'
    return os.path.join(parent, 'Logging') + os.sep


def hist_model_tile_size(hist_model: str) -> int:
    """Return the token/tile grid size used by ``hist_model`` (HIPT=16, Virchow2=14)."""
    key = str(hist_model).strip()
    if key not in HIST_MODEL_TILE_SIZE:
        raise ValueError(
            f"Unsupported hist_model={hist_model!r}. "
            f"Expected one of {sorted(HIST_MODEL_TILE_SIZE)}"
        )
    return HIST_MODEL_TILE_SIZE[key]

##########################################
## 2026.08.04 LLY add the format_coord_for_filename for the filename format
##########################################
def format_coord_for_filename(value) -> str:
    """
    Format a pixel coordinate for patch / ``.pth`` filenames.

    * Whole-number Visium coords (e.g. from ``tissue_positions_list.csv``)
      → integer token ``'10014'`` (not ``'10014.0'``), so
      ``get_image_coord(..., ST_class='Visium')`` can parse with ``int``.
    * True fractional coords (e.g. interpolated ``tissue_positions_list_add.csv``)
      → compact float token such as ``'10014.5'``.
    """
    fv = float(value)
    if abs(fv - round(fv)) < 1e-6:
        return str(int(round(fv)))
    return f"{fv:.6f}".rstrip('0').rstrip('.')


def default_output_paths(
    data_save_dir: str,
    hist_model: str,
    patch_size: int,
    output_name: Optional[str] = None,
):
    """
    Build default image-patch and embedding directories.

    Layout::

        <data_save_dir>/ImgEmbeddings/<hist_model>/<output_name>/
        <data_save_dir>/ImgEmbeddings/<hist_model>/<output_name>_image/

    Default ``output_name`` is ``pth_<patch_size>_<tile>``, where ``tile`` is
    16 for HIPT/UNI and 14 for Virchow2.

    Examples
    --------
    >>> default_output_paths('FineST_tutorial_data', 'HIPT', 64)
    ('FineST_tutorial_data/ImgEmbeddings/HIPT/pth_64_16_image',
     'FineST_tutorial_data/ImgEmbeddings/HIPT/pth_64_16')
    >>> default_output_paths('Dataset/CRC16um', 'Virchow2', 28)
    ('Dataset/CRC16um/ImgEmbeddings/Virchow2/pth_28_14_image',
     'Dataset/CRC16um/ImgEmbeddings/Virchow2/pth_28_14')
    >>> default_output_paths('FineST_tutorial_data', 'HIPT', 64, output_name='NEW_pth_64_16')
    ('FineST_tutorial_data/ImgEmbeddings/HIPT/NEW_pth_64_16_image',
     'FineST_tutorial_data/ImgEmbeddings/HIPT/NEW_pth_64_16')
    """
    tile = hist_model_tile_size(hist_model)
    if output_name is None or str(output_name).strip() == '':
        output_name = f'pth_{int(patch_size)}_{tile}'
    else:
        output_name = str(output_name).strip().rstrip('/\\')
    model_dir = os.path.join(
        str(data_save_dir).rstrip('/\\'), 'ImgEmbeddings', str(hist_model).strip()
    )
    output_pth = os.path.join(model_dir, output_name)
    output_img = output_pth + '_image'
    return output_img, output_pth



##########################################
# 2026.04.14 LLY add but not used for now
# The CODEX HE, HE_micron_per_pixel_size=0.5000061356695383 um / 1 pixel, so no need to rescale
##########################################
## Important for rescale factor to keep 0.5 um/pixel
def get_scale_factor(HE_micron_per_pixel_size: float, logger: Optional[logging.Logger] = None) -> float:
    """
    Calculate the scale factor to rescale an image to a standard resolution of 0.5 μm/pixel.

    ``scale = microns_per_pixel / 0.5``
    """
    standard_pixel_size = TARGET_UM_PER_PX  # 0.5 um/pixel
    if abs(HE_micron_per_pixel_size - standard_pixel_size) < 1e-3:
        scale = 1.0
    else:
        scale = HE_micron_per_pixel_size / standard_pixel_size
    msg = f"scale factor for rescale image to 0.5 um/pixel: {scale:.6f}"
    if logger is not None:
        logger.info(msg)
    else:
        logging.info(msg)
    return scale


def load_st_scalefactors(STfactor_path: str) -> dict:
    """
    Load ``scalefactors_json.json``.

    Parameters
    ----------
    STfactor_path : str
        Path to the JSON file itself, or to a directory that contains
        ``scalefactors_json.json``.
    """
    path = os.path.expanduser(str(STfactor_path))
    if os.path.isdir(path):
        path = os.path.join(path, 'scalefactors_json.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"STfactor file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def microns_per_pixel_from_stfactor(sf: dict, dataset_class: str) -> float:
    """
    Derive full-resolution µm/pixel from Space Ranger scalefactors.

    * **Visium**
      ``microns_per_pixel = 55 / spot_diameter_fullres``
    * **VisiumHD**
      read ``microns_per_pixel`` directly from the JSON
      (fallback: ``bin_size_um / spot_diameter_fullres`` if present)
    """
    if dataset_class is None:
        raise ValueError("dataset_class is required to interpret STfactor_path")

    if dataset_class in _VISIUM_CLASSES:
        if 'spot_diameter_fullres' not in sf:
            raise KeyError(
                "Visium scalefactors must contain 'spot_diameter_fullres' "
                f"(keys={list(sf.keys())})"
            )
        return float(VISIUM_SPOT_DIAMETER_UM) / float(sf['spot_diameter_fullres'])

    if dataset_class in _VISIUMHD_CLASSES:
        if 'microns_per_pixel' in sf:
            return float(sf['microns_per_pixel'])
        if 'bin_size_um' in sf and 'spot_diameter_fullres' in sf:
            return float(sf['bin_size_um']) / float(sf['spot_diameter_fullres'])
        raise KeyError(
            "VisiumHD scalefactors must contain 'microns_per_pixel' "
            f"(or bin_size_um + spot_diameter_fullres); keys={list(sf.keys())}"
        )

    raise ValueError(
        f"Unsupported dataset_class={dataset_class!r}. "
        f"Expected 'Visium' or 'VisiumHD'."
    )


def resolve_scale_from_stfactor(
    STfactor_path: str,
    dataset_class: str,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Compute ``--scale`` so that each pixel is 0.5 µm after rescaling.

    Equivalent to the previous manual workflow::

        scale = microns_per_pixel / 0.5
    """
    sf = load_st_scalefactors(STfactor_path)
    um_per_px = microns_per_pixel_from_stfactor(sf, dataset_class)
    if logger is not None:
        logger.info(
            "STfactor: dataset_class=%s, microns_per_pixel=%.6f, target=%.1f um/px",
            dataset_class, um_per_px, TARGET_UM_PER_PX,
        )
        if dataset_class in _VISIUM_CLASSES and 'spot_diameter_fullres' in sf:
            logger.info(
                "spot_diameter_fullres=%.6f -> microns_per_pixel = %.1f / %.6f = %.6f",
                float(sf['spot_diameter_fullres']),
                VISIUM_SPOT_DIAMETER_UM,
                float(sf['spot_diameter_fullres']),
                um_per_px,
            )
    return get_scale_factor(um_per_px, logger=logger)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'on')
    

## Rescale image to decrease split_num
def rescale_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Rescale image to decrease split_num."""
    if img.ndim == 2:
        scale_params = [scale, scale]
    elif img.ndim == 3:
        scale_params = [scale, scale, 1]
    else:
        raise ValueError(f'Unrecognized image ndim: {img.ndim}')
    img = rescale(img, scale_params, preserve_range=True)
    return img

## get integer nearest 'multiple of 14' to 'spot diameter'
# def get_patch_size(diameter, tile_size=14):
#     return int((diameter // tile_size) * tile_size)

def _scale_positions_for_image(
    tissue_position: pd.DataFrame, scale_image: bool, scale: float
) -> pd.DataFrame:
    if not scale_image:
        return tissue_position
    if "pxl_row_in_fullres" not in tissue_position.columns:
        return tissue_position
    out = tissue_position.copy()
    out["pxl_row_in_fullres"] = out["pxl_row_in_fullres"] * scale
    out["pxl_col_in_fullres"] = out["pxl_col_in_fullres"] * scale
    return out


def load_tissue_position(position_path: str, scale_image: bool, scale: float, logger: logging.Logger) -> pd.DataFrame:
    """Load and process tissue position file."""
    _, ext = os.path.splitext(position_path)
    
    if ext == ".csv":
        tissue_position = pd.read_csv(position_path)
        logger.info(f"Loaded CSV with shape: {tissue_position.shape}")

        #########################################################
        ## 2026.05.20 Xenium Select4 — VUILD96LA (coords: x_centroid, y_centroid)
        ## Xenium / Weiqin HE: x_centroid,y_centroid in µm; X_pix_HE,Y_pix_HE on HE grid
        if "x_centroid" in tissue_position.columns and "y_centroid" in tissue_position.columns:
            tissue_position = tissue_position.copy()
            if "cell_id" in tissue_position.columns:
                tissue_position = tissue_position.set_index("cell_id", drop=False)
            if "X_pix_HE" in tissue_position.columns and "Y_pix_HE" in tissue_position.columns:
                tissue_position["pxl_row_in_fullres"] = tissue_position["X_pix_HE"]
                tissue_position["pxl_col_in_fullres"] = tissue_position["Y_pix_HE"]
                src = "X_pix_HE/Y_pix_HE"
            else:
                tissue_position["pxl_row_in_fullres"] = (
                    tissue_position["x_centroid"] / XENIUM_UM_PER_PX
                )
                tissue_position["pxl_col_in_fullres"] = (
                    tissue_position["y_centroid"] / XENIUM_UM_PER_PX
                )
                src = f"x_centroid/y_centroid / {XENIUM_UM_PER_PX}"
            tissue_position = _scale_positions_for_image(
                tissue_position, scale_image, scale
            )
            logger.info(
                f"Mapped {src} -> pxl_row/pxl_col (HE full-res px); "
                f"scale_image={scale_image}, scale={scale:.4f} "
                f"({len(tissue_position):,} cells)"
            )
            return tissue_position
        #########################################################

        #########################################################            
        ## 2026.05.20 Xenium Select4 — VUILD96LA (coords: x_centroid, y_centroid) For Lu’s StarDist
        ## StarDist on HE: centroid_x, centroid_y = full-resolution HE pixel coordinates
        if "centroid_x" in tissue_position.columns and "centroid_y" in tissue_position.columns:
            tissue_position = tissue_position.copy()
            tissue_position["pxl_row_in_fullres"] = tissue_position["centroid_x"]
            tissue_position["pxl_col_in_fullres"] = tissue_position["centroid_y"]
            tissue_position = _scale_positions_for_image(
                tissue_position, scale_image, scale
            )
            logger.info(
                "Mapped StarDist centroid_x/centroid_y -> pxl_row/pxl_col (HE full-res px); "
                f"scale_image={scale_image}, scale={scale:.4f} "
                f"({len(tissue_position):,} detections)"
            )
            return tissue_position
        #########################################################

        #########################################################
        ## 2026.07.06 CODEX HCC s4769 — {acq_id}.cell_data.csv: X,Y on aligned HE (full-res px)
        if "X" in tissue_position.columns and "Y" in tissue_position.columns:
            tissue_position = tissue_position.copy()
            if "CELL_ID" in tissue_position.columns:
                tissue_position = tissue_position.set_index("CELL_ID", drop=False)
            tissue_position["pxl_row_in_fullres"] = tissue_position["X"]
            tissue_position["pxl_col_in_fullres"] = tissue_position["Y"]
            tissue_position = _scale_positions_for_image(
                tissue_position, scale_image, scale
            )
            logger.info(
                "Mapped CODEX cell_data X/Y -> pxl_row/pxl_col (HE full-res px); "
                f"scale_image={scale_image}, scale={scale:.4f} "
                f"({len(tissue_position):,} cells)"
            )
            return tissue_position
        #########################################################
        
        if tissue_position.shape[1] == 6:
            if 'cell_nums' not in tissue_position.columns.tolist():
                ## For within spot
                tissue_position = pd.read_csv(position_path, header=None).set_index(0)
                tissue_position.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
                tissue_position = tissue_position.rename(
                    columns={
                        'pxl_row_in_fullres': 'pxl_col_in_fullres', 
                        'pxl_col_in_fullres': 'pxl_row_in_fullres'
                    }
                )
                tissue_position = tissue_position[tissue_position['in_tissue'] == 1]
            else:
                ## For single-nuclei file, don't need rename the columns
                tissue_position = pd.read_csv(position_path).set_index("Unnamed: 0")
        elif tissue_position.shape[1] == 5:
            # For between spot (from Spot_interpolation.py) or single nuclei
            # Note: Spot_interpolation.py generates CSV with column names matching original Visium format,
            #       but we need to swap the coordinate columns to match the coordinate system used
            #       in Image_feature_extraction.py (which swaps coordinates for original Visium data)
            tissue_position = pd.read_csv(position_path)
            # Check if first column is index (Unnamed: 0) or actual data
            if tissue_position.columns[0] == 'Unnamed: 0':
                tissue_position = tissue_position.set_index("Unnamed: 0")
            
            # Check if columns already have correct names (from Spot_interpolation.py)
            if 'pxl_row_in_fullres' in tissue_position.columns and 'pxl_col_in_fullres' in tissue_position.columns:
                # Swap coordinate columns to match the coordinate system used for original Visium data
                # This ensures consistency with how Image_feature_extraction.py processes original Visium files
                # (which swaps pxl_row_in_fullres <-> pxl_col_in_fullres)
                tissue_position = tissue_position.rename(
                    columns={
                        'pxl_row_in_fullres': 'pxl_col_in_fullres', 
                        'pxl_col_in_fullres': 'pxl_row_in_fullres'
                    }
                )
                logger.info("Detected interpolated spots file: swapped coordinate columns to match image coordinate system")
            else:
                # Need to set column names (for single-nuclei files or other formats)
                # For single-nuclei files, we may need to swap coordinates
                if len(tissue_position.columns) == 4:
                    tissue_position.columns = ['array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
                    # For single-nuclei files, swap coordinates to match image coordinate system
                    tissue_position = tissue_position.rename(
                        columns={
                            'pxl_row_in_fullres': 'pxl_col_in_fullres', 
                            'pxl_col_in_fullres': 'pxl_row_in_fullres'
                        }
                    )
                else:
                    # Assume standard order: array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
                    tissue_position.columns = ['array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
        tissue_position = _scale_positions_for_image(
            tissue_position, scale_image, scale
        )
    elif ext == ".parquet":
        tissue_position = (pd.read_parquet(position_path)
                        .set_index('barcode')
                        .rename(columns={
                            'pxl_row_in_fullres': 'pxl_col_in_fullres', 
                            'pxl_col_in_fullres': 'pxl_row_in_fullres'
                        })
                        .query('in_tissue == 1'))
        if scale_image:
            tissue_position['pxl_col_in_fullres'] = tissue_position['pxl_col_in_fullres'] * scale
            tissue_position['pxl_row_in_fullres'] = tissue_position['pxl_row_in_fullres'] * scale
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return tissue_position


def image_feature_extraction(
    position_path: str,
    rawimage_path: str,
    hist_model: str,
    patch_size: int,
    data_save_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    output_img: Optional[str] = None,
    output_pth: Optional[str] = None,
    dataset_class: Optional[str] = None,
    STfactor_path: Optional[str] = None,
    is_05umperpix: bool = False,
    logging_folder: Optional[str] = None,
    scale_image: bool = False,
    scale: Optional[float] = None,
):
    """
    Extract image features from spatial transcriptomics data.

    Public FineST API. After ``pip install finest``::

        import FineST as fst
        fst.image_feature_extraction(...)

    Or from the terminal::

        python -m FineST.image_feature_extraction --data_save_dir FineST_tutorial_data --hist_model HIPT ...

    Parameters
    ----------
    position_path : str
        Path to tissue position file (.csv or .parquet).
    rawimage_path : str
        Path to the raw H&E image (.tif, .btf, etc.).
    hist_model : str
        Histology foundation model: 'HIPT', 'Virchow2', or 'UNI'.
    patch_size : int
        Patch size in pixels (e.g. 64 for HIPT, 112 for Virchow2).
    data_save_dir : str, optional
        Dataset root used to auto-create outputs under
        ``<data_save_dir>/ImgEmbeddings/<hist_model>/``. Required unless both
        ``output_img`` and ``output_pth`` are provided.
    output_name : str, optional
        Folder name under ``ImgEmbeddings/<hist_model>/``.
        Default: ``pth_<patch_size>_<tile>`` (tile=16 for HIPT/UNI, 14 for Virchow2).
        Use e.g. ``NEW_pth_64_16`` or ``sc_pth_16_16`` for between-spot / nuclei runs.
    output_img, output_pth : str, optional
        Optional explicit output directories (override ``data_save_dir`` auto paths).
    dataset_class : str, optional
        ``'Visium'`` or ``'VisiumHD'``. Required when ``is_05umperpix=True``.
    STfactor_path : str, optional
        Path to ``scalefactors_json.json``. Required when ``is_05umperpix=True``.
    is_05umperpix : bool, optional
        Auto-compute scale to 0.5 µm/pixel and enable ``scale_image``.
    logging_folder : str, optional
        Log directory. Default: ``<parent_of_output_pth>/Logging/``.
    scale_image : bool, optional
        Manual rescale flag (forced True when ``is_05umperpix=True``).
    scale : float, optional
        Manual scale when ``is_05umperpix=False``.

    Examples
    --------
    >>> image_feature_extraction(
    ...     data_save_dir='FineST_tutorial_data',
    ...     dataset_class='Visium',
    ...     position_path='FineST_tutorial_data/spatial/tissue_positions_list.csv',
    ...     rawimage_path='FineST_tutorial_data/image.tif',
    ...     STfactor_path='FineST_tutorial_data/spatial/scalefactors_json.json',
    ...     is_05umperpix=True,
    ...     hist_model='HIPT',
    ...     patch_size=64,
    ... )
    """
    is_05umperpix = _as_bool(is_05umperpix, default=False)
    scale_image = _as_bool(scale_image, default=False)

    if output_img is None or output_pth is None:
        if not data_save_dir:
            raise ValueError(
                "Provide data_save_dir to auto-create ImgEmbeddings/<hist_model>/..., "
                "or pass both output_img and output_pth explicitly"
            )
        auto_img, auto_pth = default_output_paths(
            data_save_dir, hist_model, patch_size, output_name=output_name
        )
        if output_img is None:
            output_img = auto_img
        if output_pth is None:
            output_pth = auto_pth

    # Prefix for patch / embedding filenames (derived from output_pth)
    dataset = os.path.basename(str(output_pth).rstrip('/\\')) or 'sample'

    if logging_folder is None:
        logging_folder = default_logging_folder(output_pth)
    elif not str(logging_folder).endswith(('/', '\\')):
        logging_folder = str(logging_folder) + os.sep

    # Create the folder with a unique timestamp
    dir_name = os.path.join(
        logging_folder.rstrip('/\\'),
        datetime.now().strftime('%Y%m%d%H%M%S%f'),
    )
    os.makedirs(dir_name, exist_ok=True)
    logger = setup_logger(dir_name, method=hist_model)
    logger.info("logging_folder: %s", logging_folder)
    logger.info("log_dir: %s", dir_name)
    logger.info("dataset=%s, dataset_class=%s, hist_model=%s", dataset, dataset_class, hist_model)
    logger.info("output_img: %s", output_img)
    logger.info("output_pth: %s", output_pth)

    # Resolve scale: prefer automatic 0.5 µm/pixel from Space Ranger JSON
    if is_05umperpix:
        if not STfactor_path:
            raise ValueError("is_05umperpix=True requires STfactor_path (scalefactors_json.json)")
        if not dataset_class:
            raise ValueError(
                "is_05umperpix=True requires dataset_class "
                "('Visium' or 'VisiumHD')"
            )
        scale = resolve_scale_from_stfactor(STfactor_path, dataset_class, logger=logger)
        scale_image = True
        logger.info(
            "is_05umperpix=True -> scale_image=True, scale=%.6f (from %s)",
            scale, STfactor_path,
        )
    else:
        if scale is None:
            scale = DEFAULT_SCALE
        logger.info("is_05umperpix=False -> scale_image=%s, scale=%s", scale_image, scale)

    # Set seed for reproducibility
    setup_seed(DEFAULT_SEED)

    # Set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load tissue position file
    # Note: When scale_image=True and position file is .parquet format,
    #       coordinates are automatically scaled by the 'scale' factor
    try:
        tissue_position = load_tissue_position(position_path, scale_image, scale, logger)
        logger.info(f'tissue_position: \n {tissue_position.head()}')
    except Exception as e:
        logger.error(f"Error loading tissue position file: {e}")
        raise

    ##############################################
    # different, need math with figure 
    ##############################################
    coordinates = list(zip(
        tissue_position["pxl_row_in_fullres"], 
        tissue_position["pxl_col_in_fullres"]
    ))
    logger.info(f'tissue_position number: {len(coordinates)}')
    logger.info(
        f'tissue_position range: '
        f'{tissue_position["pxl_row_in_fullres"].max()} '
        f'{tissue_position["pxl_col_in_fullres"].max()}'
    )

    # Load and optionally scale image
    # When scale_image=True: Image is downsampled by 'scale' factor to reduce processing time/memory
    # When scale_image=False: Image is used at original resolution
    if scale_image:
        logger.info(f'Loading image with scaling enabled (scale factor: {scale:.3f})')
        logger.info('This will reduce image size to speed up processing and reduce memory usage')
        image_obj = Image.open(rawimage_path)
        image = np.array(image_obj)

        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]  # remove alpha channel
        image = image.astype(np.float32)
        logger.info(f'Rescaling image (scale: {scale:.3f})...')
        image = rescale_image(image, scale)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)  # NumPy to PIL
        logger.info('Rescaling image DONE!')
    else:
        logger.info('Loading image at original resolution (no scaling)')
        image = Image.open(rawimage_path)

    image_width, image_height = image.size
    logger.info(f"image_width, image_height: {image_width}, {image_height}")

    ## Using spot_diamer as patch_size
    # with open(str(json_file)) as file:
    #     scalefactors = json.load(file)
    # patch_size = get_patch_size(scalefactors['spot_diameter_fullres'])

    ## Create patches
    # patch_size = 32 for Visium HD, patch_size = 64 for Visium (V2)
    patch_size = int(patch_size)
    os.makedirs(output_img, exist_ok=True)

    start_time = time.time()
    for i, point in enumerate(coordinates):
        x, y = point
        left = x - patch_size // 2
        upper = y - patch_size // 2
        right = x + patch_size // 2
        lower = y + patch_size // 2
        if left < 0 or upper < 0 or right > image_width or lower > image_height:
            continue
        patch = image.crop((left, upper, right, lower))
        
        # When scale_image=True, patch filename uses original (unscaled) coordinates
        # This ensures consistency with original position files
        if scale_image:
            x_name = x / scale  # Convert back to original coordinate system
            y_name = y / scale
        else:
            x_name, y_name = x, y
        # Visium integer coords -> '10014'; interpolated floats -> '10014.5'
        patch_name = (
            f"{dataset}_{format_coord_for_filename(x_name)}_"
            f"{format_coord_for_filename(y_name)}.png"
        )

        step = get_logging_step(len(coordinates))
        if i % step == 0:
            logger.info(f"patch_name: {i}, {patch_name}")
        patch.save(os.path.join(output_img, patch_name))

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Image segmentation time: {execution_time:.2f} seconds")


    ######################################################################
    # HIPT-vit_256(): 
    # from size '3 x patch_size x patch_size' to size '1 x 384' (16*16)
    ######################################################################
    # Note: tissue_position["pxl_row_in_fullres"].max(), tissue_position["pxl_col_in_fullres"].max()
    # should be consistent with {image_width, image_height}
    # Please check it !!!

    # HIPT ships inside the FineST package (pip-installable)
    try:
        from .HIPT.HIPT_4K import vision_transformer as vits
    except ImportError:
        _pkg_root = os.path.dirname(os.path.abspath(__file__))
        if _pkg_root not in sys.path:
            sys.path.insert(0, _pkg_root)
        from HIPT.HIPT_4K import vision_transformer as vits

    ######################################################################
    # HIPT-vit_256(): 
    # from size '3 x patch_size x patch_size' to size '1 x 384' (16*16)
    ######################################################################
    # https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/hipt_model_utils.py#L39
    def get_vit256(pretrained_weights: str, arch: str = 'vit_small', 
                   device: Optional[torch.device] = None) -> torch.nn.Module:
        r"""
        Builds ViT-256 Model.
        
        Args:
        - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
        - arch (str): Which model architecture.
        - device (torch): Torch device to save model.
        
        Returns:
        - model256 (torch.nn): Initialized model.
        """
        
        checkpoint_key = 'teacher'
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
        for p in model256.parameters():
            p.requires_grad = False
        model256.eval()
        model256.to(device)

        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model256.load_state_dict(state_dict, strict=False)
            logger.info(f'Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}')
            
        return model256

    # Load model and setup transforms
    if hist_model == 'HIPT':
        logger.info(f"hist_model: {hist_model}")
        weight_path = "https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth"
        model = get_vit256(pretrained_weights=weight_path, device=device)

        # https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/hipt_model_utils.py#L111
        def eval_transforms():
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            return transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=mean, std=std)
            ])
        logger.info(f"HIPT model loaded successfully and transform created")

    elif hist_model == 'Virchow2':
        logger.info(f"hist_model: {hist_model}")
        # Import timm only when using Virchow2 method
        try:
            import timm
            from timm.layers import SwiGLUPacked
        except ImportError:
            error_msg = (
                "timm package is required for Virchow2 method. "
                "Please install it with: pip install timm"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        ######################################################################
        # Virchow2(): 
        # from size '3 x patch_size x patch_size' to size '1 x 1280' (14*14)
        ######################################################################
        # Virchow2, need to specify MLP layer and activation function for proper init
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", 
            pretrained=True, 
            mlp_layer=SwiGLUPacked, 
            act_layer=torch.nn.SiLU
        )
        model.to(device)
        model = model.eval()

        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/constants.py#L3
        def eval_transforms():
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            return transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=mean, std=std)
            ])
        logger.info(f"Virchow2 model loaded successfully and transform created")

    elif hist_model == 'UNI':
        logger.info(f"hist_model: {hist_model}")

        # Import timm only when using UNI method
        try:
            import timm
            from huggingface_hub import hf_hub_download
        except ImportError:
            error_msg = (
                "timm package is required for UNI method. "
                "Please install it with: pip install timm"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        ######################################################################
        # UNI(): 
        # from size '3 x patch_size x patch_size' to size '1 x 1024' (16*16)
        ######################################################################    
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(
            script_dir,
            "UNI",
            "assets",
            "ckpts",
            "vit_large_patch16_224.dinov2.uni_mass100k"
        )
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        weight_file = hf_hub_download(
            repo_id="MahmoodLab/UNI",
            filename="pytorch_model.bin",
            local_dir=local_dir,
            force_download=False    # True: download even if already in cache
        )

        model = timm.create_model(
            "vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True
        )
        model.load_state_dict(torch.load(weight_file, map_location="cpu"), strict=True)

        def eval_transforms():
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            return transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        model.to(device)
        model = model.eval()

        ######################################################################
        # 2026.04.14 LLY add - used for no local model weights
        ######################################################################
        # # Import timm only when using UNI method
        # try:
        #     import timm
        #     from timm.data import resolve_data_config
        #     from timm.data.transforms_factory import create_transform
        #     # from huggingface_hub import login
        #     # login() # login with your User Access Token for UNI, found at https://huggingface.co/settings/tokens
        # except ImportError:
        #     error_msg = (
        #         "timm package is required for UNI method. "
        #         "Please install it with: pip install timm"
        #     )
        #     logger.error(error_msg)
        #     raise ImportError(error_msg)
        
        # ######################################################################
        # # UNI(): 
        # # from size '3 x patch_size x patch_size' to size '1 x 1024' (16*16)
        # # pretrained=True needed to load UNI weights (and download weights for the first time)
        # # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        # ## Note!!! Automatically download model weights to the huggingface_hub cache in your home directory.
        # ##         "~/.cache/huggingface/hub/models--MahmoodLab--UNI"
        # ######################################################################        
        # model = timm.create_model(
        #     "hf-hub:MahmoodLab/uni", 
        #     pretrained=True, 
        #     init_values=1e-5, 
        #     dynamic_img_size=True
        # )
        # model.to(device)
        # transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        # model = model.eval()

        logger.info(f"UNI model loaded successfully and transform created")

    else:
        raise ValueError(f"Unsupported hist_model: {hist_model}")

    # transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    # ## Create transforms and Normalize to match the expected input format of the model
    # transforms = transforms.Compose([
    #     # transforms.Resize((224, 224)),  # Resize to match the expected input size of the model
    #     transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    # ])

    # Process patches
    os.makedirs(output_pth, exist_ok=True)
    patches_list = os.listdir(output_img)
    step = get_logging_step(len(patches_list))
    transform_fn = eval_transforms()

    start_time = time.time()
    with torch.inference_mode():
        for i, patch in enumerate(patches_list):

            patch_base_name, extension = os.path.splitext(patch)
            patch_path = os.path.join(output_img, patch)
            with Image.open(patch_path) as patch_image:
                patch_image = patch_image.convert("RGB")

                if hist_model == 'HIPT':
                    p_image = transform_fn(patch_image).unsqueeze(dim=0).to(device, non_blocking=True)    # torch.Size([1, 3, 64, 64])
                    lay = model.get_intermediate_layers(p_image, 1)[0]  # torch.Size([1, 17, 384])
                    subtensors = lay[:, :, :]  # torch.Size([1, 17, 384])
                    subtensors_list = torch.split(subtensors, 1, dim=1)
                    subtensors_list = subtensors_list[1:]

                elif hist_model == 'Virchow2':
                    p_image = transform_fn(patch_image).unsqueeze(dim=0).to(device, non_blocking=True)    # torch.Size([1, 3, 64, 64])
                    lay = model(p_image)  # size: 1 x 261 x 1280
                    # tokens 1-4 are register tokens so we ignore those
                    subtensors = lay[:, 5:]  # size: 1 x 256 x 1280
                    subtensors_list = torch.split(subtensors, 1, dim=1)
       
                elif hist_model == 'UNI':
                    p_image = transform_fn(patch_image).unsqueeze(dim=0).to(device, non_blocking=True)    # torch.Size([1, 3, 64, 64])
                    lay = model(p_image)  # size: 1 x 1024
                    subtensors_list = (lay.unsqueeze(1),)  # size: 1 x 1 x 1024

            # Save image embeddings
            saved_name = patch_base_name + '.pth'
            if i % step == 0:
                logger.info(f"saved_name: {i}, {saved_name}")
            saved_path = os.path.join(output_pth, saved_name)
            torch.save(subtensors_list, saved_path)

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Feature extraction time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract H&E image features for FineST (HIPT / Virchow2 / UNI).'
    )
    parser.add_argument(
        '--dataset_class',
        default=None,
        choices=['Visium', 'VisiumHD'],
        help="Platform class for 0.5 µm/pixel scale rules: Visium or VisiumHD",
    )
    parser.add_argument('--position_path', required=True, help='Tissue position file path')
    parser.add_argument('--rawimage_path', required=True, help='H&E image path')
    parser.add_argument('--STfactor_path', default=None,
                        help='Path to scalefactors_json.json (file or directory)')
    parser.add_argument('--is_05umperpix', default='False',
                        help='Auto-compute scale to 0.5 µm/pixel from STfactor_path (true/false)')
    parser.add_argument('--hist_model', required=True,
                        help='Histology foundation model: HIPT, Virchow2, or UNI')
    parser.add_argument('--patch_size', required=True, type=int,
                        help='Patch size for image segmentation')
    parser.add_argument(
        '--data_save_dir',
        default=None,
        help='Dataset root; auto-creates ImgEmbeddings/<hist_model>/pth_<patch>_<tile>/',
    )
    parser.add_argument(
        '--output_name',
        default=None,
        help='Optional folder name under ImgEmbeddings/<hist_model>/ '
             '(default: pth_<patch_size>_<tile>; e.g. NEW_pth_64_16, sc_pth_16_16)',
    )
    parser.add_argument('--output_img', default=None,
                        help='Optional explicit patch output dir (overrides data_save_dir)')
    parser.add_argument('--output_pth', default=None,
                        help='Optional explicit embedding output dir (overrides data_save_dir)')
    parser.add_argument('--scale_image', default='False',
                        help='Whether to scale/downsample the image (true/false). '
                             'Forced True when is_05umperpix=True. Default: False')
    parser.add_argument('--scale', type=float, default=None,
                        help='Manual scale factor when is_05umperpix=False and scale_image=True')
    parser.add_argument(
        '--logging',
        default=None,
        help='Optional log directory. Default: <parent_of_output_pth>/Logging/',
    )
    args = parser.parse_args()

    image_feature_extraction(
        dataset_class=args.dataset_class,
        position_path=args.position_path,
        rawimage_path=args.rawimage_path,
        STfactor_path=args.STfactor_path,
        is_05umperpix=args.is_05umperpix,
        hist_model=args.hist_model,
        patch_size=args.patch_size,
        data_save_dir=args.data_save_dir,
        output_name=args.output_name,
        output_img=args.output_img,
        output_pth=args.output_pth,
        logging_folder=args.logging,
        scale_image=args.scale_image,
        scale=args.scale,
    )


# Backward-compatible alias for older demo / script call sites
main = image_feature_extraction
