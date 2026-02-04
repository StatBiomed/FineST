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
## Note: timm is only imported when method='Virchow2' is used
## If you plan to use Virchow2, install it with: pip install timm
## if use transforms in Virchow2
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform  
#################################################################


## Constants
DEFAULT_SCALE = 0.5
DEFAULT_SEED = 666
LARGE_DATASET_THRESHOLD = 50000
LARGE_DATASET_STEP = 1000
SMALL_DATASET_STEP = 100

## Set logging
logging.getLogger().setLevel(logging.INFO)


def setup_logger(model_save_folder: str) -> logging.Logger:
    """Setup logger with file and console handlers."""
    level = logging.INFO
    log_name = 'HIPT_image_feature_extract.log'
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

def load_tissue_position(position_path: str, scale_image: bool, scale: float, logger: logging.Logger) -> pd.DataFrame:
    """Load and process tissue position file."""
    _, ext = os.path.splitext(position_path)
    
    if ext == ".csv":
        tissue_position = pd.read_csv(position_path)
        logger.info(f"Loaded CSV with shape: {tissue_position.shape}")
        
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


def main(dataset: str, position_path: str, rawimage_path: str, scale_image: bool, 
         method: str, patch_size: int, output_img: str, output_pth: str, 
         logging_folder: str, scale: float = DEFAULT_SCALE):
    """
    Extract image features from spatial transcriptomics data.
    
    This function processes H&E stained images and extracts feature embeddings for each spot
    or cell location using either HIPT or Virchow2 models.
    
    Parameters
    ----------
    dataset : str
        Dataset name (used for naming output files)
    position_path : str
        Path to tissue position file (.csv or .parquet format)
        - For Visium: typically 'tissue_positions_list.csv'
        - For Visium HD: typically 'tissue_positions.parquet'
    rawimage_path : str
        Path to the raw H&E stained image file (.tif, .btf, etc.)
    scale_image : bool
        Whether to scale (downsample) the image before processing.
        
        **When to use scale_image=True:**
        - For Visium HD data with very large images (e.g., .btf files)
        - When processing time or memory is a concern
        - When the original image resolution is higher than needed
        - Example: Visium HD 16um data often uses scale_image=True with scale=0.5
        
        **When to use scale_image=False:**
        - For standard Visium data (typically .tif files)
        - When maximum resolution is required
        - When the image size is manageable
        - Example: Standard Visium data typically uses scale_image=False
        
        **Important:** When scale_image=True:
        - The image will be resized by the factor specified in 'scale' parameter
        - For .parquet position files, coordinates will be automatically scaled to match
        - Output patch filenames will use original (unscaled) coordinates for consistency
    method : str
        Feature extraction method: 'HIPT' or 'Virchow2'
        - 'HIPT': Uses HIPT vision transformer (vit256_small_dino)
        - 'Virchow2': Uses Virchow2 model from Hugging Face
    patch_size : int
        Size of image patches to extract (in pixels)
        - For Visium: typically 112 (for Virchow2) or 64 (for HIPT)
        - For Visium HD: typically 28 (for 16um bins with Virchow2)
        - For single-cell resolution: typically 14 or 16
    output_img : str
        Output directory for extracted image patches (.png files)
    output_pth : str
        Output directory for extracted feature embeddings (.pth files)
    logging_folder : str
        Directory for log files
    scale : float, optional
        Image scaling factor when scale_image=True (default: 0.5)
        
        **Usage with scale_image:**
        - scale=0.5: Resize image to 50% of original size (common for Visium HD)
        - scale=0.25: Resize image to 25% of original size (for very large images)
        - Only used when scale_image=True
        - For .parquet files, position coordinates are automatically scaled by this factor
        
        **Example scenarios:**
        1. Visium HD 16um with large .btf image:
           scale_image=True, scale=0.5
           
        2. Standard Visium with .tif image:
           scale_image=False (scale parameter ignored)
           
        3. Very large image needing aggressive downsampling:
           scale_image=True, scale=0.25
    
    Notes
    -----
    - When scale_image=True, the image is downsampled to reduce processing time and memory
    - Position coordinates in .parquet files (usually for VisiumHD data) are automatically adjusted when scale_image=True
    - Output patch filenames preserve original coordinates (divided by scale) for consistency
    - The scale parameter only affects processing when scale_image=True
    
    Examples
    --------
    >>> # Standard Visium (no scaling)
    >>> main(
    ...     dataset='NPC',
    ...     position_path='tissue_positions_list.csv',
    ...     rawimage_path='image.tif',
    ...     scale_image=False,
    ...     method='Virchow2',
    ...     patch_size=112,
    ...     output_img='./patches',
    ...     output_pth='./embeddings',
    ...     logging_folder='./logs'
    ... )
    
    >>> # Visium HD (with scaling)
    >>> main(
    ...     dataset='HD_CRC_16um',
    ...     position_path='tissue_positions.parquet',
    ...     rawimage_path='image.btf',
    ...     scale_image=True,
    ...     method='Virchow2',
    ...     patch_size=28,
    ...     output_img='./patches',
    ...     output_pth='./embeddings',
    ...     logging_folder='./logs',
    ...     scale=0.5
    ... )
    """
    # Create the folder with a unique timestamp
    dir_name = logging_folder + datetime.now().strftime('%Y%m%d%H%M%S%f')
    os.makedirs(dir_name, exist_ok=True)
    logger = setup_logger(dir_name)

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
            x_scaled = x / scale  # Convert back to original coordinate system
            y_scaled = y / scale
            patch_name = f"{dataset}_{x_scaled}_{y_scaled}.png"
        else:
            patch_name = f"{dataset}_{x}_{y}.png"

        step = get_logging_step(len(coordinates))
        if i % step == 0:
            logger.info(f"patch_name: {i}, {patch_name}")
        patch.save(os.path.join(output_img, patch_name))

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Image segmentation time: {execution_time:.2f} seconds")


    # Add HIPT to path
    sys.path.append("./FineST")
    from HIPT.HIPT_4K import vision_transformer as vits

    ######################################################################
    # HIPT-vit_256(): 
    # from size '3 x patch_size x patch_size' to size '1 x 384' (16*16)
    ######################################################################
    # Note: tissue_position["pxl_row_in_fullres"].max(), tissue_position["pxl_col_in_fullres"].max()
    # should be consistent with {image_width, image_height}
    # Please check it !!!

    # Add HIPT to path
    sys.path.append("./FineST")
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
    if method == 'HIPT':
        logger.info(f"Method: {method}")
        weight_path = "https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth"
        model = get_vit256(pretrained_weights=weight_path, device=device)

        # https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/hipt_model_utils.py#L111
        def eval_transforms():
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            return transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=mean, std=std)
            ])
        
    elif method == 'Virchow2':
        logger.info(f"Method: {method}")
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
    else:
        raise ValueError(f"Unsupported method: {method}")

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

    start_time = time.time()
    for i, patch in enumerate(patches_list):

        patch_base_name, extension = os.path.splitext(patch)
        patch_path = os.path.join(output_img, patch)
        patch_image = Image.open(patch_path)

        if method == 'Virchow2':
            p_image = eval_transforms()(patch_image).unsqueeze(dim=0).to(device)    # torch.Size([1, 3, 64, 64])
            lay = model(p_image)  # size: 1 x 261 x 1280
            # tokens 1-4 are register tokens so we ignore those
            subtensors = lay[:, 5:]  # size: 1 x 256 x 1280
            subtensors_list = torch.split(subtensors, 1, dim=1)
   
        elif method == 'HIPT':
            p_image = eval_transforms()(patch_image).unsqueeze(dim=0).to(device)    # torch.Size([1, 3, 64, 64])
            lay = model.get_intermediate_layers(p_image, 1)[0]  # torch.Size([1, 17, 384])
            subtensors = lay[:, :, :]  # torch.Size([1, 17, 384])
            subtensors_list = torch.split(subtensors, 1, dim=1)
            subtensors_list = subtensors_list[1:]

        # Save image embeddings
        saved_name = patch_base_name + '.pth'
        
        step = get_logging_step(len(patches_list))
        if i % step == 0:
            logger.info(f"saved_name: {i}, {saved_name}")
        saved_path = os.path.join(output_pth, saved_name)
        torch.save(subtensors_list, saved_path)

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Feature extraction time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--position_path', required=True, help='Position file name')
    parser.add_argument('--rawimage_path', required=True, help='Image file name')
    parser.add_argument('--scale_image', required=True, help='Image file name')
    parser.add_argument('--method', required=True, help='Image feature extract method')
    parser.add_argument('--patch_size', required=True, help='Patch size for image segmentation')
    parser.add_argument('--output_img', required=True, help='Output image path')
    parser.add_argument('--output_pth', required=True, help='Output path')
    parser.add_argument('--logging', required=True, help='Logging folder path')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE, 
                       help=f'Image scale factor (default: {DEFAULT_SCALE})')
    args = parser.parse_args()

    # Convert string to boolean for scale_image
    scale_image_bool = args.scale_image.lower() in ('true', '1', 'yes', 'on')
    patch_size_int = int(args.patch_size)

    main(
        args.dataset, 
        args.position_path, 
        args.rawimage_path, 
        scale_image_bool, 
        args.method, 
        patch_size_int, 
        args.output_img, 
        args.output_pth, 
        args.logging,
        args.scale
    )
    

## Python script examples with `Virchow2` model

##########################
# NPC: sub-spot 
##########################
# python ./demo/Image_feature_extraction.py \
#    --dataset NPC \
#    --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv \
#    --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
#    --scale_image False \
#    --method Virchow2 \
#    --patch_size 112 \
#    --output_img FineST_tutorial_data/ImgEmbeddings/pth_112_14_image \
#    --output_pth FineST_tutorial_data/ImgEmbeddings/pth_112_14 \
#    --logging FineST_tutorial_data/ImgEmbeddings/Logging/ \
#    --scale 0.5  # Optional, default is 0.5


##########################
# CRC 16um: 
##########################
# python ./demo/Image_feature_extraction.py \
#     --dataset HD_CRC_16um \
#     --position_path ./Dataset/CRC16um/square_016um/tissue_positions.parquet \
#     --rawimage_path ./Dataset/CRC16um/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
#     --scale_image True \
#     --method Virchow2 \
#     --patch_size 28 \
#     --output_img ./Dataset/CRC16um/HIPT/HD_CRC_16um_pth_28_14_image_test \
#     --output_pth ./Dataset/CRC16um/HIPT/HD_CRC_16um_pth_28_14_test \
#     --logging ./Logging/HIPT_HD_CRC_16um/ \
#     --scale 0.5  # Optional, default is 0.5


##########################
# NPC: single nuclei
##########################
# python ./demo/Image_feature_extraction.py \
#    --dataset AH_Patient1 \
#    --position_path ./Dataset/NPC/StarDist/DataOutput/NPC1_allspot_p075_test/_position_all_tissue_sc.csv \
#    --rawimage_path ./Dataset/NPC/patient1/20210809-C-AH4199551.tif \
#    --scale_image False \
#    --method Virchow2 \
#    --patch_size 14 \
#    --output_img ./Dataset/NPC/HIPT/sc_Patient1_pth_14_14_image \
#    --output_pth ./Dataset/NPC/HIPT/sc_Patient1_pth_14_14 \
#    --logging ./Logging/HIPT_AH_Patient1/ \
#    --scale 0.5  # Optional, default is 0.5



## Python script examples with `HIPT` model

##########################
# NPC: sub-spot 
##########################
# python ./demo/Image_feature_extraction.py \
#    --dataset NPC \
#    --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv \
#    --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
#    --scale_image False \
#    --method HIPT \
#    --patch_size 64 \
#    --output_img FineST_tutorial_data/ImgEmbeddings/pth_64_16_image \
#    --output_pth FineST_tutorial_data/ImgEmbeddings/pth_64_16 \
#    --logging FineST_tutorial_data/ImgEmbeddings/Logging/ \
#    --scale 0.5  # Optional, default is 0.5