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


import os
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
# from FineST.HIPT import *
# from ..HIPT import *

#################################################################
## For Virchow2 
import timm
import torch
from timm.layers import SwiGLUPacked
from PIL import Image
## if use transforms
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform    
#################################################################

# Set logging
logging.getLogger().setLevel(logging.INFO)
def setup_logger(model_save_folder):
        
    level = logging.INFO

    log_name = 'HIPT_image_feature_extract.log'
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(model_save_folder + log_name)
    logger.setLevel(level)
    
    fileHandler = logging.FileHandler(os.path.join(model_save_folder, log_name), mode = 'a')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    return logger

## rescalse image to decrease split_num
def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img

## get integer nearest 'multiple of 14' to 'spot diameter'
# def get_patch_size(diameter, tile_size=14):
#     return int((diameter // tile_size) * tile_size)

def main(dataset, position_path, rawimage_path, scale_image, method, patch_size, 
         output_img, output_pth, logging):

    ## Create the folder with a unique timestamp only once
    dir_name = logging + datetime.now().strftime('%Y%m%d%H%M%S%f')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logger = setup_logger(dir_name)

    ## Set seed
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    setup_seed(666)

    ## set GPU or CPU
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    ## set scale
    scale = 0.5

    ## Load csv file with "no head"
    _, ext = os.path.splitext(position_path)
    if ext == ".csv":
        tissue_position = pd.read_csv(position_path)
        print(tissue_position)
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
                ## For sing-nuclei file, dont need rename the colnums
                tissue_position = pd.read_csv(position_path).set_index("Unnamed: 0")
                # tissue_position = tissue_position.rename(
                #     columns={
                #         'pxl_row_in_fullres': 'pxl_col_in_fullres', 
                #         'pxl_col_in_fullres': 'pxl_row_in_fullres'
                #     }
                # )
        elif tissue_position.shape[1] == 5:
            ## For between spot or single nuclei
            tissue_position = pd.read_csv(position_path).set_index("Unnamed: 0")
            tissue_position.columns = ['array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
            tissue_position = tissue_position.rename(
                columns={
                    'pxl_row_in_fullres': 'pxl_col_in_fullres', 
                    'pxl_col_in_fullres': 'pxl_row_in_fullres'
                }
            )

    elif ext == ".parquet":
        tissue_position = (pd.read_parquet(position_path)
                        .set_index('barcode')
                        .rename(columns={'pxl_row_in_fullres': 'pxl_col_in_fullres', 'pxl_col_in_fullres': 'pxl_row_in_fullres'})
                        .query('in_tissue == 1'))
        if str(scale_image) == 'True':
            tissue_position['pxl_col_in_fullres'] = tissue_position['pxl_col_in_fullres']*scale
            tissue_position['pxl_row_in_fullres'] = tissue_position['pxl_row_in_fullres']*scale
    else:
        logger.info(f"Unsupported file type: {ext}")
    logger.info(f'tissue_position: \n {tissue_position.head()}')

    ##############################################
    # different, need math with figure 
    ##############################################
    coordinates = list(zip(tissue_position["pxl_row_in_fullres"], tissue_position["pxl_col_in_fullres"]))
    logger.info(f'tissue_position number: {len(coordinates)}')
    logger.info(
        f'tissue_position range: '
        f'{tissue_position["pxl_row_in_fullres"].max()} '
        f'{tissue_position["pxl_col_in_fullres"].max()}'
    )
    # https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/hipt_4k.py
    import sys
    sys.path.append("./FineST")
    from HIPT.HIPT_4K import vision_transformer as vits
    ###################################################################################################
    # Note: tissue_position["pxl_row_in_fullres"].max(), tissue_position["pxl_col_in_fullres"].max()
    # should consistant with {image_width, image_height}
    # Please check it !!!
    ###################################################################################################

    ## Load image
    if str(scale_image) == 'True':
        image_obj = Image.open(rawimage_path)
        image = np.array(image_obj)

        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]  # remove alpha channel
        image = image.astype(np.float32)
        logger.info(f'Rescaling image (scale: {scale:.3f})...')
        image = rescale_image(image, scale)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)    # NumPy to PIL
        logger.info(f'Rescaling image DONE!...')

    elif str(scale_image) == 'False':      
        image = Image.open(rawimage_path)
        # image_width, image_height = image.size

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
        
        if str(scale_image) == 'True':
            x_scaled = x/scale
            y_scaled = y/scale 
            patch_name = f"{dataset}_{x_scaled}_{y_scaled}.png"
        else: 
            patch_name = f"{dataset}_{x}_{y}.png"

        if len(coordinates) > 50000:
            step = 1000
        else:
            step = 100

        if i % step == 0:
            # print(f"patch_name: {i}, {patch_name}")
            logger.info(f"patch_name: {i}, {patch_name}")
        patch.save(os.path.join(output_img, patch_name))

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Image segmentation time: {execution_time:.2f} seconds")


    ######################################################################
    # HIPT-vit_256(): 
    # from size '3 x patch_size x patch_size' to size '1 x 384' (16*16)
    ######################################################################
    # https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/hipt_model_utils.py#L39
    def get_vit256(pretrained_weights, arch='vit_small', device=torch.device('cuda:0')):
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
        # device = torch.device("cpu")
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
        for p in model256.parameters():
            p.requires_grad = False
        model256.eval()
        model256.to(device)

        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                # print(f"Take key {checkpoint_key} in provided checkpoint dict")
                logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model256.load_state_dict(state_dict, strict=False)
            # print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
            logger.info(f'Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}')
            
        return model256

    # Load model
    if str(method) == 'HIPT':
        print("Method: ", str(method))    
        weight_path = "https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth"
        model = get_vit256(pretrained_weights = weight_path)

        # https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/hipt_model_utils.py#L111
        def eval_transforms():
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            eval_t = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean = mean, std = std)])
            return eval_t
        
    elif str(method) == 'Virchow2':  
        print("Method: ", str(method))    
        ######################################################################
        # Virchow2(): 
        # from size '3 x patch_size x patch_size' to size '1 x 1280' (14*14)
        ######################################################################
        # Virchow2, need to specify MLP layer and activation function for proper init
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, 
                                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.to(device)
        model = model.eval()

        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/constants.py#L3
        def eval_transforms_Virchow2():
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            eval_t = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean = mean, std = std)])
            return eval_t

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

        if str(method) == 'Virchow2':
            # p_image = transforms(patch_image).unsqueeze(0).to(device)                   # size: 1 x 3 x 224 x 224
            p_image = eval_transforms_Virchow2()(patch_image).unsqueeze(dim=0).to(device) # torch.Size([1, 3, 64, 64])
            lay = model(p_image)       # size: 1 x 261 x 1280
            subtensors = lay[:, 5:]    # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
            subtensors_list = torch.split(subtensors, 1, dim=1)
   
        elif str(method) == 'HIPT':      
            p_image = eval_transforms()(patch_image).unsqueeze(dim=0).to(device)    # torch.Size([1, 3, 64, 64])
            lay = model.get_intermediate_layers(p_image, 1)[0]                      # torch.Size([1, 17, 384])
            subtensors = lay[:, :, :]                                               # torch.Size([1, 17, 384])
            subtensors_list = torch.split(subtensors, 1, dim=1)
            subtensors_list = subtensors_list[1:]

        ## save image embeddings
        saved_name = patch_base_name + '.pth'
        
        ## print imformation
        if len(patches_list) > 50000:
            step = 1000
        else:
            step = 100

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
    args = parser.parse_args()

    main(args.dataset, args.position_path, args.rawimage_path, args.scale_image, args.method, args.patch_size, 
         args.output_img, args.output_pth, args.logging)
    

## Python script

##########################
# NPC: sub-spot
##########################
# python ./demo/Image_feature_extraction.py \
#    --dataset NPC \
#    --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv  \
#    --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
#    --scale_image False \
#    --method Virchow2 \
#    --patch_size 112 \
#    --output_img FineST_tutorial_data/ImgEmbeddings/pth_112_14_image \
#    --output_pth FineST_tutorial_data/ImgEmbeddings/pth_112_14 \
#    --logging FineST_tutorial_data/ImgEmbeddings/


##########################
# CRC 16um: 
##########################
# time python ./FineST/HIPT_image_feature_extract_virchow2.py \
#     --dataset HD_CRC_16um \
#     --position_path ./Dataset/CRC16um/square_016um/tissue_positions.parquet \
#     --rawimage_path ./Dataset/CRC16um/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
#     --scale_image True \
#     --method Virchow2 \
#     --output_img ./Dataset/CRC16um/HIPT/HD_CRC_16um_pth_28_14_image_test \
#     --output_pth ./Dataset/CRC16um/HIPT/HD_CRC_16um_pth_28_14_test \
#     --patch_size 28 \
#     --logging ./Logging/HIPT_HD_CRC_16um/


##########################
# NPC: single nuclei
##########################
# cd /mnt/lingyu/nfs_share2/Python/FineST/FineST_local/
# time python ./FineST/HIPT_image_feature_extract_virchow2.py \
#    --dataset AH_Patient1 \
#    --position_path ./Dataset/NPC/StarDist/DataOutput/NPC1_allspot_p075_test/_position_all_tissue_sc.csv \
#    --rawimage_path ./Dataset/NPC/patient1/20210809-C-AH4199551.tif \
#    --scale_image False \
#    --method Virchow2 \
#    --output_img ./Dataset/NPC/HIPT/sc_Patient1_pth_14_14_image \
#    --output_pth ./Dataset/NPC/HIPT/sc_Patient1_pth_14_14 \
#    --patch_size 14 \
#    --logging ./Logging/HIPT_AH_Patient1/