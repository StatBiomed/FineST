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

import os
import torch
from PIL import Image 
import numpy as np
import pandas as pd
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms
from datetime import datetime
import random
import time
import argparse
import logging


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


def main(dataset, position, image, output_path_img, output_path_pth, patch_size, logging_folder):

    # Create the folder with a unique timestamp only once
    dir_name = logging_folder + datetime.now().strftime('%Y%m%d%H%M%S%f')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logger = setup_logger(dir_name)

    # Set seed
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    setup_seed(666)

    # set GPU or CPU
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load csv file with "no head"
    _, ext = os.path.splitext(position)
    if ext == ".csv":
        tissue_position = pd.read_csv(position)
        print(tissue_position)
        # print(tissue_position.shape[1])
        if tissue_position.shape[1] == 6:
            if 'cell_nums' not in tissue_position.columns.tolist():
                ## For within spot
                tissue_position = pd.read_csv(position, header=None).set_index(0)
                tissue_position.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
                tissue_position = tissue_position.rename(columns={'pxl_row_in_fullres': 'pxl_col_in_fullres', 'pxl_col_in_fullres': 'pxl_row_in_fullres'})
                tissue_position = tissue_position[tissue_position['in_tissue'] == 1]
            else:
                ## For within spot
                tissue_position = pd.read_csv(position).set_index("Unnamed: 0")
                tissue_position = tissue_position.rename(columns={'pxl_row_in_fullres': 'pxl_col_in_fullres', 'pxl_col_in_fullres': 'pxl_row_in_fullres'})
        elif tissue_position.shape[1] == 5:
            ## For between spot or single nuclei
            tissue_position = pd.read_csv(position).set_index("Unnamed: 0")
            tissue_position.columns = ['array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
            tissue_position = tissue_position.rename(columns={'pxl_row_in_fullres': 'pxl_col_in_fullres', 'pxl_col_in_fullres': 'pxl_row_in_fullres'})

    elif ext == ".parquet":
        tissue_position = (pd.read_parquet(position)
                        .set_index('barcode')
                        .rename(columns={'pxl_row_in_fullres': 'pxl_col_in_fullres', 'pxl_col_in_fullres': 'pxl_row_in_fullres'})
                        .query('in_tissue == 1'))
    else:
        # print(f"Unsupported file type: {ext}")
        logger.info(f"Unsupported file type: {ext}")
    # print('tissue_position: \n', tissue_position.head())
    logger.info(f'tissue_position: \n {tissue_position.head()}')

    ##############################################
    # different, need math with figure 
    ##############################################
    coordinates = list(zip(tissue_position["pxl_row_in_fullres"], tissue_position["pxl_col_in_fullres"]))
    # print('tissue_position range: \n', tissue_position['pxl_row_in_fullres'].max(), tissue_position['pxl_col_in_fullres'].max())
    logger.info(f'tissue_position range: \n {tissue_position["pxl_row_in_fullres"].max()} {tissue_position["pxl_col_in_fullres"].max()}')

    # https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/hipt_4k.py
    from HIPT.HIPT_4K import vision_transformer as vits

    # Load image
    image = Image.open(image)
    image_width, image_height = image.size
    # print("image_width, image_height: ", image_width, image_height)
    logger.info(f"image_width, image_height: {image_width}, {image_height}")

    # Create patches
    # patch_size = 32 for Visium HD, patch_size = 64 for Visium (V2)
    patch_size = int(patch_size)
    os.makedirs(output_path_img, exist_ok=True)

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
        patch_name = f"{dataset}_{x}_{y}.png"

        if len(coordinates) > 50000:
            step = 5000
        else:
            step = 500

        if i % step == 0:
            # print(f"patch_name: {i}, {patch_name}")
            logger.info(f"patch_name: {i}, {patch_name}")
        patch.save(os.path.join(output_path_img, patch_name))

    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"The image segment execution time for the loop is: {execution_time} seconds")
    logger.info(f"The image segment execution time for the loop is: {execution_time} seconds")

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
    weight_path = "https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth"
    model = get_vit256(pretrained_weights = weight_path)

    # https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/hipt_model_utils.py#L111
    def eval_transforms():
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
        return eval_t

    # Process patches
    os.makedirs(output_path_pth, exist_ok=True)
    patches_list = os.listdir(output_path_img)

    start_time = time.time()
    for i, patch in enumerate(patches_list):
        patch_base_name, extension = os.path.splitext(patch)
        patch_path = os.path.join(output_path_img, patch)
        patch_image = Image.open(patch_path)
        p_image = eval_transforms()(patch_image).unsqueeze(dim=0).to(device)
        lay = model.get_intermediate_layers(p_image, 1)[0]
        subtensors = lay[:, :, :]
        subtensors_list = torch.split(subtensors, 1, dim=1)
        subtensors_list = subtensors_list[1:]
        saved_name = patch_base_name + '.pth'

        if len(patches_list) > 50000:
            step = 5000
        else:
            step = 500

        if i % step == 0:
            # print(f"saved_name: {i}, {saved_name}")
            logger.info(f"saved_name: {i}, {saved_name}")
        saved_path = os.path.join(output_path_pth, saved_name)
        torch.save(subtensors_list, saved_path)

    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"The image feature extract  time for the loop is: {execution_time} seconds")
    logger.info(f"The image feature extract time for the loop is: {execution_time} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--position', required=True, help='Position file name')
    parser.add_argument('--image', required=True, help='Image file name')
    parser.add_argument('--output_path_img', required=True, help='Output image path')
    parser.add_argument('--output_path_pth', required=True, help='Output path')
    parser.add_argument('--patch_size', required=True, help='Patch size for image segmentation')
    parser.add_argument('--logging_folder', required=True, help='Logging folder path')
    args = parser.parse_args()

    main(args.dataset, args.position, args.image, args.output_path_img, args.output_path_pth, args.patch_size, args.logging_folder)