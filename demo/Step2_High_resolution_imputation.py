import os
import time
import warnings
import numpy as np
import torch
import logging
from datetime import datetime
import json
import argparse
import pandas as pd
print("torch version: %s" % torch.__version__)

from FineST.utils import *
from FineST import datasets
from FineST.processData import *
from FineST.model import *
from FineST.train import *
from FineST.plottings import *
from FineST.inference import *

##################
# Basic setting
##################
warnings.filterwarnings('ignore')
setup_seed(666)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    return True

def load_and_process_data(args):

    adata = datasets.NPC()
    print(" **** Load the original NPC patient1 adata: **** \n", adata)
    adata = adata_LR(adata, args.LRgene_path)
    adata = adata_preprocess(adata)
    gene_hv = np.array(adata.var_names)
    matrix = adata2matrix(adata, gene_hv)
    file_paths = sorted(os.listdir(os.path.join(args.system_path, args.imag_within_path)))
    # Map dataset_class to ST_class for image coordinate parsing
    if args.dataset_class in ['Visium16', 'Visium64']:
        ST_class = 'Visium'
    elif args.dataset_class == 'VisiumHD':
        ST_class = 'VisiumHD'
    elif args.dataset_class == 'VisiumSC':
        ST_class = 'VisiumSC'
    else:
        ST_class = 'Visium'
    
    position_image = get_image_coord(file_paths, ST_class=ST_class)
    position = pd.read_csv(os.path.join(args.system_path, args.visium_path), header=None)
    position = position.rename(columns={position.columns[-2]: 'pixel_x', position.columns[-1]: 'pixel_y'})
    position_image = image_coord_merge(position_image, position, ST_class=ST_class)
    spotID_order = np.array(position_image[0])
    matrix_order, matrix_order_df = sort_matrix(matrix, position_image, spotID_order, gene_hv)
    adata = update_adata_coord(adata, matrix_order, position_image)
    gene_expr(adata, matrix_order_df, gene_selet=args.gene_selected, 
              save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_orig_gene_expr.pdf'))

    if args.dataset_class == 'Visium':

        ################################
        # For all spot image embeddings
        ################################
        file_paths_spot = os.listdir(os.path.join(args.system_path, args.imag_within_path))
        print(" **** Within_spot number: ", len(file_paths_spot))
        file_paths_between_spot = os.listdir(os.path.join(args.system_path, args.imag_betwen_path))
        print(" **** Between_spot number:", len(file_paths_between_spot))
        file_paths_all = file_paths_spot + file_paths_between_spot

        ## Merge, sort and process file paths
        # get_image_coord_all doesn't need ST_class parameter
        data_all = get_image_coord_all(file_paths_all)
        position_order_allspot = pd.DataFrame(data_all, columns=['pixel_y', 'pixel_x'])
        print(" **** The coords of image patch: **** \n", position_order_allspot.shape)
        print(position_order_allspot.head())
        position_order_allspot.to_csv(os.path.join(args.system_path, args.spatial_pos_path), index=False, header=False)
        file_paths = file_paths_all

    elif args.dataset_class == 'VisiumSC':

        ####################################
        # For all spot-sc image embeddings
        ####################################
        file_paths_sc = os.listdir(os.path.join(args.system_path, args.image_embed_path_sc))
        print(" **** Single-nuclei number: ", len(file_paths_sc))
        # get_image_coord_all doesn't need ST_class parameter
        data_all_sc = get_image_coord_all(file_paths_sc)  
        spatial_loc_sc = pd.DataFrame(data_all_sc, columns=['pixel_y', 'pixel_x'])
        spatial_loc_sc = spatial_loc_sc[['pixel_x', 'pixel_y']]

        print(" **** The coords of single-nuclei image patch: **** \n", spatial_loc_sc.shape)
        print(spatial_loc_sc.head())
        spatial_loc_sc.to_csv(os.path.join(args.system_path, args.spatial_pos_path_sc), index=False, header=False)
        file_paths = file_paths_sc
        len(file_paths)

    else:
        raise ValueError('Invalid dataset_class. Only "Visium", "VisiumSC" and "VisiumHD" are supported.')     

    return adata, gene_hv, file_paths
              
def setup_logging(args):
    logging.getLogger().setLevel(logging.INFO)

    weight_path = os.path.join(args.system_path, args.weight_path)
    dir_name = weight_path + datetime.now().strftime('%Y%m%d%H%M%S%f')

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logger = setup_logger(dir_name)
    print("dir_name: \n", dir_name)

    parame_path = os.path.join(args.system_path, args.parame_path)
    with open(parame_path, "r") as json_file:
        params = json.load(json_file)
    logger.info("Load parameters:\n" + json.dumps(params, indent=2))

    return logger, parame_path, params, dir_name

def infer_gene_expr(model, file_paths, args, gene_hv, logger):
    model.to(device)   

    if args.dataset_class == 'Visium': 
        all_dataset = build_loaders_inference_allimage(batch_size=len(file_paths), 
                                                    file_paths_spot=os.path.join(args.system_path,
                                                                                    args.imag_within_path,'*.pth'),
                                                    file_paths_between_spot=os.path.join(args.system_path,
                                                                                            args.imag_betwen_path,'*.pth'), 
                                                    spatial_pos_path=args.spatial_pos_path, 
                                                    dataset_class=args.dataset_class)
        logger.info("Running inference task between spot...")

        start_infer_time = time.time()
        (recon_ref_adata_image_f2, reconstructed_matrix_reshaped,
        _, _, input_coord_all) = perform_inference_image_between_spot(model, all_dataset, dataset_class=args.dataset_class)
        print("--- %s seconds for inference within&between spots ---" % (time.time() - start_infer_time))
        print(" **** Reconstructed_matrix_reshaped shape: ", reconstructed_matrix_reshaped.shape)
        logger.info("Running inference task between spot DONE!")

        ## Get coords
        spatial_loc_all = get_allspot_coors(input_coord_all)
        print(" **** The spatial coords of all spots: \n", spatial_loc_all)

        ## Plot 
        gene_expr_allspots(args.gene_selected, spatial_loc_all, recon_ref_adata_image_f2, gene_hv, 
                        'Inferred all spot', s=4, 
                        save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_all_spot_inferred.pdf'))

        ## reshape
        reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(reconstructed_matrix_reshaped, 
                                                                    dataset_class=args.dataset_class)
        print(" **** The size of all reconstructed tensor data:", reconstructed_matrix_reshaped_tensor.shape)

        (_, _, all_spot_all_variable, 
        C2_all, adata_infer_all) =  subspot_coord_expr_adata(reconstructed_matrix_reshaped_tensor,
                                                        spatial_loc_all, gene_hv,
                                                        dataset_class=args.dataset_class)
        print(" **** All_spot_all_variable shape:", all_spot_all_variable.shape)
        adata_infer = adata_infer_all
        spatial_loc = spatial_loc_all
        C2 = C2_all


    elif args.dataset_class == 'VisiumSC': 
        all_dataset_sc = build_loaders_inference_allimage(batch_size=len(file_paths),
                                                        file_paths_spot=os.path.join(args.system_path,
                                                                                    args.image_embed_path_sc, '*.pth'),
                                                        spatial_pos_path=args.spatial_pos_path_sc, 
                                                        dataset_class=args.dataset_class)
        logger.info("Running inference task single-nuclei...")

        start_infer_time = time.time()
        (recon_ref_adata_image_f2, reconstructed_matrix_reshaped,
        _, _, input_coord_all) = perform_inference_image_between_spot(model, all_dataset_sc, dataset_class=args.dataset_class)
        print("--- %s seconds for inference single-nuclei spots ---" % (time.time() - start_infer_time))
        print(" **** Reconstructed_matrix_reshaped shape: ", reconstructed_matrix_reshaped.shape)
        logger.info("Running inference task between spot DONE!")

        ## Get coords
        spatial_loc_sc = get_allspot_coors(input_coord_all)
        print(" **** The spatial coords of all single-nuclei: \n", spatial_loc_sc)

        ## Plot 
        gene_expr_allspots(args.gene_selected, spatial_loc_sc, recon_ref_adata_image_f2, gene_hv, 
                        'Inferred single-cell', s=0.5, 
                        save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_single-cell_inferred.pdf'))

        ## reshape
        reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(reconstructed_matrix_reshaped, 
                                                                    dataset_class=args.dataset_class)
        print(" **** The size of all reconstructed tensor data:", reconstructed_matrix_reshaped_tensor.shape)

        (_, _, all_spot_all_variable, 
        C2_sc, adata_infer_sc) = subspot_coord_expr_adata(reconstructed_matrix_reshaped_tensor,
                                                        spatial_loc_sc, gene_hv, 
                                                        dataset_class=args.dataset_class)
        print(" **** All_spot_all_variable shape:", all_spot_all_variable.shape)

        adata_infer = adata_infer_sc
        spatial_loc = spatial_loc_sc
        C2 = C2_sc

    else:
        raise ValueError('Invalid dataset_class. Only "Visium", "VisiumSC" and "VisiumHD" are supported.')

    return adata_infer, spatial_loc, C2


def main(args):
    # Check if required files exist
    required_files = [
        args.LRgene_path, 
        os.path.join(args.system_path, args.visium_path),
        os.path.join(args.system_path, args.imag_within_path),
        os.path.join(args.system_path, args.parame_path)
    ]
    
    for file_path in required_files:
        if not check_file_exists(file_path):
            return

    # Load and process data
    adata, gene_hv, file_paths = load_and_process_data(args)

    # Setup logging
    logger, parame_path, params, dir_name = setup_logging(args)

    # Load the trained model
    dir_name = args.weight_save_path
    model = load_model(dir_name, parame_path, params, gene_hv)

    # Perform inference
    adata_infer, spatial_loc, C2 = infer_gene_expr(model, file_paths, args, gene_hv, logger)

    if args.dataset_class == 'Visium': 
        ########################################
        # Impute super-resolved gene expr.
        ########################################
        adata_impt = impute_adata(adata, adata_infer, C2, gene_hv, k=6)
        adata_impt, data_impt_all = weight_adata(adata_infer, adata_impt, gene_hv, w=0.5)
        adata_impt.write_h5ad(os.path.join(args.system_path, args.adata_all_supr_path))   

        _, adata_impt_reshape = reshape_latent_image(data_impt_all, dataset_class=args.dataset_class)
        print("data_impt_all shape:", adata_impt.shape)
        print("adata_impt_reshape shape:", adata_impt_reshape.shape)

        ########################################
        # Integrate to spot-resolved gene expr.
        ########################################
        adata_impt_spot = sc.AnnData(X = pd.DataFrame(adata_impt_reshape.cpu().detach().numpy()))
        adata_impt_spot.var_names = gene_hv
        adata_impt_spot.obs['x'] = spatial_loc[:,0]
        adata_impt_spot.obs['y'] = spatial_loc[:,1]
        print("adata_impt_spot: \n", adata_impt_spot)
        print(adata_impt_spot.obs['x'])
        adata_impt_spot.write_h5ad(os.path.join(args.system_path, args.adata_all_spot_path))  

        ########################################
        # Visualize predicted gene expr.
        ########################################
        gene_expr_allspots(args.gene_selected, spatial_loc, adata_impt_reshape, gene_hv, 
                            'FineST all spot', s=8, 
                            save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_all_spot.pdf'))
        logger.info("Running low-resolution all-spot plot DONE!")

        gene_expr_allspots(args.gene_selected, C2, adata_impt.X, gene_hv, 
                            'FineST all sub-spot', s=0.5, 
                            save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_all_sub-spot.pdf'))
        logger.info("Running high-resolution all-sub-spot plot DONE!")

    elif args.dataset_class == 'VisiumSC': 

        ########################################
        # Impute super-resolved gene expr.
        ########################################
        adata_impt = impute_adata(adata, adata_infer, C2, gene_hv, k=6)
        adata_impt, data_impt_all = weight_adata(adata_infer, adata_impt, gene_hv, w=0.5)
        adata_impt.write_h5ad(os.path.join(args.system_path, args.adata_super_path_sc))   

        _, adata_impt_reshape = reshape_latent_image(data_impt_all, dataset_class=args.dataset_class)
        print("data_impt_sc shape:", adata_impt.shape)
        print("adata_impt_reshape shape:", adata_impt_reshape.shape)

        ########################################
        # Visualize predicted gene expr.
        ########################################
        gene_expr_allspots(args.gene_selected, spatial_loc, adata_impt_reshape, gene_hv, 
                            'FineST single-cell', s=8, 
                            save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_all_sc.pdf'))
        logger.info("Running low-resolution all-spot plot DONE!")

    else:
        raise ValueError('Invalid dataset_class. Only "Visium", "VisiumSC" and "VisiumHD" are supported.')   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FineST Model Training and Inference")

    parser.add_argument('--system_path', type=str, required=True, help='System path for data and weights')
    parser.add_argument('--LRgene_path', type=str, required=True, help='Path to LR genes')
    parser.add_argument('--dataset_class', type=str, required=True, help='Visium or VisiumHD')
    parser.add_argument('--gene_selected', type=str, required=True, help='Marker gene visualization')
    parser.add_argument('--weight_path', type=str, default='weights', help='Directory to save weights')
    parser.add_argument('--parame_path', type=str, required=True, help='Path to parameter file')
    parser.add_argument('--visium_path', type=str, required=True, help='Path to Visium data')

    ## sub-spot from geometric segmentation: 
    parser.add_argument('--imag_within_path', type=str, required=False, help='Path to within-spot image embeddings')
    parser.add_argument('--imag_betwen_path', type=str, required=False, help='Path to between-spot image embeddings')
    parser.add_argument('--spatial_pos_path', type=str, default='spatial_pos.csv', help='Path to save spatial positions')
    parser.add_argument('--adata_all_supr_path', type=str, required=False, help='Path to predicted super adata')
    parser.add_argument('--adata_all_spot_path', type=str, required=False, help='Path to predicted spot adata')

    ## single-nuclei from nuclei segmentation: 
    parser.add_argument('--image_embed_path_sc', type=str, required=False, help='Path to single-nuclei image embeddings')
    parser.add_argument('--spatial_pos_path_sc', type=str, required=False, help='Path to save sc spatial positions')
    parser.add_argument('--adata_super_path_sc', type=str, required=False, help='Path to predicted super adata for sc')

    parser.add_argument('--figure_save_path', type=str, default='figures', help='Directory to save figures')
    parser.add_argument('--weight_save_path', type=str, default=None, help='Path to pre-trained weights, if available')

    args = parser.parse_args()

    main(args)


## Python Script:

#############################
# For geometric segmentation
#############################
# python ./FineST/FineST/demo/High_resolution_imputation.py \
#     --system_path '/mnt/lingyu/nfs_share2/Python/' \
#     --weight_path 'FineST/FineST_local/Finetune/' \
#     --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
#     --dataset_class 'Visium' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
#     --imag_within_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/' \
#     --imag_betwen_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/NEW_AH_Patient1_pth_64_16/' \
#     --spatial_pos_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order_all.csv' \
#     --weight_save_path 'FineST/FineST_local/Finetune/20240125140443830148' \
#     --figure_save_path 'FineST/FineST_local/Dataset/NPC/Figures/' \
#     --adata_all_supr_path 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all.h5ad' \
#     --adata_all_spot_path 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all_spot.h5ad' 

#############################
# For Nuclei segmentation
#############################
# python ./FineST/FineST/demo/High_resolution_imputation.py \
#     --system_path '/mnt/lingyu/nfs_share2/Python/' \
#     --weight_path 'FineST/FineST_local/Finetune/' \
#     --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
#     --dataset_class 'VisiumSC' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
#     --imag_within_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/' \
#     --image_embed_path_sc 'NPC/Data/stdata/ZhuoLiang/LLYtest/sc_Patient1_pth_16_16/' \
#     --spatial_pos_path_sc 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order_sc.csv' \
#     --adata_super_path_sc 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all_sc.h5ad' \
#     --weight_save_path 'FineST/FineST_local/Finetune/20240125140443830148' \
#     --figure_save_path 'FineST/FineST_local/Dataset/NPC/Figures/'
