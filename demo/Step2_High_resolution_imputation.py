import os
import sys
import time
import warnings
import numpy as np
import torch
import logging
from datetime import datetime
import json
import argparse
import pandas as pd
import scanpy as sc

# Custom class to tee output to both console and file
class TeeOutput:
    """Class to write output to both console and file simultaneously."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()

print("torch version: %s" % torch.__version__)

# Add local FineST package to Python path (before importing FineST)
# This ensures we use the local development version instead of any installed package
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from demo/ to FineST/ directory (where FineST package is located)
fineST_root = os.path.dirname(script_dir)
# Add FineST/ to sys.path so we can import FineST package
if fineST_root not in sys.path:
    sys.path.insert(0, fineST_root)

# Now import FineST (will use local version)
import FineST as fst
from FineST.datasets import dataset
import FineST.plottings as fstplt
print("FineST version: %s" % fst.__version__)
print("Using FineST from: %s" % os.path.dirname(fst.__file__))

from FineST.utils import *
from FineST import datasets
from FineST.processData import *
from FineST.model import *
from FineST.plottings import *
from FineST.inference import *

##################
# Basic setting
##################
warnings.filterwarnings('ignore')
setup_seed(666)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################
# Basic functions
##################
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    return True

def ensure_dir_exists(file_path):
    """Ensure the directory for a file path exists."""
    dir_path = os.path.dirname(file_path)
    if dir_path: 
        os.makedirs(dir_path, exist_ok=True)

def get_figure_save_path(args):
    """Get and create figure save directory."""
    if os.path.isabs(args.figure_save_path):
        figure_dir = args.figure_save_path
    else:
        figure_dir = os.path.join(args.system_path, args.figure_save_path)
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir

def setup_log_file(args):
    """
    Setup log file to save all terminal output.
    Log file will be saved in the same directory as figures (figure_save_path).
    File name format: Results + timestamp.
    """
    # Get figure directory (same as where figures are saved)
    figure_dir = get_figure_save_path(args)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    log_file_path = os.path.join(figure_dir, f'Results{timestamp}.log')
    
    # Create TeeOutput to write to both console and file
    tee = TeeOutput(log_file_path)
    
    # Redirect stdout and stderr
    sys.stdout = tee
    sys.stderr = tee
    
    print(f"Log file saved to: {log_file_path}")
    print("=" * 80)
    
    return tee, log_file_path, timestamp

def setup_logging(args, timestamp, figure_dir):
    """
    Setup logging for inference.
    """
    logging.getLogger().setLevel(logging.INFO)

    # Create logger directory in figure_dir with shared timestamp
    dir_name = os.path.join(figure_dir, f'weights{timestamp}')

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logger = setup_logger(dir_name)
    print("dir_name: \n", dir_name)

    parame_path = os.path.join(args.system_path, args.parame_path)
    with open(parame_path, "r") as json_file:
        params = json.load(json_file)
    logger.info("Load parameters:\n" + json.dumps(params, indent=2))

    return logger, parame_path, params, dir_name

def load_and_process_data(args):
    """
    Load and process spatial transcriptomics data.
    This function loads NPC dataset, filters to LR genes, and processes data.
    For Visium: prepares within-spot and between-spot image embeddings.
    For VisiumSC: prepares single-nuclei image embeddings.
    """
    adata = datasets.NPC()
    print(" **** Load the original NPC patient1 adata: **** \n", adata)
    
    # Use LRgene_path as gene_list parameter (can be file path or 'LR_genes', 'HV_genes', 'LR_HV_genes')
    lr_gene_path = os.path.join(args.system_path, args.LRgene_path)
    adata = adata_LR(adata, gene_list=lr_gene_path)
    adata = adata_preprocess(adata, normalize=False)
    print(" **** Processed NPC patient1 adata: **** \n", adata)
    gene_hv = np.array(adata.var_names)
    print(" **** The length of LR genes: ", len(gene_hv))

    matrix = adata2matrix(adata, gene_hv)
    
    # For VisiumSC, skip within-spot processing and go directly to single-nuclei processing
    if args.dataset_class == 'VisiumSC':
        # For VisiumSC, we don't need to process within-spots
        # Just prepare the adata for imputation (which needs the original spot-level data)
        # The single-nuclei processing will be done below
        pass
    else:
        # For Visium16/Visium64, process within-spots first
        file_paths = sorted(os.listdir(os.path.join(args.system_path, args.imag_within_path)))
        print(" **** Image embedding file (First 3): **** \n", file_paths[:3])
        
        # Map dataset_class to ST_class for image coordinate parsing
        if args.dataset_class in ['Visium16', 'Visium64']:
            ST_class = 'Visium'
        elif args.dataset_class == 'VisiumHD':
            ST_class = 'VisiumHD'
        else:
            ST_class = 'Visium'
        
        position_image = get_image_coord(file_paths, ST_class)
        position = pd.read_csv(os.path.join(args.system_path, args.visium_path), header=None)
        position = position.rename(columns={position.columns[-2]: 'pixel_x', position.columns[-1]: 'pixel_y'})
        position_image = image_coord_merge(position_image, position, ST_class)
        spotID_order = np.array(position_image[0])
        matrix_order, matrix_order_df = sort_matrix(adata, position_image, spotID_order, gene_hv)
        adata = update_adata_coord(adata, matrix_order_df, position_image)
        
        # Ensure figure directory exists
        figure_dir = get_figure_save_path(args)
        gene_expr(adata, matrix_order_df, gene_selet=args.gene_selected, 
                  save_path=os.path.join(figure_dir, str(args.gene_selected)+'_orig_gene_expr.pdf'))

    if args.dataset_class in ['Visium16', 'Visium64']:
        ################################
        # For all spot image embeddings
        ################################
        file_paths_spot = os.listdir(os.path.join(args.system_path, args.imag_within_path))
        print(" **** Within_spot number: ", len(file_paths_spot))
        file_paths_between_spot = os.listdir(os.path.join(args.system_path, args.imag_betwen_path))
        print(" **** Between_spot number:", len(file_paths_between_spot))
        file_paths_all = file_paths_spot + file_paths_between_spot
        print(" **** All_spot number:", len(file_paths_all))

        ## Merge, sort and process file paths
        # get_image_coord_all doesn't need ST_class parameter
        data_all = get_image_coord_all(file_paths_all)
        position_order_allspot = pd.DataFrame(data_all, columns=['pixel_y', 'pixel_x'])
        print(" **** The coords of image patch: **** \n", position_order_allspot.shape)
        print(position_order_allspot.head())
        
        # Ensure directory exists before saving
        spatial_pos_full_path = os.path.join(args.system_path, args.spatial_pos_path)
        ensure_dir_exists(spatial_pos_full_path)
        position_order_allspot.to_csv(spatial_pos_full_path, index=False, header=False)
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

        print(" **** The coords of single-nuclei image patch: **** \n", spatial_loc_sc.shape)
        print(spatial_loc_sc.head())
        
        # Ensure directory exists before saving
        spatial_pos_sc_full_path = os.path.join(args.system_path, args.spatial_pos_path_sc)
        ensure_dir_exists(spatial_pos_sc_full_path)
        spatial_loc_sc.to_csv(spatial_pos_sc_full_path, index=False, header=False)
        file_paths = file_paths_sc

    else:
        raise ValueError('Invalid dataset_class. Only "Visium16", "Visium64", "VisiumSC" and "VisiumHD" are supported.')     

    return adata, gene_hv, file_paths

def infer_gene_expr(model, file_paths, args, gene_hv, logger, patch_size=64):
    """
    Perform inference to predict gene expression from image features for all spots.
    
    For Visium: infers gene expression for both within-spots and between-spots.
    For VisiumSC: infers gene expression for single-nuclei.
    """
    model.to(device)   

    if args.dataset_class in ['Visium16', 'Visium64']:
        # Map to 'Visium' for processing
        dataset_class_visium = 'Visium'
        
        # Determine patch_size from dataset_class
        # Determine patch_size from dataset_class
        if args.dataset_class == 'Visium16':
            patch_size = 64  # HIPT typically uses patch_size=64
        elif args.dataset_class == 'Visium64':
            patch_size = 112  # Virchow2 typically uses patch_size=112
        
        all_dataset = build_loaders_inference_allimage(
            batch_size=len(file_paths), 
            file_paths_spot=os.path.join(args.system_path, args.imag_within_path, '*.pth'),
            file_paths_between_spot=os.path.join(args.system_path, args.imag_betwen_path, '*.pth'), 
            spatial_pos_path=os.path.join(args.system_path, args.spatial_pos_path), 
            dataset_class=args.dataset_class
        )
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
        figure_dir = get_figure_save_path(args)
        gene_expr_allspots(args.gene_selected, spatial_loc_all, recon_ref_adata_image_f2, gene_hv, 
                        'Inferred all spot', s=1.5, marker='s',
                        figsize=(5, 4),
                        save_path=os.path.join(figure_dir, str(args.gene_selected)+'_all_spot_inferred.pdf'))

        ## reshape
        reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(reconstructed_matrix_reshaped, 
                                                                    dataset_class=args.dataset_class)
        print(" **** The size of all reconstructed tensor data:", reconstructed_matrix_reshaped_tensor.shape)

        # Use patch_size parameter (from notebook: patch_size=64 for Visium16)
        # Note: subspot_coord_expr_adata needs 'Visium16' or 'Visium64', not 'Visium'
        (_, _, all_spot_all_variable, 
        C2_all, adata_infer_all) = subspot_coord_expr_adata(reconstructed_matrix_reshaped_tensor,
                                                        spatial_loc_all, gene_hv, 
                                                        patch_size=patch_size,
                                                        dataset_class=args.dataset_class)
        print(" **** All_spot_all_variable shape:", all_spot_all_variable.shape)
        print(" **** adata_infer_all: \n", adata_infer_all)
        adata_infer = adata_infer_all
        spatial_loc = spatial_loc_all
        C2 = C2_all

    elif args.dataset_class == 'VisiumSC': 
        all_dataset_sc = build_loaders_inference_allimage(
            batch_size=len(file_paths),
            file_paths_spot=os.path.join(args.system_path, args.image_embed_path_sc, '*.pth'),
            spatial_pos_path=os.path.join(args.system_path, args.spatial_pos_path_sc), 
            dataset_class=args.dataset_class
        )
        logger.info("Running inference task single-nuclei...")

        start_infer_time = time.time()
        (recon_ref_adata_image_f2, reconstructed_matrix_reshaped,
        _, _, input_coord_all) = perform_inference_image_between_spot(model, all_dataset_sc, dataset_class=args.dataset_class)
        print("--- %s seconds for inference single-nuclei spots ---" % (time.time() - start_infer_time))
        print(" **** Reconstructed_matrix_reshaped shape: ", reconstructed_matrix_reshaped.shape)
        logger.info("Running inference task single-nuclei DONE!")

        ## Get coords
        spatial_loc_sc = get_allspot_coors(input_coord_all)
        print(" **** The spatial coords of all single-nuclei: \n", spatial_loc_sc)

        ## Plot -- omit for inference results visualization
        # figure_dir = get_figure_save_path(args)
        # gene_expr_allspots(args.gene_selected, spatial_loc_sc, recon_ref_adata_image_f2, gene_hv, 
        #                 'Inferred single-cell', s=0.6, 
        #                 figsize=(5, 4),
        #                 save_path=os.path.join(figure_dir, str(args.gene_selected)+'_single-cell_inferred.pdf'))

        ## reshape
        reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(reconstructed_matrix_reshaped, 
                                                                    dataset_class=args.dataset_class)
        print(" **** The size of all reconstructed tensor data:", reconstructed_matrix_reshaped_tensor.shape)

        # Use patch_size parameter (from notebook: patch_size=14 for VisiumSC)
        (_, _, all_spot_all_variable, 
        C2_sc, adata_infer_sc) = subspot_coord_expr_adata(reconstructed_matrix_reshaped_tensor,
                                                        spatial_loc_sc, gene_hv, 
                                                        patch_size=14,
                                                        dataset_class=args.dataset_class)
        print(" **** All_spot_all_variable shape:", all_spot_all_variable.shape)
        print(" **** adata_infer_sc: \n", adata_infer_sc)

        adata_infer = adata_infer_sc
        spatial_loc = spatial_loc_sc
        C2 = C2_sc

    else:
        raise ValueError('Invalid dataset_class. Only "Visium16", "Visium64", "VisiumSC" and "VisiumHD" are supported.')

    return adata_infer, spatial_loc, C2


def main(args):
    """
    Main function for high-resolution imputation.
    
    This function:
    1. Loads and processes data
    2. Loads trained model
    3. Performs inference on all spots (within + between for Visium, or single-nuclei for VisiumSC)
    4. Imputes super-resolved gene expression
    5. Saves results and visualizations
    """
    try:
        # Setup log file first (before any other output)
        tee, log_file_path, timestamp = setup_log_file(args)
        
        # Get figure directory for logging setup
        figure_dir = get_figure_save_path(args)
        
        # Setup logging
        logger, parame_path, params, _ = setup_logging(args, timestamp, figure_dir)
        
        # Check if required files exist
        required_files = [
            args.LRgene_path,
            os.path.join(args.system_path, args.parame_path)
        ]
        
        if args.dataset_class in ['Visium16', 'Visium64']:
            required_files.extend([
                os.path.join(args.system_path, args.visium_path),
                os.path.join(args.system_path, args.imag_within_path),
                os.path.join(args.system_path, args.imag_betwen_path)
            ])
        elif args.dataset_class == 'VisiumSC':
            # For VisiumSC, we need image_embed_path_sc and spatial_pos_path_sc
            # visium_path and imag_within_path are not needed
            required_files.append(os.path.join(args.system_path, args.image_embed_path_sc))
            if hasattr(args, 'spatial_pos_path_sc') and args.spatial_pos_path_sc:
                # spatial_pos_path_sc is optional (will be generated if not exists)
                pass
        
        for file_path in required_files:
            if not check_file_exists(file_path):
                return

        # Load and process data
        adata, gene_hv, file_paths = load_and_process_data(args)

        # Load the trained model
        # weight_save_path should be the full path to the weights directory
        if os.path.isabs(args.weight_save_path):
            weight_dir = args.weight_save_path
        else:
            weight_dir = os.path.join(args.system_path, args.weight_save_path)
        
        model = load_model(weight_dir, parame_path, params, gene_hv)

        # Perform inference
        adata_infer, spatial_loc, C2 = infer_gene_expr(model, file_paths, args, gene_hv, logger)

        if args.dataset_class in ['Visium16', 'Visium64']:
            ########################################
            # Impute super-resolved gene expr.
            ########################################
            # Use dataset_class and weight_exponent=2 (from notebook)
            # Note: impute_adata needs 'Visium16' or 'Visium64', not 'Visium'
            adata_smooth_all = impute_adata(adata, adata_infer, C2, gene_hv, 
                                           dataset_class=args.dataset_class, 
                                           weight_exponent=2)
            print("adata_smooth_all: \n", adata_smooth_all)
            
            adata_impt_all, data_impt_all = weight_adata(adata_infer, adata_smooth_all, gene_hv, w=0.5)
            print("adata_impt_all: \n", adata_impt_all)
            
            # Ensure directory exists before saving
            adata_all_supr_full_path = os.path.join(args.system_path, args.adata_all_supr_path)
            ensure_dir_exists(adata_all_supr_full_path)
            adata_impt_all.write_h5ad(adata_all_supr_full_path)   

            _, adata_impt_all_reshape = reshape_latent_image(data_impt_all, dataset_class=args.dataset_class)
            print("data_impt_all shape:", adata_impt_all.shape)
            print("adata_impt_all_reshape shape:", adata_impt_all_reshape.shape)

            ########################################
            # Convert to spot-resolved gene expr.
            ########################################
            # Use reshape2adata function (from notebook)
            adata_impt_spot = reshape2adata(adata, adata_impt_all_reshape, gene_hv, spatial_loc_all=spatial_loc)
            print("adata_impt_spot: \n", adata_impt_spot)

            # Ensure directory exists before saving
            adata_all_spot_full_path = os.path.join(args.system_path, args.adata_all_spot_path)
            ensure_dir_exists(adata_all_spot_full_path)
            adata_impt_spot.write_h5ad(adata_all_spot_full_path)  

            ########################################
            # Visualize predicted gene expr.
            ########################################
            figure_dir = get_figure_save_path(args)
            gene_expr_allspots(args.gene_selected, spatial_loc, adata_impt_all_reshape, gene_hv, 
                            'FineST all spot', marker='h', s=2.5, 
                            figsize=(5, 4),
                            save_path=os.path.join(figure_dir, str(args.gene_selected)+'_all_spot.pdf'))
            logger.info("Running low-resolution all-spot plot DONE!")

            gene_expr_allspots(args.gene_selected, C2, adata_impt_all.X, gene_hv, 
                            'FineST all sub-spot', marker='s', s=0.3, 
                            figsize=(15, 12),
                            save_path=os.path.join(figure_dir, str(args.gene_selected)+'_all_sub-spot.pdf'))
            logger.info("Running high-resolution all-sub-spot plot DONE!")

        elif args.dataset_class == 'VisiumSC': 

            ########################################
            # Impute super-resolved gene expr.
            ########################################
            # Use dataset_class and weight_exponent=2 (from notebook)
            adata_smooth_sc = impute_adata(adata, adata_infer, C2, gene_hv, 
                                          dataset_class=args.dataset_class, 
                                          weight_exponent=2)
            print("adata_smooth_sc: \n", adata_smooth_sc)
            
            adata_impt_sc, data_impt_sc = weight_adata(adata_infer, adata_smooth_sc, gene_hv, w=0.5)
            print("adata_impt_sc: \n", adata_impt_sc)
            
            # Ensure directory exists before saving
            adata_super_sc_full_path = os.path.join(args.system_path, args.adata_super_path_sc)
            ensure_dir_exists(adata_super_sc_full_path)
            adata_impt_sc.write_h5ad(adata_super_sc_full_path)   

            _, adata_impt_sc_reshape = reshape_latent_image(data_impt_sc, dataset_class=args.dataset_class)
            print("data_impt_sc shape:", adata_impt_sc.shape)
            print("adata_impt_sc_reshape shape:", adata_impt_sc_reshape.shape)

            ########################################
            # Visualize predicted gene expr.
            ########################################
            figure_dir = get_figure_save_path(args)
            gene_expr_allspots(args.gene_selected, spatial_loc, adata_impt_sc_reshape, gene_hv, 
                            'FineST single-cell', s=0.6, 
                            figsize=(5, 4),
                            save_path=os.path.join(figure_dir, str(args.gene_selected)+'_all_sc.pdf'))
            logger.info("Running low-resolution all-spot plot DONE!")

        else:
            raise ValueError('Invalid dataset_class. Only "Visium16", "Visium64", "VisiumSC" and "VisiumHD" are supported.')
    
    finally:
        # Restore stdout and stderr, close log file
        if 'tee' in locals():
            sys.stdout = tee.terminal
            sys.stderr = tee.terminal
            tee.close()
            print(f"Log file closed: {log_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FineST High-Resolution Imputation")

    parser.add_argument('--system_path', type=str, required=True, help='System path for data and weights')
    parser.add_argument('--LRgene_path', type=str, required=True, help='Path to LR genes (relative to system_path)')
    parser.add_argument('--dataset_class', type=str, required=True, 
                       help='Dataset class: Visium16, Visium64, VisiumSC, or VisiumHD')
    parser.add_argument('--gene_selected', type=str, required=True, help='Marker gene for visualization')
    parser.add_argument('--weight_path', type=str, default='weights', help='Directory to save weights (deprecated, not used)')
    parser.add_argument('--parame_path', type=str, required=True, help='Path to parameter file (relative to system_path)')
    parser.add_argument('--visium_path', type=str, required=False, help='Path to Visium data (relative to system_path, required for Visium16/Visium64)')

    ## sub-spot from geometric segmentation: 
    parser.add_argument('--imag_within_path', type=str, required=False, 
                       help='Path to within-spot image embeddings (relative to system_path)')
    parser.add_argument('--imag_betwen_path', type=str, required=False, 
                       help='Path to between-spot image embeddings (relative to system_path)')
    parser.add_argument('--spatial_pos_path', type=str, default='spatial_pos.csv', 
                       help='Path to save spatial positions (relative to system_path)')
    parser.add_argument('--adata_all_supr_path', type=str, required=False, 
                       help='Path to predicted super adata (relative to system_path)')
    parser.add_argument('--adata_all_spot_path', type=str, required=False, 
                       help='Path to predicted spot adata (relative to system_path)')

    ## single-nuclei from nuclei segmentation: 
    parser.add_argument('--image_embed_path_sc', type=str, required=False, 
                       help='Path to single-nuclei image embeddings (relative to system_path)')
    parser.add_argument('--spatial_pos_path_sc', type=str, required=False, 
                       help='Path to save sc spatial positions (relative to system_path)')
    parser.add_argument('--adata_super_path_sc', type=str, required=False, 
                       help='Path to predicted super adata for sc (relative to system_path)')

    parser.add_argument('--figure_save_path', type=str, default='figures', 
                       help='Directory to save figures (relative to system_path)')
    parser.add_argument('--weight_save_path', type=str, required=True, 
                       help='Path to pre-trained weights directory (relative to system_path or absolute)')

    args = parser.parse_args()

    main(args)


## Python Script Examples:

###################
# Example 1: High-resolution imputation for geometric segmentation (Visium16 with HIPT)
###################
# python ./demo/Step2_High_resolution_imputation.py \
#     --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#     --parame_path 'parameter/parameters_NPC_HIPT.json' \
#     --dataset_class 'Visium16' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv' \
#     --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
#     --imag_within_path 'FineST_tutorial_data/ImgEmbeddings/pth_64_16' \
#     --imag_betwen_path 'FineST_tutorial_data/ImgEmbeddings/NEW_pth_64_16' \
#     --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order_all.csv' \
#     --weight_save_path 'FineST_tutorial_data/Figures/weights20260204191708183236' \
#     --figure_save_path 'FineST_tutorial_data/Figures/' \
#     --adata_all_supr_path 'FineST_tutorial_data/SaveData/adata_imput_all_subspot.h5ad' \
#     --adata_all_spot_path 'FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad'

###################
# Example 2: High-resolution imputation for geometric segmentation (Visium64 with Virchow2)
###################
# python ./demo/Step2_High_resolution_imputation.py \
#     --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#     --parame_path 'FineST_tutorial_data/parameter/parameters_NPC_virchow2.json' \
#     --dataset_class 'Visium64' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST_tutorial_data/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
#     --imag_within_path 'FineST_tutorial_data/ImgEmbeddings/pth_112_14' \
#     --imag_betwen_path 'FineST_tutorial_data/ImgEmbeddings/NEW_pth_112_14' \
#     --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order_all.csv' \
#     --weight_save_path 'FineST_tutorial_data/Figures/weights20260204191708183236' \
#     --figure_save_path 'FineST_tutorial_data/Figures/' \
#     --adata_all_supr_path 'FineST_tutorial_data/SaveData/adata_imput_all_subspot.h5ad' \
#     --adata_all_spot_path 'FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad'

###################
# Example 3: High-resolution imputation for nuclei segmentation (VisiumSC with HIPT)
###################
# python ./demo/Step2_High_resolution_imputation.py \
#     --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#     --parame_path 'parameter/parameters_NPC_HIPT.json' \
#     --dataset_class 'VisiumSC' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv' \
#     --image_embed_path_sc 'FineST_tutorial_data/ImgEmbeddings/sc_pth_16_16' \
#     --spatial_pos_path_sc 'FineST_tutorial_data/OrderData/position_order_sc.csv' \
#     --weight_save_path 'FineST_tutorial_data/Figures/weights20260204191708183236' \
#     --figure_save_path 'FineST_tutorial_data/Figures/' \
#     --adata_super_path_sc 'FineST_tutorial_data/SaveData/adata_imput_all_sc.h5ad'

###################
# Notes:
# - dataset_class: 'Visium16', 'Visium64', or 'VisiumSC'
#   * Visium16: 16 sub-spots per spot (use with HIPT, patch_size=64)
#   * Visium64: 64 sub-spots per spot (use with Virchow2, patch_size=112)
#   * VisiumSC: Single-nuclei resolution (from nuclei segmentation)
# - imag_within_path: Path to within-spot image embeddings (from Step0)
# - imag_betwen_path: Path to between-spot image embeddings (from Step0, only for Visium16/Visium64)
# - image_embed_path_sc: Path to single-nuclei image embeddings (from Step0, only for VisiumSC)
# - weight_save_path: Path to pre-trained weights directory (from Step1)
#   * Should point to the weights directory created during Step1 training
#   * Format: 'FineST_tutorial_data/Figures/weights[timestamp]' or absolute path
# - spatial_pos_path: Path to save/load all spot coordinates (within + between)
#   * Will be generated if it doesn't exist
# - spatial_pos_path_sc: Path to save/load single-nuclei coordinates (only for VisiumSC)
#   * Will be generated if it doesn't exist
# - adata_all_supr_path: Output path for imputed sub-spot level gene expression
# - adata_all_spot_path: Output path for imputed spot level gene expression (only for Visium16/Visium64)
# - adata_super_path_sc: Output path for imputed single-nuclei gene expression (only for VisiumSC)
# - All paths are relative to system_path unless specified as absolute paths

