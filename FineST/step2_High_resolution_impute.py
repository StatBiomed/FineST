## 2026.08.05 copy step2_High_resolution_imputation.py from demo to FineST/FineST
## 2026.08.05 add step2_high_resolution_imputation package API

"""
Step 2: FineST high-resolution spatial RNA-seq imputation.

Use as a package API::

    import FineST as fst
    fst.step2_high_resolution_imputation(...)

Or from the terminal::

    python -m FineST.step2_High_resolution_impute --system_path ... --dataset_class Visium16 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from . import datasets
from .inference import *
from .model import *
from .plottings import *
from .processData import *
from .utils import *

from .paths import (
    apply_data_root_step2,
    infer_data_root,
    normalize_data_root,
    tutorial_path_presets,
)

warnings.filterwarnings('ignore')
setup_seed(666)

try:
    from .utils import device
except Exception:  # pragma: no cover
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TeeOutput:
    """Write stdout/stderr to both console and a log file."""

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

def resolve_step2_paths(args):
    """Fill default output paths from ``--data_root`` or embedding directories."""
    hist = getattr(args, 'hist_model', None) or getattr(args, 'image_class', 'HIPT')
    data_root = getattr(args, 'data_root', None)
    if data_root:
        args = apply_data_root_step2(args, normalize_data_root(data_root), hist_model=hist)

    embed_path = args.imag_within_path or args.imag_betwen_path or args.image_embed_path_sc
    if embed_path:
        data_root = infer_data_root(embed_path)
    elif data_root:
        data_root = normalize_data_root(data_root)
    else:
        return args

    presets = tutorial_path_presets(data_root, hist_model=hist)
    defaults = {
        'figure_save_path': presets['figure_save_path'],
        'spatial_pos_path': presets['position_order_all_path'],
        'spatial_pos_path_sc': presets['position_order_sc_path'],
        'adata_all_supr_path': presets['adata_all_supr_path'],
        'adata_all_spot_path': presets['adata_all_spot_path'],
        'adata_super_path_sc': presets['adata_super_path_sc'],
    }
    for key, value in defaults.items():
        current = getattr(args, key, None)
        if current is None or current in ('figures', 'spatial_pos.csv'):
            setattr(args, key, value)
    return args

def resolve_gene_list(args):
    """Resolve LR gene list keyword or file path for adata_LR."""
    if args.LRgene_path in (None, '', 'LR_genes', 'HV_genes', 'LR_HV_genes'):
        return args.LRgene_path or 'LR_genes', getattr(args, 'species', 'human')
    gene_path = args.LRgene_path
    if not os.path.isabs(gene_path):
        candidate = os.path.join(args.system_path, gene_path)
        if os.path.exists(candidate):
            gene_path = candidate
    return gene_path, getattr(args, 'species', 'human')

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
    
    gene_list, species = resolve_gene_list(args)
    adata = adata_LR(adata, gene_list=gene_list, species=species)
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
        
        all_dataset = build_loader_allspot(
            batch_size=len(file_paths), 
            file_paths_spot=os.path.join(args.system_path, args.imag_within_path, '*.pth'),
            file_paths_between_spot=os.path.join(args.system_path, args.imag_betwen_path, '*.pth'), 
            spatial_pos_path=os.path.join(args.system_path, args.spatial_pos_path), 
            dataset_class=args.dataset_class
        )
        logger.info("Running inference task between spot...")

        start_infer_time = time.time()
        (recon_ref_adata_image_f2, reconstructed_matrix_reshaped,
        _, _, input_coord_all) = infer_expr_img2mat(model, all_dataset, logger, dataset_class=args.dataset_class)
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
        all_dataset_sc = build_loader_allspot(
            batch_size=len(file_paths),
            file_paths_spot=os.path.join(args.system_path, args.image_embed_path_sc, '*.pth'),
            spatial_pos_path=os.path.join(args.system_path, args.spatial_pos_path_sc), 
            dataset_class=args.dataset_class
        )
        logger.info("Running inference task single-nuclei...")

        start_infer_time = time.time()
        (recon_ref_adata_image_f2, reconstructed_matrix_reshaped,
        _, _, input_coord_all) = infer_expr_img2mat(model, all_dataset_sc, logger, dataset_class=args.dataset_class)
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
    args = resolve_step2_paths(args)
    try:
        # Setup log file first (before any other output)
        tee, log_file_path, timestamp = setup_log_file(args)
        
        # Get figure directory for logging setup
        figure_dir = get_figure_save_path(args)
        
        # Setup logging
        logger, parame_path, params, _ = setup_logging(args, timestamp, figure_dir)
        
        # Check if required files exist
        required_files = [os.path.join(args.system_path, args.parame_path)]
        gene_list, _ = resolve_gene_list(args)
        if isinstance(gene_list, str) and (gene_list.endswith('.csv') or os.path.exists(gene_list)):
            required_files.append(gene_list if os.path.isabs(gene_list)
                                  else os.path.join(args.system_path, gene_list))
        
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
        
        model = load_model(weight_dir, parame_path, gene_hv)

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


def step2_high_resolution_imputation(
    system_path,
    dataset_class,
    gene_selected,
    parame_path,
    weight_save_path,
    LRgene_path='LR_genes',
    species='human',
    visium_path=None,
    imag_within_path=None,
    imag_betwen_path=None,
    spatial_pos_path=None,
    adata_all_supr_path=None,
    adata_all_spot_path=None,
    image_embed_path_sc=None,
    spatial_pos_path_sc=None,
    adata_super_path_sc=None,
    figure_save_path=None,
    weight_path='weights',
):
    """Run Step 2 high-resolution imputation (sub-spot or single-nuclei)."""
    _ = weight_path  # backward-compatible
    args = argparse.Namespace(
        system_path=system_path,
        LRgene_path=LRgene_path,
        species=species,
        dataset_class=dataset_class,
        gene_selected=gene_selected,
        weight_path=weight_path,
        parame_path=parame_path,
        visium_path=visium_path,
        imag_within_path=imag_within_path,
        imag_betwen_path=imag_betwen_path,
        spatial_pos_path=spatial_pos_path,
        adata_all_supr_path=adata_all_supr_path,
        adata_all_spot_path=adata_all_spot_path,
        image_embed_path_sc=image_embed_path_sc,
        spatial_pos_path_sc=spatial_pos_path_sc,
        adata_super_path_sc=adata_super_path_sc,
        figure_save_path=figure_save_path,
        weight_save_path=weight_save_path,
    )
    main(args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='FineST Step2: high-resolution imputation')
    parser.add_argument('--system_path', type=str, required=True, help='System path for data and weights')
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Dataset root (e.g. FineST_tutorial_data); fills default paths when set',
    )
    parser.add_argument(
        '--hist_model',
        type=str,
        default='HIPT',
        help='HIPT or Virchow2 (must match Step0; used with --data_root)',
    )
    parser.add_argument('--LRgene_path', type=str, default='LR_genes',
                        help="LR gene source: 'LR_genes' (default), 'HV_genes', 'LR_HV_genes', or CSV path")
    parser.add_argument('--species', type=str, default='human',
                        help="Species for bundled LR gene list: 'human' or 'mouse' (default: human)")
    parser.add_argument('--dataset_class', type=str, required=True,
                        help='Dataset class: Visium16, Visium64, VisiumSC, or VisiumHD')
    parser.add_argument('--gene_selected', type=str, required=True, help='Marker gene for visualization')
    parser.add_argument('--weight_path', type=str, default='weights',
                        help='Directory to save weights (deprecated, not used)')
    parser.add_argument('--parame_path', type=str, required=True,
                        help='Path to parameter file (relative to system_path)')
    parser.add_argument('--visium_path', type=str, required=False,
                        help='Path to Visium data (relative to system_path, required for Visium16/Visium64)')
    parser.add_argument('--imag_within_path', type=str, required=False,
                        help='Path to within-spot image embeddings (relative to system_path)')
    parser.add_argument('--imag_betwen_path', type=str, required=False,
                        help='Path to between-spot image embeddings (relative to system_path)')
    parser.add_argument('--spatial_pos_path', type=str, default=None,
                        help='Path to save spatial positions (default: <data_root>/OrderData/position_order_all.csv)')
    parser.add_argument('--adata_all_supr_path', type=str, default=None,
                        help='Path to sub-spot imputed h5ad (default: <data_root>/SaveData/adata_imput_all_subspot.h5ad)')
    parser.add_argument('--adata_all_spot_path', type=str, default=None,
                        help='Path to spot-level imputed h5ad (default: <data_root>/SaveData/adata_imput_all_spot.h5ad)')
    parser.add_argument('--image_embed_path_sc', type=str, required=False,
                        help='Path to single-nuclei image embeddings (relative to system_path)')
    parser.add_argument('--spatial_pos_path_sc', type=str, default=None,
                        help='Path to sc spatial positions (default: <data_root>/OrderData/position_order_sc.csv)')
    parser.add_argument('--adata_super_path_sc', type=str, default=None,
                        help='Path to sc imputed h5ad (default: <data_root>/SaveData/adata_imput_all_sc.h5ad)')
    parser.add_argument('--figure_save_path', type=str, default=None,
                        help='Directory to save figures (default: <data_root>/Figures/)')
    parser.add_argument('--weight_save_path', type=str, required=True,
                        help='Path to pre-trained weights directory (relative to system_path or absolute)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())


## Python Script Examples:

###################
# Example 1: High-resolution imputation for geometric segmentation (Visium16 with HIPT)
###################
# python -m FineST.step2_High_resolution_impute \
#     --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#     --parame_path 'parameter/parameters_NPC_HIPT.json' \
#     --dataset_class 'Visium16' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv' \
#     --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
#     --imag_within_path 'FineST_tutorial_data/ImgEmbeddings/HIPT/pth_64_16' \
#     --imag_betwen_path 'FineST_tutorial_data/ImgEmbeddings/HIPT/NEW_pth_64_16' \
#     --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order_all.csv' \
#     --weight_save_path 'FineST_tutorial_data/Figures/weights20260204191708183236' \
#     --figure_save_path 'FineST_tutorial_data/Figures/' \
#     --adata_all_supr_path 'FineST_tutorial_data/SaveData/adata_imput_all_subspot.h5ad' \
#     --adata_all_spot_path 'FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad'

###################
# Example 2: High-resolution imputation for geometric segmentation (Visium64 with Virchow2)
###################
# python -m FineST.step2_High_resolution_impute \
#     --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#     --parame_path 'FineST_tutorial_data/parameter/parameters_NPC_virchow2.json' \
#     --dataset_class 'Visium64' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST_tutorial_data/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
#     --imag_within_path 'FineST_tutorial_data/ImgEmbeddings/Virchow2/pth_112_14' \
#     --imag_betwen_path 'FineST_tutorial_data/ImgEmbeddings/Virchow2/NEW_pth_112_14' \
#     --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order_all.csv' \
#     --weight_save_path 'FineST_tutorial_data/Figures/weights20260204191708183236' \
#     --figure_save_path 'FineST_tutorial_data/Figures/' \
#     --adata_all_supr_path 'FineST_tutorial_data/SaveData/adata_imput_all_subspot.h5ad' \
#     --adata_all_spot_path 'FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad'

###################
# Example 3: High-resolution imputation for nuclei segmentation (VisiumSC with HIPT)
###################
# python -m FineST.step2_High_resolution_impute \
#     --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#     --parame_path 'parameter/parameters_NPC_HIPT.json' \
#     --dataset_class 'VisiumSC' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv' \
#     --image_embed_path_sc 'FineST_tutorial_data/ImgEmbeddings/HIPT/sc_pth_16_16' \
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