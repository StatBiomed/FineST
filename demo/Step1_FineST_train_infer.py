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

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    return True

def get_figure_save_path(args):
    """
    Get the full path to the figure save directory.
    If figure_save_path is absolute, use it as is.
    Otherwise, join it with system_path.
    Also ensures the directory exists.
    """
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
    File name format: Results + timestamp (similar to weights directory).
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    
    Returns
    -------
    TeeOutput
        TeeOutput object that redirects output to both console and file
    str
        Path to the log file
    str
        Timestamp string (to be used for weights directory as well)
    """
    # Get figure directory (same as where figures are saved)
    figure_dir = get_figure_save_path(args)
    
    # Generate timestamp (will be shared with weights directory)
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

def ensure_dir_exists(file_path):
    """
    Ensure the directory containing the file exists.
    Creates parent directories if they don't exist.
    
    Parameters
    ----------
    file_path : str
        Full path to a file (can be absolute or relative)
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:  # Only create if there's a directory component
        os.makedirs(dir_path, exist_ok=True)

def load_and_process_data(args):
    """
    Load and process spatial transcriptomics data.
    
    This function:
    1. Loads NPC dataset
    2. Filters to LR genes
    3. Preprocesses the data
    4. Aligns image patches with ST spots
    5. Orders data by image coordinates
    6. Saves ordered position and matrix files
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
    file_paths = sorted(os.listdir(os.path.join(args.system_path, args.image_embed_path)))
    print(" **** Image embedding file (First 3): **** \n", file_paths[:3])

    # Map dataset_class to ST_class for image coordinate parsing
    # Visium16 and Visium64 both use 'Visium' format for image file names
    if args.dataset_class in ['Visium16', 'Visium64']:
        ST_class = 'Visium'
    elif args.dataset_class == 'VisiumHD':
        ST_class = 'VisiumHD'
    elif args.dataset_class == 'VisiumSC':
        ST_class = 'VisiumSC'
    else:
        # Default to 'Visium' for backward compatibility
        ST_class = 'Visium'
        print(f"Warning: Unknown dataset_class '{args.dataset_class}', using ST_class='Visium'")
    
    position_image = get_image_coord(file_paths, ST_class)
    position = pd.read_csv(os.path.join(args.system_path, args.visium_path), header=None)
    position = position.rename(columns={position.columns[-2]: 'pixel_x', position.columns[-1]: 'pixel_y'})
    position_image = image_coord_merge(position_image, position, ST_class)
    position_order = update_st_coord(position_image)
    print(" **** The coords of image patch: **** \n", position_order.shape)
    print(position_order.head())
    # Ensure directory exists before saving
    spatial_pos_full_path = os.path.join(args.system_path, args.spatial_pos_path)
    ensure_dir_exists(spatial_pos_full_path)
    position_order.to_csv(spatial_pos_full_path, index=False, header=False)

    spotID_order = np.array(position_image[0])
    matrix_order, matrix_order_df = sort_matrix(adata, position_image, spotID_order, gene_hv)
    # Ensure directory exists before saving
    reduced_mtx_full_path = os.path.join(args.system_path, args.reduced_mtx_path)
    ensure_dir_exists(reduced_mtx_full_path)
    np.save(reduced_mtx_full_path, matrix_order_df.T)

    adata = update_adata_coord(adata, matrix_order_df, position_image)
    
    # Save adata_count (original) and adata_norml (normalized)
    # This matches the notebook behavior: save adata_count and adata_norml after processing
    adata_count = adata.copy()
    adata_norml = adata_preprocess(adata.copy(), normalize=True)
    
    # Save h5ad files if save_data_path is provided
    if hasattr(args, 'save_data_path') and args.save_data_path:
        save_dir = os.path.join(args.system_path, args.save_data_path)
        os.makedirs(save_dir, exist_ok=True)
        adata_count.write_h5ad(os.path.join(save_dir, 'adata_count.h5ad'))
        adata_norml.write_h5ad(os.path.join(save_dir, 'adata_norml.h5ad'))
        print(f" **** Saved adata_count.h5ad and adata_norml.h5ad to {save_dir} ****")
    
    # Ensure figure save directory exists and get full path
    figure_dir = get_figure_save_path(args)
    gene_expr(adata, matrix_order_df, gene_selet=args.gene_selected, 
              save_path=os.path.join(figure_dir, str(args.gene_selected)+'_orig_gene_expr.pdf'))

    return adata, gene_hv, matrix_order_df, adata_count, adata_norml

def setup_logging(args, timestamp, figure_dir):
    """
    Setup logging for training.
    Weights directory will be saved in the same directory as figures (figure_save_path).
    Directory name format: weights + timestamp (same timestamp as log file).
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    timestamp : str
        Timestamp string (shared with log file)
    figure_dir : str
        Figure directory path (where weights will also be saved)
    
    Returns
    -------
    logger : logging.Logger
        Logger instance
    parame_path : str
        Path to parameter file
    params : dict
        Loaded parameters
    dir_name : str
        Path to weights directory
    """
    logging.getLogger().setLevel(logging.INFO)

    # Create weights directory in figure_dir with shared timestamp
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

def train_model_fst_wrapper(params, model, train_loader, test_loader, optimizer, l, dir_name, logger, dataset_class):
    """
    Wrapper function to call train_model_fst from FineST package.
    
    This function calls the FineST train_model_fst function which handles:
    - Training loop with train/test split
    - Loss tracking
    - Model saving at best epoch
    - Returns training history and best model info
    """
    from FineST.traintest import train_model_fst
    return train_model_fst(params, model, train_loader, test_loader, optimizer, l, dir_name, logger, dataset_class)

def infer_gene_expr(model, adata, args, gene_hv, logger, patch_size=112):
    """
    Perform inference to predict gene expression from image features.
    
    This function:
    1. Builds inference data loader
    2. Runs inference using trained model
    3. Reshapes reconstructed matrix
    4. Maps sub-spot coordinates and expression to AnnData format
    
    Parameters
    ----------
    model : FineSTModel
        Trained FineST model
    adata : AnnData
        Original spatial transcriptomics data
    args : argparse.Namespace
        Command line arguments
    gene_hv : np.ndarray
        Array of gene names
    logger : logging.Logger
        Logger for recording progress
    patch_size : int
        Patch size used for image feature extraction (default: 112)
    
    Returns
    -------
    adata_infer : AnnData
        Inferred gene expression at sub-spot resolution
    first_spot_first_variable : np.ndarray
        Expression of first gene in first spot (for visualization)
    C : np.ndarray
        Coordinates for first spot visualization
    C2 : np.ndarray
        All sub-spot coordinates
    """
    from FineST.inference import infer_model_fst
    
    model.to(device)
    test_loader = build_loaders_inference(
        batch_size=adata.shape[0], 
        image_embed_path=os.path.join(args.system_path, args.image_embed_path, '*.pth'),
        spatial_pos_path=os.path.join(args.system_path, args.spatial_pos_path),
        reduced_mtx_path=os.path.join(args.system_path, args.reduced_mtx_path),
        image_clacss=args.image_class,
        dataset_class=args.dataset_class
    )

    logger.info("Running inference task...")
    start_infer_time = time.time()

    # Use infer_model_fst which returns the correct values
    (matrix_profile, reconstructed_matrix, recon_ref_adata_image_f2, 
     reconstructed_matrix_reshaped, input_coord_all) = infer_model_fst(
        model, test_loader, logger, dataset_class=args.dataset_class
    )

    print("--- %s seconds for inference within spots ---" % (time.time() - start_infer_time))
    print("Reconstructed_matrix_reshaped shape: ", reconstructed_matrix_reshaped.shape)
    logger.info("Running inference task DONE!")

    # Reshape to tensor format
    # Note: reconstructed_matrix_reshaped from infer_model_fst is a torch tensor
    # reshape_latent_image requires torch tensor and works for both HIPT and Virchow2
    # - HIPT + Visium16: reshapes to [n_spots, 16, n_genes]
    # - Virchow2 + Visium64: reshapes to [n_spots, 64, n_genes]
    reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(
        reconstructed_matrix_reshaped, 
        dataset_class=args.dataset_class
    )
    print(" **** The size of reconstructed tensor data:", reconstructed_matrix_reshaped_tensor.shape)

    # Get first spot first variable for visualization
    # This works for both HIPT and Virchow2:
    # - HIPT: p=0, q=0, patch_size=64, dataset_class='Visium16'
    # - Virchow2: p=0, q=0, patch_size=112, dataset_class='Visium64'
    (first_spot_first_variable, C, 
     _, _, _) = subspot_coord_expr_adata(
        reconstructed_matrix_reshaped_tensor,
        adata, gene_hv, p=0, q=0, 
        patch_size=patch_size,
        dataset_class=args.dataset_class
    )
    
    # Get all spots all variables
    # This extracts all sub-spots for all spots:
    # - HIPT + Visium16: 16 sub-spots per spot
    # - Virchow2 + Visium64: 64 sub-spots per spot
    (_, _, 
     all_spot_all_variable, C2, adata_infer) = subspot_coord_expr_adata(
        reconstructed_matrix_reshaped_tensor,
        adata, gene_hv, 
        patch_size=patch_size,
        dataset_class=args.dataset_class
    )
    print(" **** All_spot_all_variable shape:", all_spot_all_variable.shape)

    return adata_infer, first_spot_first_variable, C, C2

def main(args):
    # Setup log file to save all terminal output (generates shared timestamp)
    tee, log_file_path, timestamp = setup_log_file(args)
    
    try:
        # Check if required files exist
        required_files = [
            os.path.join(args.system_path, args.LRgene_path), 
            os.path.join(args.system_path, args.visium_path),
            os.path.join(args.system_path, args.image_embed_path),
            os.path.join(args.system_path, args.parame_path)
        ]
        
        for file_path in required_files:
            if not check_file_exists(file_path):
                return

        # Ensure figure save directory exists
        figure_dir = get_figure_save_path(args)

        # Load and process data
        adata, gene_hv, matrix_order_df, adata_count, adata_norml = load_and_process_data(args)

        # Setup logging (use shared timestamp and figure_dir)
        logger, parame_path, params, dir_name = setup_logging(args, timestamp, figure_dir)

        # Initialize the model
        params['n_input_matrix'] = len(gene_hv)
        # n_input_image should be set in parameter file:
        # - 384 for HIPT
        # - 1280 for Virchow2
        # If not set, use default based on image_class
        if 'n_input_image' not in params:
            if args.image_class == 'HIPT':
                params['n_input_image'] = 384
            elif args.image_class == 'Virchow2':
                params['n_input_image'] = 1280
            else:
                params['n_input_image'] = 384  # default to HIPT
                logger.warning(f"Unknown image_class {args.image_class}, using default n_input_image=384")

        model = FineSTModel(
            n_input_matrix=params['n_input_matrix'],
            n_input_image=params['n_input_image'],
            n_encoder_hidden_matrix=params["n_encoder_hidden_matrix"],
            n_encoder_hidden_image=params["n_encoder_hidden_image"],
            n_encoder_latent=params["n_encoder_latent"],
            n_projection_hidden=params["n_projection_hidden"],
            n_projection_output=params["n_projection_output"],
            n_encoder_layers=params["n_encoder_layers"]
        ).to(device)

        # Initialize loss function with w1, w2, w3, w4 parameters
        l = ContrastiveLoss(
            temperature=params['temperature'],
            w1=params.get('w1', 0),
            w2=params.get('w2', 0),
            w3=params.get('w3', 1),
            w4=params.get('w4', 1)
        )

        # Set optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=params['inital_learning_rate'], 
                                    momentum=0.9, weight_decay=5e-4)

        # Load the data - add image_clacss parameter
        train_loader, test_loader = build_loaders(
            batch_size=params['batch_size'],
            image_embed_path=os.path.join(args.system_path, args.image_embed_path, '*.pth'),
            spatial_pos_path=os.path.join(args.system_path, args.spatial_pos_path),
            reduced_mtx_path=os.path.join(args.system_path, args.reduced_mtx_path),
            image_clacss=args.image_class,
            dataset_class=args.dataset_class
        )

        # Train the model if no pre-trained weights are provided
        best_epoch = None
        train_losses = None
        test_losses = None
        best_loss = None
        if args.weight_save_path is None:
            (dir_name, train_losses, test_losses, 
             best_epoch, best_loss) = train_model_fst_wrapper(
                params, model, train_loader, test_loader, optimizer, l, dir_name, logger, args.dataset_class
            )
            logger.info(f"Training completed. Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")
            
            # Plot loss curve if training was performed
            if train_losses is not None and test_losses is not None:
                figure_dir = get_figure_save_path(args)
                loss_curve_path = os.path.join(figure_dir, 'loss_curve.svg')
                loss_curve(train_losses, test_losses, best_epoch, best_loss, 
                         max_step=5, min_step=1, fig_size=(5, 4), format='svg', 
                         save_path=loss_curve_path)
                logger.info(f"Loss curve saved to {loss_curve_path}")
        else:
            dir_name = args.weight_save_path

        # Load the trained model
        # Note: load_model signature is (dir_name, parameter_file_path, gene_hv, device=None, best_epoch=None)
        model = load_model(dir_name, parame_path, gene_hv, best_epoch=best_epoch)

        # Perform inference - use patch_size from args or default
        patch_size = getattr(args, 'patch_size', 112)
        adata_infer, first_spot_first_variable, C, C2 = infer_gene_expr(
        model, adata, args, gene_hv, logger, patch_size=patch_size
        )

        ###################################
        # Evaluate inferred gene expr. (inference only, before imputation)
        ###################################
        # Generate adata_infer_reshape for spot-level evaluation
        _, adata_infer_reshape = reshape_latent_image(
        torch.tensor(adata_infer.X), dataset_class=args.dataset_class
        )
    
        # Generate adata_infer_spot for correlation histogram
        adata_infer_spot = reshape2adata(adata, adata_infer_reshape, gene_hv)
    
        # Plot gene expression comparison (inference only)
        gene_expr_compare(adata, args.gene_selected, adata_infer_reshape, gene_hv, s=50, 
                     save_path=os.path.join(figure_dir, str(args.gene_selected)+'_infer_gene_expr.pdf'))
        logger.info("Running gene_expr_compare (inference only) plot DONE!")
    
        # Plot correlation box plots (inference only)
        mean_cor_box(adata, adata_infer_reshape, logger, 
                save_path=os.path.join(figure_dir, 'Boxplot_infer_cor_count.pdf'))
        logger.info("Running mean_cor_box (inference only, count) plot DONE!")
    
        mean_cor_box(adata_norml, adata_infer_reshape, logger, 
                save_path=os.path.join(figure_dir, 'Boxplot_infer_cor_norml.pdf'))
        logger.info("Running mean_cor_box (inference only, normalized) plot DONE!")
    
        # Plot correlation histogram (inference only)
        cor_hist(adata, adata_infer_spot.to_df(), 
            fig_size=(5, 4), trans=False, format='svg', 
            save_path=os.path.join(figure_dir, 'Hist_infer_cor_count.svg'))
        logger.info("Running cor_hist (inference only) plot DONE!")

        ######################
        # Impute gene expr.
        ######################
        # Note: impute_adata doesn't have k parameter, it uses weight_exponent instead
        adata_imput = impute_adata(adata, adata_infer, C2, gene_hv, 
                                dataset_class=args.dataset_class, weight_exponent=2)
        _, data_impt = weight_adata(adata_infer, adata_imput, gene_hv, w=args.weight_w)
        _, data_impt_reshape = reshape_latent_image(
        torch.tensor(data_impt), dataset_class=args.dataset_class
        )
        print(" **** data_impt shape:", data_impt.shape)
        print(" **** data_impt_reshape shape:", data_impt_reshape.shape)

        ###################################
        # Evaluate predicted gene expr.
        ###################################
        subspot_expr(C, first_spot_first_variable, 
                patch_size=patch_size, dataset_class=args.dataset_class,
                save_path=os.path.join(figure_dir, '1st_spot_1st_gene.pdf'))
        logger.info("Running first_spot_first_variable plot DINE!")

        gene_expr_compare(adata, args.gene_selected, data_impt_reshape, gene_hv, s=50, 
                        save_path=os.path.join(figure_dir, str(args.gene_selected)+'_pred_gene_expr.pdf'))
        logger.info("Running gene_expr_compare plot DINE!")

        sele_gene_cor(adata, data_impt_reshape, gene_hv, gene = args.gene_selected, 
                        ylabel='FineST Expression', title = str(args.gene_selected)+' expression', size=5, 
                        save_path=os.path.join(figure_dir, str(args.gene_selected)+'_gene_corr.pdf'))    
        logger.info("Running sele_gene_cor plot DINE!")

        logger.info("Running Gene Correlation task...")
        (pearson_cor_gene, 
        spearman_cor_gene, 
        cosine_sim_gene) = mean_cor(adata, data_impt_reshape, 'reconf2', sample="gene")
        logger.info("Pearson, Spearman, Cosine corr_gene: [{}: {}: {}]".format(pearson_cor_gene, spearman_cor_gene, cosine_sim_gene))
        logger.info("Running Gene Correlation task DINE!")

    
        # mean_cor_box(adata, data_impt_reshape)
        mean_cor_box(adata, data_impt_reshape, logger, save_path=os.path.join(figure_dir, 'Box_spot_gene_corr.pdf'))    
    
        ######################
        # Save inference and imputation results as h5ad files
        ######################
        if hasattr(args, 'save_data_path') and args.save_data_path:
            save_dir = os.path.join(args.system_path, args.save_data_path)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save adata_infer (inferred sub-spot level expression)
            adata_infer.write_h5ad(os.path.join(save_dir, 'adata_infer.h5ad'))
            logger.info(f"Saved adata_infer.h5ad to {save_dir}")
            
            # Generate and save adata_infer_spot (spot-level aggregated from sub-spots)
            # This matches notebook: reshape adata_infer to spot level
            _, adata_infer_reshape = reshape_latent_image(
                torch.tensor(adata_infer.X), dataset_class=args.dataset_class
            )
            adata_infer_spot = reshape2adata(adata, adata_infer_reshape, gene_hv)
            adata_infer_spot.write_h5ad(os.path.join(save_dir, 'adata_infer_spot.h5ad'))
            logger.info(f"Saved adata_infer_spot.h5ad to {save_dir}")
            print(f" **** adata_infer shape: {adata_infer.shape}")
            print(f" **** adata_infer_spot shape: {adata_infer_spot.shape}")
            
            # Save adata_imput (imputed expression)
            adata_imput.write_h5ad(os.path.join(save_dir, 'adata_imput.h5ad'))
            logger.info(f"Saved adata_imput.h5ad to {save_dir}")
    
    finally:
        # Restore stdout and stderr, and close log file
        if 'tee' in locals():
            sys.stdout = tee.terminal
            sys.stderr = tee.terminal
            tee.close()
            print(f"\nLog file saved to: {log_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CellContrast Model Training and Inference")
    parser.add_argument('--system_path', type=str, required=True, help='System path for data and weights')
    parser.add_argument('--LRgene_path', type=str, required=True, help='Path to LR genes')
    parser.add_argument('--dataset_class', type=str, required=True, 
                        help='Dataset class: Visium16, Visium64, or VisiumHD')
    parser.add_argument('--image_class', type=str, default='Virchow2',
                        help='Image feature extraction method: HIPT or Virchow2 (default: Virchow2)')
    parser.add_argument('--gene_selected', type=str, required=True, help='Marker gene visualization')
    parser.add_argument('--image_embed_path', type=str, required=True, help='Path to image embeddings')
    parser.add_argument('--visium_path', type=str, required=True, help='Path to Visium data')
    parser.add_argument('--parame_path', type=str, required=True, help='Path to parameter file')
    parser.add_argument('--spatial_pos_path', type=str, default='spatial_pos.csv', 
                        help='Path to save spatial positions (relative to system_path)')
    parser.add_argument('--reduced_mtx_path', type=str, default='reduced_mtx.npy', 
                        help='Path to save reduced matrix (relative to system_path)')
    parser.add_argument('--figure_save_path', type=str, default='figures', help='Directory to save figures')
    parser.add_argument('--weight_save_path', type=str, default=None, 
                        help='Path to pre-trained weights, if available')
    parser.add_argument('--patch_size', type=int, default=112, 
                        help='Patch size used for image feature extraction (default: 112)')
    parser.add_argument('--save_data_path', type=str, default='SaveData/', 
                        help='Directory to save h5ad files (relative to system_path, default: SaveData/)')
    parser.add_argument('--weight_w', type=float, default=0.5, 
                        help='Weight parameter for combining adata_infer and adata_imput (default: 0.5)')
    args = parser.parse_args()

    main(args)


## Python Script Examples:

###################
# Example 1: Train and Infer (if haven't trained) using Virchow2 with Visium64 (patch_size=112)
###################
# python ./demo/Step1_FineST_train_infer.py \
#     --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#     --parame_path 'FineST_tutorial_data/parameter/parameters_NPC_virchow2.json' \
#     --dataset_class 'Visium64' \
#     --image_class 'Virchow2' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST_tutorial_data/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
#     --image_embed_path 'FineST_tutorial_data/ImgEmbeddings/pth_112_14' \
#     --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order.csv' \
#     --reduced_mtx_path 'FineST_tutorial_data/OrderData/matrix_order.npy' \
#     --figure_save_path 'FineST_tutorial_data/Figures/' \
#     --save_data_path 'FineST_tutorial_data/SaveData/' \
#     --patch_size 112 \
#     --weight_w 0.5

###################
# Example 2: Only Infer (if already trained) using Virchow2 with Visium64 (patch_size=112)
###################
# python ./demo/Step1_FineST_train_infer.py \
#     --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#     --parame_path 'FineST_tutorial_data/parameter/parameters_NPC_virchow2.json' \
#     --dataset_class 'Visium64' \
#     --image_class 'Virchow2' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST_tutorial_data/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
#     --image_embed_path 'FineST_tutorial_data/ImgEmbeddings/pth_112_14' \
#     --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order.csv' \
#     --reduced_mtx_path 'FineST_tutorial_data/OrderData/matrix_order.npy' \
#     --weight_save_path 'FineST_tutorial_data/weights/20250621001835815284' \
#     --figure_save_path 'FineST_tutorial_data/Figures/' \
#   --save_data_path 'FineST_tutorial_data/SaveData/' \
#     --patch_size 112 \
#     --weight_w 0.5

###################
# Example 3: Train and Infer (if haven't trained) using HIPT with Visium16 (patch_size=64)
###################
# python ./demo/Step1_FineST_train_infer.py \
#   --system_path '/home/lingyu/ssd/Python/FineST_submit/FineST/' \
#   --parame_path 'parameter/parameters_NPC_HIPT.json' \
#   --dataset_class 'Visium16' \
#   --image_class 'HIPT' \
#   --gene_selected 'CD70' \
#   --LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv' \
#   --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
#   --image_embed_path 'FineST_tutorial_data/ImgEmbeddings/pth_64_16' \
#   --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order.csv' \
#   --reduced_mtx_path 'FineST_tutorial_data/OrderData/matrix_order.npy' \
#   --figure_save_path 'FineST_tutorial_data/Figures/' \
#   --save_data_path 'FineST_tutorial_data/SaveData/' \
#   --patch_size 64 \
#   --weight_w 0.5

###################
# Notes:
# - dataset_class: 'Visium16', 'Visium64', or 'VisiumHD'
#   * Visium16: 16 sub-spots per spot (use with patch_size=64, HIPT)
#   * Visium64: 64 sub-spots per spot (use with patch_size=112, Virchow2)
#   * VisiumHD: 4 sub-spots per spot (for Visium HD data)
# - image_class: 'HIPT' or 'Virchow2' (must match the method used in Image_feature_extraction.py)
#   * HIPT: n_input_image=384 (default)
#   * Virchow2: n_input_image=1280 (default)
# - patch_size: Should match the patch_size used in Image_feature_extraction.py
#   * For HIPT + Visium16: typically 64
#   * For Virchow2 + Visium64: typically 112
# - image_embed_path: Should match the output from Image_feature_extraction.py
#   * For HIPT + patch_size=64: typically 'pth_64_16'
#   * For Virchow2 + patch_size=112: typically 'pth_112_14'
# - weight_save_path: Only needed when using pre-trained weights (skip training)
# - All paths are relative to system_path unless specified as absolute paths 
