## 2026.08.04 copy the step1_FineST_train_infer.py from demo to FineST/FineST
#             add the step1_FineST_train_infer API function
# 2026.08.05 merge step1_train.py API docs / usage examples into this module

"""
Step 1: FineST Model Training and Inference

This module provides a high-level interface for training FineST models and performing
gene expression inference from image features.

Use as a package API::

    import FineST as fst
    fst.step1_FineST_train_infer(...)

Or from the terminal::

    python -m FineST.step1_FineST_train_infer --system_path ... --dataset_class Visium16 ...
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
import torch

from . import datasets
from .inference import *
from .model import *
from .plottings import *
from .processData import *
from .traintest import train_model
from .utils import *

warnings.filterwarnings('ignore')
setup_seed(666)

# Prefer package-level device if available
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


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    return True


def ensure_dir_exists(file_path):
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


from .paths import (
    apply_data_root_step1,
    infer_data_root,
    normalize_data_root,
    tutorial_path_presets,
)


def load_spatial_dataset(args):
    """Load AnnData for the selected platform."""
    if args.dataset_class == 'VisiumHD':
        return datasets.CRC16um()
    return datasets.NPC()


def resolve_step1_paths(args):
    """Fill default output paths from ``--data_root`` or ``image_embed_path``."""
    hist = getattr(args, 'hist_model', 'HIPT')
    data_root = getattr(args, 'data_root', None)
    if data_root:
        args = apply_data_root_step1(args, normalize_data_root(data_root), hist_model=hist)
    if getattr(args, 'image_embed_path', None):
        data_root = infer_data_root(args.image_embed_path)
    elif data_root:
        data_root = normalize_data_root(data_root)
    else:
        return args
    from .paths import _path_presets_for_dataset

    presets = _path_presets_for_dataset(
        data_root,
        hist_model=hist,
        dataset_class=getattr(args, 'dataset_class', None),
        patch_size=getattr(args, 'patch_size', None),
    )
    defaults = {
        'spatial_pos_path': presets['spatial_pos_path'],
        'reduced_mtx_path': presets['reduced_mtx_path'],
        'figure_save_path': presets['figure_save_path'],
        'save_data_path': presets['save_data_path'],
    }
    for key, value in defaults.items():
        current = getattr(args, key, None)
        if current is None or current in ('spatial_pos.csv', 'reduced_mtx.npy', 'figures', 'SaveData/'):
            setattr(args, key, value)
    return args


def resolve_gene_list(args):
    """Resolve LR gene list keyword or file path for adata_LR."""
    if args.LRgene_path in (None, '', 'LR_genes', 'HV_genes', 'LR_HV_genes'):
        return args.LRgene_path or 'LR_genes', args.species
    gene_path = args.LRgene_path
    if not os.path.isabs(gene_path):
        candidate = os.path.join(args.system_path, gene_path)
        if os.path.exists(candidate):
            gene_path = candidate
    return gene_path, args.species


def get_figure_save_path(args):
    if os.path.isabs(args.figure_save_path):
        figure_dir = args.figure_save_path
    else:
        figure_dir = os.path.join(args.system_path, args.figure_save_path)
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir


def setup_log_file(args):
    """Tee terminal output to ``Results{timestamp}.log`` under the figure directory."""
    figure_dir = get_figure_save_path(args)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    log_file_path = os.path.join(figure_dir, f'Results{timestamp}.log')
    tee = TeeOutput(log_file_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"Log file saved to: {log_file_path}")
    print("=" * 80)
    return tee, log_file_path, timestamp


def setup_logging(args, timestamp, figure_dir):
    """Create weights dir + logger; load parameter JSON."""
    logging.getLogger().setLevel(logging.INFO)
    dir_name = os.path.join(figure_dir, f'weights{timestamp}')
    os.makedirs(dir_name, exist_ok=True)
    logger = setup_logger(dir_name)
    print("dir_name: \n", dir_name)

    parame_path = os.path.join(args.system_path, args.parame_path)
    with open(parame_path, 'r') as json_file:
        params = json.load(json_file)
    logger.info("Load parameters:\n" + json.dumps(params, indent=2))
    return logger, parame_path, params, dir_name


def _st_class_from_dataset(dataset_class):
    if dataset_class in ('Visium16', 'Visium64'):
        return 'Visium'
    if dataset_class == 'VisiumHD':
        return 'VisiumHD'
    if dataset_class == 'VisiumSC':
        return 'VisiumSC'
    print(f"Warning: Unknown dataset_class '{dataset_class}', using ST_class='Visium'")
    return 'Visium'


def load_and_process_data(args):
    """Load ST data, align to image embeddings, and save OrderData."""
    adata = load_spatial_dataset(args)
    print(f" **** Load the original adata ({args.dataset_class}): **** \n", adata)

    gene_list, species = resolve_gene_list(args)
    adata = adata_LR(adata, gene_list=gene_list, species=species)
    adata = adata_preprocess(adata, normalize=False)
    print(f" **** Processed adata ({args.dataset_class}): **** \n", adata)

    ST_class = _st_class_from_dataset(args.dataset_class)
    image_embed_dir = os.path.join(args.system_path, args.image_embed_path)
    position_path = os.path.join(args.system_path, args.visium_path)
    order_save_dir = os.path.dirname(os.path.join(args.system_path, args.spatial_pos_path))

    position_order, position_image, matrix_order_df, gene_hv = order_adata_by_image(
        adata,
        position_path=position_path,
        image_embed_dir=image_embed_dir,
        ST_class=ST_class,
        save_dir=order_save_dir,
    )
    print(" **** The length of LR genes: ", len(gene_hv))
    print(" **** The coords of image patch: **** \n", position_order.shape)
    print(position_order.head())

    # Honor explicit OrderData filenames if they differ from the defaults
    spatial_pos_full = os.path.join(args.system_path, args.spatial_pos_path)
    reduced_mtx_full = os.path.join(args.system_path, args.reduced_mtx_path)
    default_pos = os.path.join(order_save_dir, 'position_order.csv')
    default_mtx = os.path.join(order_save_dir, 'matrix_order.npy')
    if os.path.abspath(spatial_pos_full) != os.path.abspath(default_pos):
        ensure_dir_exists(spatial_pos_full)
        position_order.to_csv(spatial_pos_full, index=False, header=False)
    if os.path.abspath(reduced_mtx_full) != os.path.abspath(default_mtx):
        ensure_dir_exists(reduced_mtx_full)
        np.save(reduced_mtx_full, matrix_order_df.T)

    adata = update_adata_coord(adata, matrix_order_df, position_image)
    adata_count = adata.copy()
    adata_norml = adata_preprocess(adata.copy(), normalize=True)

    if getattr(args, 'save_data_path', None):
        save_dir = os.path.join(args.system_path, args.save_data_path)
        os.makedirs(save_dir, exist_ok=True)
        adata_count.write_h5ad(os.path.join(save_dir, 'adata_count.h5ad'))
        adata_norml.write_h5ad(os.path.join(save_dir, 'adata_norml.h5ad'))
        print(f" **** Saved adata_count.h5ad and adata_norml.h5ad to {save_dir} ****")

    figure_dir = get_figure_save_path(args)
    gene_expr(
        adata,
        matrix_order_df,
        gene_selet=args.gene_selected,
        save_path=os.path.join(figure_dir, str(args.gene_selected) + '_orig_gene_expr.pdf'),
    )
    return adata, gene_hv, matrix_order_df, adata_count, adata_norml


def infer_gene_expr(model, adata, args, gene_hv, logger, patch_size=112):
    """Run within-spot inference and map sub-spot expression to AnnData."""
    model.to(device)
    test_loader = build_loader_withinspot(
        batch_size=adata.shape[0],
        image_embed_path=os.path.join(args.system_path, args.image_embed_path, '*.pth'),
        spatial_pos_path=os.path.join(args.system_path, args.spatial_pos_path),
        reduced_mtx_path=os.path.join(args.system_path, args.reduced_mtx_path),
        hist_model=args.hist_model,
        dataset_class=args.dataset_class,
    )

    logger.info("Running inference task...")
    start_infer_time = time.time()
    (
        _matrix_profile,
        _reconstructed_matrix,
        _recon_ref_adata_image_f2,
        reconstructed_matrix_reshaped,
        _input_coord_all,
    ) = infer_expr(model, test_loader, logger, dataset_class=args.dataset_class)
    print("--- %s seconds for inference within spots ---" % (time.time() - start_infer_time))
    print("Reconstructed_matrix_reshaped shape: ", reconstructed_matrix_reshaped.shape)
    logger.info("Running inference task DONE!")

    reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(
        reconstructed_matrix_reshaped,
        dataset_class=args.dataset_class,
    )
    print(" **** The size of reconstructed tensor data:", reconstructed_matrix_reshaped_tensor.shape)

    (_, _, all_spot_all_variable, C2, adata_infer) = subspot_coord_expr_adata(
        reconstructed_matrix_reshaped_tensor,
        adata,
        gene_hv,
        patch_size=patch_size,
        dataset_class=args.dataset_class,
    )
    print(" **** All_spot_all_variable shape:", all_spot_all_variable.shape)
    return adata_infer, reconstructed_matrix_reshaped_tensor, C2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    if str(v).lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _normalize_hist_model_args(args, *, default='HIPT'):
    """Resolve ``hist_model``; warn when deprecated ``image_class`` is used."""
    legacy = getattr(args, 'image_class', None)
    hist = getattr(args, 'hist_model', None)
    if legacy is not None:
        warnings.warn(
            'image_class is deprecated and will be removed; use hist_model instead.',
            DeprecationWarning,
            stacklevel=3,
        )
    if legacy is not None and hist is not None and str(legacy) != str(hist):
        raise ValueError(
            f'Conflicting hist_model={hist!r} and deprecated image_class={legacy!r}'
        )
    args.hist_model = resolve_hist_model(hist, legacy, default=default)
    return args


def default_patch_size(dataset_class, hist_model):
    if dataset_class == 'VisiumHD':
        return 32 if hist_model == 'HIPT' else 28
    if dataset_class == 'Visium16':
        return 64
    if dataset_class == 'Visium64':
        return 112
    if hist_model == 'HIPT':
        return 64
    if hist_model == 'Virchow2':
        return 112
    return 64


def main(args):
    """Run the full Step1 train + within-spot infer pipeline."""
    args = _normalize_hist_model_args(args)
    print("torch version: %s" % torch.__version__)
    args = resolve_step1_paths(args)
    tee, log_file_path, timestamp = setup_log_file(args)

    try:
        required_files = [
            os.path.join(args.system_path, args.visium_path),
            os.path.join(args.system_path, args.image_embed_path),
            os.path.join(args.system_path, args.parame_path),
        ]
        gene_list, _ = resolve_gene_list(args)
        if isinstance(gene_list, str) and (gene_list.endswith('.csv') or os.path.exists(gene_list)):
            required_files.append(
                gene_list if os.path.isabs(gene_list) else os.path.join(args.system_path, gene_list)
            )
        for file_path in required_files:
            if not check_file_exists(file_path):
                return

        figure_dir = get_figure_save_path(args)
        adata, gene_hv, matrix_order_df, adata_count, adata_norml = load_and_process_data(args)
        logger, parame_path, params, dir_name = setup_logging(args, timestamp, figure_dir)

        params['n_input_matrix'] = len(gene_hv)
        if 'n_input_image' not in params:
            if args.hist_model == 'HIPT':
                params['n_input_image'] = 384
            elif args.hist_model == 'Virchow2':
                params['n_input_image'] = 1280
            else:
                params['n_input_image'] = 384
                logger.warning(f"Unknown hist_model {args.hist_model}, using default n_input_image=384")

        ## 2026.08.05 add the prepare_training function for the API function
        image_embed_path = os.path.join(args.system_path, args.image_embed_path, '*.pth')
        model, train_loader, test_loader, optimizer, l = prepare_training(
            params,
            image_embed_path,
            device=device,
            hist_model=args.hist_model,
            dataset_class=args.dataset_class,
            spatial_pos_path=os.path.join(args.system_path, args.spatial_pos_path),
            reduced_mtx_path=os.path.join(args.system_path, args.reduced_mtx_path),
        )

        best_epoch = None
        train_losses = None
        test_losses = None
        best_loss = None
        if args.weight_save_path is None:
            (dir_name, train_losses, test_losses, best_epoch, best_loss) = train_model(
                params,
                model,
                train_loader,
                test_loader,
                optimizer,
                l,
                dir_name,
                logger,
                dataset_class=args.dataset_class,
            )
            logger.info(f"Training completed. Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")
            if train_losses is not None and test_losses is not None:
                loss_curve_path = os.path.join(figure_dir, 'loss_curve.svg')
                loss_curve(
                    train_losses,
                    test_losses,
                    best_epoch,
                    best_loss,
                    max_step=5,
                    min_step=1,
                    fig_size=(5, 4),
                    format='svg',
                    save_path=loss_curve_path,
                )
                logger.info(f"Loss curve saved to {loss_curve_path}")
        else:
            dir_name = (
                args.weight_save_path
                if os.path.isabs(args.weight_save_path)
                else os.path.join(args.system_path, args.weight_save_path)
            )

        model = load_model(dir_name, parame_path, gene_hv, best_epoch=best_epoch)
        patch_size = getattr(args, 'patch_size', None) or default_patch_size(
            args.dataset_class, args.hist_model
        )
        adata_infer, reconstructed_matrix_reshaped_tensor, C2 = infer_gene_expr(
            model, adata, args, gene_hv, logger, patch_size=patch_size
        )

        _, adata_infer_reshape = reshape_latent_image(
            torch.tensor(adata_infer.X), dataset_class=args.dataset_class
        )
        adata_infer_spot = reshape2adata(adata, adata_infer_reshape, gene_hv)

        gene_expr_compare(
            adata,
            args.gene_selected,
            adata_infer_reshape,
            gene_hv,
            s=50,
            save_path=os.path.join(figure_dir, str(args.gene_selected) + '_infer_gene_expr.pdf'),
        )
        logger.info("Running gene_expr_compare (inference only) plot DONE!")
        mean_cor_box(
            adata,
            adata_infer_reshape,
            logger,
            save_path=os.path.join(figure_dir, 'Boxplot_infer_cor_count.pdf'),
        )
        logger.info("Running mean_cor_box (inference only, count) plot DONE!")
        mean_cor_box(
            adata_norml,
            adata_infer_reshape,
            logger,
            save_path=os.path.join(figure_dir, 'Boxplot_infer_cor_norml.pdf'),
        )
        logger.info("Running mean_cor_box (inference only, normalized) plot DONE!")
        cor_hist(
            adata,
            adata_infer_spot.to_df(),
            fig_size=(5, 4),
            trans=False,
            format='svg',
            save_path=os.path.join(figure_dir, 'Hist_infer_cor_count.svg'),
        )
        logger.info("Running cor_hist (inference only) plot DONE!")

        adata_smooth = impute_adata(
            adata,
            adata_infer,
            C2,
            gene_hv,
            dataset_class=args.dataset_class,
            weight_exponent=2,
        )
        adata_imput, data_impt = weight_adata(
            adata_infer,
            adata_smooth,
            gene_hv,
            w=args.weight_w,
            do_scale=args.do_scale,
        )
        _, data_impt_reshape = reshape_latent_image(
            torch.tensor(data_impt), dataset_class=args.dataset_class
        )
        adata_imput_spot = reshape2adata(adata, data_impt_reshape, gene_hv)
        print(" **** data_impt shape:", data_impt.shape)
        print(" **** data_impt_reshape shape:", data_impt_reshape.shape)
        print(" **** adata_imput shape:", adata_imput.shape)
        print(" **** adata_imput_spot shape:", adata_imput_spot.shape)

        first_obs_first_var_expr(
            reconstructed_matrix_reshaped_tensor,
            adata,
            gene_hv,
            patch_size=patch_size,
            dataset_class=args.dataset_class,
            save_path=os.path.join(figure_dir, '1st_spot_1st_gene.pdf'),
        )
        logger.info("Running first_obs_first_var_expr plot DONE!")
        gene_expr_compare(
            adata,
            args.gene_selected,
            data_impt_reshape,
            gene_hv,
            s=50,
            save_path=os.path.join(figure_dir, str(args.gene_selected) + '_pred_gene_expr.pdf'),
        )
        logger.info("Running gene_expr_compare plot DONE!")
        sele_gene_cor(
            adata,
            data_impt_reshape,
            gene_hv,
            gene=args.gene_selected,
            ylabel='FineST Expression',
            title=str(args.gene_selected) + ' expression',
            size=5,
            save_path=os.path.join(figure_dir, str(args.gene_selected) + '_gene_corr.pdf'),
        )
        logger.info("Running sele_gene_cor plot DONE!")

        logger.info("Running Gene Correlation task...")
        pearson_cor_gene, spearman_cor_gene, cosine_sim_gene = mean_cor(
            adata, data_impt_reshape, 'reconf2', sample='gene'
        )
        logger.info(
            "Pearson, Spearman, Cosine corr_gene: [{}: {}: {}]".format(
                pearson_cor_gene, spearman_cor_gene, cosine_sim_gene
            )
        )
        logger.info("Running Gene Correlation task DONE!")
        mean_cor_box(
            adata,
            data_impt_reshape,
            logger,
            save_path=os.path.join(figure_dir, 'Box_spot_gene_corr.pdf'),
        )


        #########################################################
        ## 2026.08.05 add the save_data_path function for the API function  
        ######################################################### 
        ## Default SaveData outputs (aligned with tutorial notebook)
        save_dir = os.path.join(args.system_path, args.save_data_path)
        os.makedirs(save_dir, exist_ok=True)
        # count / normalized ST (also written in load_and_process_data when save_data_path is set)
        adata_count.write_h5ad(os.path.join(save_dir, 'adata_count.h5ad'))
        adata_norml.write_h5ad(os.path.join(save_dir, 'adata_norml.h5ad'))
        logger.info(f"Saved adata_count.h5ad and adata_norml.h5ad to {save_dir}")
        # inferred within-spot
        adata_infer.write_h5ad(os.path.join(save_dir, 'adata_infer.h5ad'))
        adata_infer_spot.write_h5ad(os.path.join(save_dir, 'adata_infer_spot.h5ad'))
        logger.info(f"Saved adata_infer.h5ad and adata_infer_spot.h5ad to {save_dir}")
        print(f" **** adata_infer shape: {adata_infer.shape}")
        print(f" **** adata_infer_spot shape: {adata_infer_spot.shape}")
        # imputed / weighted within-spot
        adata_imput.write_h5ad(os.path.join(save_dir, 'adata_imput.h5ad'))
        adata_imput_spot.write_h5ad(os.path.join(save_dir, 'adata_imput_spot.h5ad'))
        logger.info(f"Saved adata_imput.h5ad and adata_imput_spot.h5ad to {save_dir}")
        print(f" **** adata_imput shape: {adata_imput.shape}")
        print(f" **** adata_imput_spot shape: {adata_imput_spot.shape}")
        #########################################################
        
    finally:
        if 'tee' in locals():
            sys.stdout = tee.terminal
            sys.stderr = tee.terminal
            tee.close()
            print(f"\nLog file saved to: {log_file_path}")


def step1_FineST_train_infer(
    system_path,
    dataset_class,
    gene_selected,
    image_embed_path,
    visium_path,
    parame_path,
    LRgene_path='LR_genes',
    species='human',
    hist_model='HIPT',
    image_class=None,
    patch_size=None,
    weight_path='weights',
    spatial_pos_path=None,
    reduced_mtx_path=None,
    figure_save_path=None,
    weight_save_path=None,
    save_data_path=None,
    weight_w=0.5,
    do_scale=False,
):
    """
    Train FineST model and perform gene expression inference.

    This function provides a programmatic interface to train FineST models on
    spot-resolved ST data and infer super-resolved gene expression from image features.
    It supports both HIPT and Virchow2 image feature extraction methods,
    and can work with Visium16, Visium64, or VisiumHD datasets.

    Parameters
    ----------
    system_path : str
        Base system path for data and weights. All other paths are relative to this path.
    dataset_class : str
        Dataset class type. Must be one of:
        - 'Visium16': 16 sub-spots per spot (typically used with HIPT, patch_size=64)
        - 'Visium64': 64 sub-spots per spot (typically used with Virchow2, patch_size=112)
        - 'VisiumHD': 4 sub-spots per spot (for Visium HD data)
    gene_selected : str
        Marker gene name for visualization (e.g., 'CD70', 'CD27').
    image_embed_path : str
        Path to image embedding directory (relative to system_path).
    visium_path : str
        Path to Visium tissue positions file (CSV format, relative to system_path).
    parame_path : str
        Path to parameter JSON file (relative to system_path).
    LRgene_path : str, optional
        LR gene source: ``'LR_genes'`` (default bundled list), ``'HV_genes'``,
        ``'LR_HV_genes'``, or path to a CSV file.
    species : str, optional
        Species for bundled LR gene list: ``'human'`` or ``'mouse'``.
        Default: ``'human'``
    hist_model : str, optional
        Histology foundation model: ``'HIPT'`` or ``'Virchow2'``.
        Must match Step 0 ``fst.image_feature_extraction(hist_model=...)``.
        Default: ``'HIPT'``
    image_class : str, optional
        Deprecated alias for ``hist_model``.
    patch_size : int, optional
        Patch size used in Step 0. If None, inferred from ``dataset_class`` / ``hist_model``.
    weight_path : str, optional
        Directory name for saving trained model weights (relative to system_path).
        Kept for backward compatibility; weights are written under Figures/.
        Default: 'weights'
    spatial_pos_path : str, optional
        Path to save ordered spatial positions. Default: ``<data_root>/OrderData/position_order.csv``
    reduced_mtx_path : str, optional
        Path to save ordered expression matrix. Default: ``<data_root>/OrderData/matrix_order.npy``
    figure_save_path : str, optional
        Directory for saving figures. Default: ``<data_root>/Figures/``
    weight_save_path : str, optional
        Path to pre-trained model weights directory (relative to system_path).
        If provided, training will be skipped and only inference will be performed.
        If None, model will be trained from scratch.
        Default: None
    save_data_path : str, optional
        Directory to save h5ad outputs. Default: ``<data_root>/SaveData/``
    weight_w : float, optional
        Weight for combining inferred and imputed expression in ``weight_adata``.
        Default: 0.5
    do_scale : bool, optional
        Whether to scale expression before combining in ``weight_adata``.
        Default: False

    Returns
    -------
    None
        The function saves outputs to disk:
        - Trained model weights (if weight_save_path is None)
        - Inferred gene expression data
        - Visualization figures
        - Log files
    """
    _ = weight_path  # reserved / backward-compatible
    hist_model = resolve_hist_model(hist_model, image_class, default='HIPT')
    if image_class is not None:
        warnings.warn(
            'image_class is deprecated and will be removed; use hist_model instead.',
            DeprecationWarning,
            stacklevel=2,
        )
    if patch_size is None:
        patch_size = default_patch_size(dataset_class, hist_model)
    ## Create argparse.Namespace object from function parameters
    args = argparse.Namespace(
        system_path=system_path,
        LRgene_path=LRgene_path,
        species=species,
        dataset_class=dataset_class,
        hist_model=hist_model,
        gene_selected=gene_selected,
        image_embed_path=image_embed_path,
        visium_path=visium_path,
        parame_path=parame_path,
        spatial_pos_path=spatial_pos_path,
        reduced_mtx_path=reduced_mtx_path,
        figure_save_path=figure_save_path,
        weight_save_path=weight_save_path,
        patch_size=patch_size,
        save_data_path=save_data_path,
        weight_w=weight_w,
        do_scale=do_scale,
    )
    main(args)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='FineST Step1: train on within-spot data and run inference.'
    )
    parser.add_argument('--system_path', type=str, required=True, help='System path for data and weights')
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Dataset root (e.g. FineST_tutorial_data); fills default paths when set',
    )
    parser.add_argument(
        '--LRgene_path',
        type=str,
        default='LR_genes',
        help="LR gene source: 'LR_genes', 'HV_genes', 'LR_HV_genes', or CSV path",
    )
    parser.add_argument('--species', type=str, default='human', help="'human' or 'mouse'")
    parser.add_argument(
        '--dataset_class',
        type=str,
        required=True,
        help='Visium16, Visium64, or VisiumHD',
    )
    parser.add_argument(
        '--hist_model',
        type=str,
        default='HIPT',
        help='HIPT or Virchow2 (must match Step 0; default: HIPT)',
    )
    parser.add_argument(
        '--image_class',
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--gene_selected', type=str, required=True, help='Marker gene for visualization')
    parser.add_argument('--image_embed_path', type=str, required=True, help='Path to image embeddings')
    parser.add_argument('--visium_path', type=str, required=True, help='Path to Visium positions')
    parser.add_argument('--parame_path', type=str, required=True, help='Path to parameter JSON')
    parser.add_argument(
        '--spatial_pos_path',
        type=str,
        default=None,
        help='Default: <data_root>/OrderData/position_order.csv',
    )
    parser.add_argument(
        '--reduced_mtx_path',
        type=str,
        default=None,
        help='Default: <data_root>/OrderData/matrix_order.npy',
    )
    parser.add_argument(
        '--figure_save_path',
        type=str,
        default=None,
        help='Default: <data_root>/Figures/',
    )
    parser.add_argument(
        '--weight_save_path',
        type=str,
        default=None,
        help='Pretrained weights dir; skip training when set',
    )
    parser.add_argument('--patch_size', type=int, default=None, help='Patch size from Step0')
    parser.add_argument(
        '--save_data_path',
        type=str,
        default=None,
        help='Default: <data_root>/SaveData/',
    )
    parser.add_argument('--weight_w', type=float, default=0.5, help='weight_adata mix weight')
    parser.add_argument('--do_scale', type=str2bool, default=False, help='Scale in weight_adata')
    args = parser.parse_args(argv)
    args = _normalize_hist_model_args(args)
    if args.patch_size is None:
        args.patch_size = default_patch_size(args.dataset_class, args.hist_model)
    return args


if __name__ == '__main__':
    main(parse_args())


###############################
# Usage
###############################
# import FineST as fst
#
# ## Method 1: Training and inference（Virchow2）
# fst.step1_FineST_train_infer(
#     system_path='/home/lingyu/ssd/Python/FineST_submit/FineST/',
#     dataset_class='Visium64',
#     hist_model='Virchow2',
#     gene_selected='CD70',
#     image_embed_path='FineST_tutorial_data/ImgEmbeddings/Virchow2/pth_112_14',
#     visium_path='FineST_tutorial_data/spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_virchow2.json',
# )
#
# ## Method 2: Training and inference（HIPT）
# fst.step1_FineST_train_infer(
#     system_path='/home/lingyu/ssd/Python/FineST_submit/FineST/',
#     dataset_class='Visium16',
#     hist_model='HIPT',
#     gene_selected='CD70',
#     image_embed_path='FineST_tutorial_data/ImgEmbeddings/HIPT/pth_64_16',
#     visium_path='FineST_tutorial_data/spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_HIPT.json',
# )
#
# ## Method 3: Only inference（with pre-trained weights）
# fst.step1_FineST_train_infer(
#     system_path='/home/lingyu/ssd/Python/FineST_submit/FineST/',
#     LRgene_path='LRgene/LRgene_CellChatDB_baseline.csv',
#     dataset_class='Visium64',
#     hist_model='Virchow2',
#     gene_selected='CD70',
#     image_embed_path='ImgEmbeddings/Virchow2/pth_112_14',
#     visium_path='spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_virchow2.json',
#     patch_size=112,
#     weight_save_path='weights/20250621001835815284',
# )
#
# ## CLI
# # python -m FineST.step1_FineST_train_infer \
# #     --system_path './FineST/' \
# #     --dataset_class Visium16 \
# #     --hist_model HIPT \
# #     --gene_selected CD70 \
# #     --visium_path FineST_tutorial_data/spatial/tissue_positions_list.csv \
# #     --image_embed_path FineST_tutorial_data/ImgEmbeddings/HIPT/pth_64_16 \
# #     --parame_path parameter/parameters_NPC_HIPT.json
