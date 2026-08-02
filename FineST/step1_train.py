"""
Step 1: FineST Model Training and Inference

This module provides a high-level interface for training FineST models and performing
gene expression inference from image features. It wraps the functionality from
demo/Step1_FineST_train_infer.py to enable programmatic access.
"""

import os
import sys
import argparse
from pathlib import Path

def _get_demo_dir():
    """Get the demo directory path."""
    _current_dir = Path(__file__).parent  
    _demo_dir = _current_dir / 'demo'  
    if not _demo_dir.exists():
        _demo_dir = _current_dir.parent / 'demo'  
    return _demo_dir

def _import_main():
    """Lazy import of main function from Step1_FineST_train_infer."""
    _demo_dir = _get_demo_dir()
    if str(_demo_dir) not in sys.path:
        sys.path.insert(0, str(_demo_dir))
    from Step1_FineST_train_infer import main
    return main

def Step1_FineST_train_infer(
    system_path,
    dataset_class,
    gene_selected,
    image_embed_path,
    visium_path,
    parame_path,
    LRgene_path='LR_genes',
    species='human',
    image_class='Virchow2',
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
    image_class : str, optional
        Image feature extraction method. Must be 'HIPT' or 'Virchow2'.
        Should match the method used in Image_feature_extraction.py.
        Default: 'Virchow2'
    patch_size : int, optional
        Patch size used in Step 0. If None, inferred from ``dataset_class`` / ``image_class``.
    weight_path : str, optional
        Directory name for saving trained model weights (relative to system_path).
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
    ## Create argparse.Namespace object from function parameters
    args = argparse.Namespace(
        system_path=system_path,
        LRgene_path=LRgene_path,
        species=species,
        dataset_class=dataset_class,
        image_class=image_class,
        gene_selected=gene_selected,
        image_embed_path=image_embed_path,
        visium_path=visium_path,
        weight_path=weight_path,
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
    
    ## Call the main function from Step1_FineST_train_infer.py 
    _main = _import_main()
    _main(args)


###############################
# Usage
###############################
# import FineST as fst

# ## Method 1: Training and inference（Virchow2）
# fst.Step1_FineST_train_infer(
#     system_path='/home/lingyu/ssd/Python/FineST_submit/FineST/',
#     dataset_class='Visium64',
#     image_class='Virchow2',
#     gene_selected='CD70',
#     image_embed_path='FineST_tutorial_data/ImgEmbeddings/pth_112_14',
#     visium_path='FineST_tutorial_data/spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_virchow2.json',
# )

# ## Method 2: Training and inference（HIPT）
# fst.Step1_FineST_train_infer(
#     system_path='/home/lingyu/ssd/Python/FineST_submit/FineST/',
#     dataset_class='Visium16',
#     image_class='HIPT',
#     gene_selected='CD70',
#     image_embed_path='FineST_tutorial_data/ImgEmbeddings/pth_64_16',
#     visium_path='FineST_tutorial_data/spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_HIPT.json',
# )

# ## Method 3: Only inference（with pre-trained weights）
# fst.Step1_FineST_train_infer(
#     system_path='/home/lingyu/ssd/Python/FineST_submit/FineST/',
#     LRgene_path='LRgene/LRgene_CellChatDB_baseline.csv',
#     dataset_class='Visium64',
#     image_class='Virchow2',
#     gene_selected='CD70',
#     image_embed_path='ImgEmbeddings/pth_112_14',
#     visium_path='spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_virchow2.json',
#     patch_size=112,
#     weight_save_path='weights/20250621001835815284'  
# )