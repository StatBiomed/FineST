"""
Step 1: FineST Model Training and Inference

This module provides a high-level interface for training FineST models and performing
gene expression inference from image features. It wraps the functionality from
demo/Step1_FineST_train_infer.py to enable programmatic access.

Example:
    >>> import FineST as fst
    >>> 
    >>> # Train and infer
    >>> fst.Step1_FineST_train_infer(
    ...     system_path='/path/to/data/',
    ...     LRgene_path='path/to/LRgenes.csv',
    ...     dataset_class='Visium64',
    ...     image_class='Virchow2',
    ...     gene_selected='CD70',
    ...     image_embed_path='ImgEmbeddings/pth_112_14',
    ...     visium_path='spatial/tissue_positions_list.csv',
    ...     parame_path='parameter/parameters_NPC_virchow2.json',
    ...     patch_size=112
    ... )
"""

import os
import sys
import argparse
from pathlib import Path

# Lazy import to avoid circular import issues
# Step1_FineST_train_infer.py imports FineST, which would cause a circular import
# if we import it at module level. Instead, we import it only when needed.
def _get_demo_dir():
    """Get the demo directory path."""
    _current_dir = Path(__file__).parent  # FineST/FineST/
    # Try FineST/FineST/demo/ first (correct location)
    _demo_dir = _current_dir / 'demo'  # FineST/FineST/demo/
    if not _demo_dir.exists():
        # Fallback: try FineST/demo/ (if demo is at package root)
        _demo_dir = _current_dir.parent / 'demo'  # FineST/demo/
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
    LRgene_path,
    dataset_class,
    gene_selected,
    image_embed_path,
    visium_path,
    parame_path,
    image_class='Virchow2',
    patch_size=112,
    weight_path='weights',
    spatial_pos_path='spatial_pos.csv',
    reduced_mtx_path='reduced_mtx.npy',
    figure_save_path='figures',
    weight_save_path=None
):
    """
    Train FineST model and perform gene expression inference.
    
    This function provides a programmatic interface to train FineST models on
    spatial transcriptomics data and infer super-resolved gene expression from
    image features. It supports both HIPT and Virchow2 image feature extraction
    methods, and can work with Visium16, Visium64, or VisiumHD datasets.
    
    Parameters
    ----------
    system_path : str
        Base system path for data and weights. All other paths are relative to this
        unless specified as absolute paths.
    LRgene_path : str
        Path to ligand-receptor gene list file (CSV format).
        Can be relative to system_path or absolute path.
    dataset_class : str
        Dataset class type. Must be one of:
        - 'Visium16': 16 sub-spots per spot (typically used with HIPT, patch_size=64)
        - 'Visium64': 64 sub-spots per spot (typically used with Virchow2, patch_size=112)
        - 'VisiumHD': 4 sub-spots per spot (for Visium HD data)
    gene_selected : str
        Marker gene name for visualization (e.g., 'CD70', 'CD27').
    image_embed_path : str
        Path to image embedding directory (relative to system_path).
        Should match the output from Image_feature_extraction.py:
        - For HIPT + patch_size=64: typically 'ImgEmbeddings/pth_64_16'
        - For Virchow2 + patch_size=112: typically 'ImgEmbeddings/pth_112_14'
    visium_path : str
        Path to Visium tissue positions file (CSV format, relative to system_path).
        Typically 'spatial/tissue_positions_list.csv'.
    parame_path : str
        Path to parameter JSON file (relative to system_path).
        Should contain model hyperparameters:
        - For HIPT: typically 'parameter/parameters_NPC_HIPT.json'
        - For Virchow2: typically 'parameter/parameters_NPC_virchow2.json'
    image_class : str, optional
        Image feature extraction method. Must be 'HIPT' or 'Virchow2'.
        Should match the method used in Image_feature_extraction.py.
        Default: 'Virchow2'
    patch_size : int, optional
        Patch size used for image feature extraction. Must match the patch_size
        used in Image_feature_extraction.py.
        - For HIPT + Visium16: typically 64
        - For Virchow2 + Visium64: typically 112
        Default: 112
    weight_path : str, optional
        Directory name for saving trained model weights (relative to system_path).
        Default: 'weights'
    spatial_pos_path : str, optional
        Path to save/load ordered spatial positions CSV file (relative to system_path).
        This file will be generated if it doesn't exist.
        Default: 'spatial_pos.csv'
    reduced_mtx_path : str, optional
        Path to save/load ordered gene expression matrix (relative to system_path).
        This file will be generated if it doesn't exist.
        Default: 'reduced_mtx.npy'
    figure_save_path : str, optional
        Directory for saving output figures (relative to system_path).
        Default: 'figures'
    weight_save_path : str, optional
        Path to pre-trained model weights directory (relative to system_path).
        If provided, training will be skipped and only inference will be performed.
        If None, model will be trained from scratch.
        Default: None
    
    Returns
    -------
    None
        The function saves outputs to disk:
        - Trained model weights (if weight_save_path is None)
        - Inferred gene expression data
        - Visualization figures
        - Log files
    
    Examples
    --------
    >>> # Example 1: Train and infer with Virchow2
    >>> import FineST as fst
    >>> fst.Step1_FineST_train_infer(
    ...     system_path='/path/to/FineST_tutorial_data/',
    ...     LRgene_path='LRgene/LRgene_CellChatDB_baseline.csv',
    ...     dataset_class='Visium64',
    ...     image_class='Virchow2',
    ...     gene_selected='CD70',
    ...     image_embed_path='ImgEmbeddings/pth_112_14',
    ...     visium_path='spatial/tissue_positions_list.csv',
    ...     parame_path='parameter/parameters_NPC_virchow2.json',
    ...     patch_size=112
    ... )
    
    >>> # Example 2: Train and infer with HIPT
    >>> fst.Step1_FineST_train_infer(
    ...     system_path='/path/to/FineST_tutorial_data/',
    ...     LRgene_path='LRgene/LRgene_CellChatDB_baseline.csv',
    ...     dataset_class='Visium16',
    ...     image_class='HIPT',
    ...     gene_selected='CD70',
    ...     image_embed_path='ImgEmbeddings/pth_64_16',
    ...     visium_path='spatial/tissue_positions_list.csv',
    ...     parame_path='parameter/parameters_NPC_HIPT.json',
    ...     patch_size=64
    ... )
    
    >>> # Example 3: Only infer (using pre-trained weights)
    >>> fst.Step1_FineST_train_infer(
    ...     system_path='/path/to/FineST_tutorial_data/',
    ...     LRgene_path='LRgene/LRgene_CellChatDB_baseline.csv',
    ...     dataset_class='Visium64',
    ...     image_class='Virchow2',
    ...     gene_selected='CD70',
    ...     image_embed_path='ImgEmbeddings/pth_112_14',
    ...     visium_path='spatial/tissue_positions_list.csv',
    ...     parame_path='parameter/parameters_NPC_virchow2.json',
    ...     patch_size=112,
    ...     weight_save_path='weights/20250621001835815284'
    ... )
    
    Notes
    -----
    - The function automatically handles data preprocessing, model training, and inference
    - Output files are saved in directories specified by the path parameters
    - Training can take significant time depending on dataset size and hardware
    - Ensure all required input files exist before calling this function
    - The function uses the NPC dataset by default (can be modified in the source code)
    """
    # Create argparse.Namespace object from function parameters
    args = argparse.Namespace(
        system_path=system_path,
        LRgene_path=LRgene_path,
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
        patch_size=patch_size
    )
    
    # Call the main function from Step1_FineST_train_infer.py (lazy import to avoid circular import)
    _main = _import_main()
    _main(args)


###############################
# Usage
###############################
# import FineST as fst

# ## Method 1: Training and inference（Virchow2）
# fst.Step1_FineST_train_infer(
#     system_path='/path/to/data/',
#     LRgene_path='LRgene/LRgene_CellChatDB_baseline.csv',
#     dataset_class='Visium64',
#     image_class='Virchow2',
#     gene_selected='CD70',
#     image_embed_path='ImgEmbeddings/pth_112_14',
#     visium_path='spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_virchow2.json',
#     patch_size=112
# )

# ## Method 2: Training and inference（HIPT）
# fst.Step1_FineST_train_infer(
#     system_path='/path/to/data/',
#     LRgene_path='LRgene/LRgene_CellChatDB_baseline.csv',
#     dataset_class='Visium16',
#     image_class='HIPT',
#     gene_selected='CD70',
#     image_embed_path='ImgEmbeddings/pth_64_16',
#     visium_path='spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_HIPT.json',
#     patch_size=64
# )

# ## Method 3: Only inference（with pre-trained weights）
# fst.Step1_FineST_train_infer(
#     system_path='/path/to/data/',
#     LRgene_path='LRgene/LRgene_CellChatDB_baseline.csv',
#     dataset_class='Visium64',
#     image_class='Virchow2',
#     gene_selected='CD70',
#     image_embed_path='ImgEmbeddings/pth_112_14',
#     visium_path='spatial/tissue_positions_list.csv',
#     parame_path='parameter/parameters_NPC_virchow2.json',
#     patch_size=112,
#     weight_save_path='weights/20250621001835815284'  # 跳过训练
# )