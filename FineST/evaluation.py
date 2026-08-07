import numpy as np
import  logging
logging.getLogger().setLevel(logging.INFO)
from .utils import *
from .loadData import *
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import MinMaxScaler
import torch


#############################
# 2025.02.12 add ssim index
#############################
def vector2matrix(locs, cnts, shape):
    """
    Convert vector to matrix.
        locs : list. Locations.
        cnts : list. Counts.
        shape : tuple. Shape.
    Returns:
        x_reconstructed : numpy array. Reconstructed matrix.
    """
    x_reconstructed = np.full(shape, np.nan)
    for loc, cnt in zip(locs, cnts):
        x_reconstructed[loc[0], loc[1]] = cnt
    return x_reconstructed

def compute_ssim(x, x_reconstructed):
    """
    Compute SSIM.
        x : numpy array. Original matrix.
        x_reconstructed : numpy array. Reconstructed matrix.
    Returns:
        ssim_index : float. SSIM index.
    """
    x = np.nan_to_num(x)
    x_reconstructed = np.nan_to_num(x_reconstructed)
    ssim_index = ssim(x, x_reconstructed, data_range=x.max() - x.min())
    return ssim_index

def compute_ssim_scale(x, x_reconstructed):
    """
    Compute SSIM scale.
        x : numpy array. Original matrix.
        x_reconstructed : numpy array. Reconstructed matrix.
    Returns:
        ssim_index : float. SSIM index.
    """
    scaler = MinMaxScaler()    # Initialize MinMaxScaler to scale data to [0, 1].
    # Replace NaN values with zero and scale to [0, 1] range
    x = np.nan_to_num(x)
    x_scaled = scaler.fit_transform(x.reshape(-1, 1)).reshape(x.shape)
    x_reconstructed = np.nan_to_num(x_reconstructed)
    x_reconstructed_scaled = scaler.fit_transform(x_reconstructed.reshape(-1, 1)).reshape(x_reconstructed.shape)
    ssim_index = ssim(x_scaled, x_reconstructed_scaled, data_range=1)    # set data_range=1, if data has been scaled to [0, 1]
    return ssim_index   

#############################
# 2024.11.16 align 8um adata
#############################
def align_adata_fst2hd(adata_impt, adata_8um):
    """
    Args:
        adata_impt (anndata.AnnData): The dataset to be aligned.
        adata_8um (anndata.AnnData): The reference dataset.
    Returns:
        adata_impt_align (anndata.AnnData): The aligned dataset.
        shared_finest_df (pandas.DataFrame): DataFrame  of 'adata_impt_align'.
        shared_visium_df (pandas.DataFrame): DataFrame  of 'adata_8um'.
    """
    tree = cKDTree(adata_impt.obsm['spatial'])
    _, closest_points_indices = tree.query(adata_8um.obsm['spatial'], k=1)
    
    adata_impt_align = adata_impt[closest_points_indices]
    adata_impt_align.obs_names = adata_8um.obs_names
    
    shared_finest_df = adata_impt_align.to_df()
    shared_visium_df = adata_8um.to_df()
    
    return adata_impt_align, shared_finest_df, shared_visium_df


#############################
# 2024.11.08 more fast
#############################
def calculate_correlation(matrix_tensor_test_np, reconstructed_matrix_test_np, 
                          method="pearson", sample="spot"):
    """
    Calculate correlation.
        matrix_tensor_test_np : numpy array. Matrix tensor test.
        reconstructed_matrix_test_np : numpy array. Reconstructed matrix test.
        method : str. Method.
        sample : str. Sample.
    Returns:
        correlation_coefficients : list. Correlation coefficients.
    """
    correlation_coefficients = []
    
    if sample == "spot":
        loop_range = matrix_tensor_test_np.shape[0]
        data_index = 0
    elif sample == "gene":
        loop_range = matrix_tensor_test_np.shape[1]
        data_index = 1
    else:
        raise ValueError("Invalid sample type, choose either 'spot' or 'gene'")

    for i in range(loop_range):
        x = matrix_tensor_test_np[i] if data_index==0 else matrix_tensor_test_np[:,i]
        y = reconstructed_matrix_test_np[i] if data_index==0 else reconstructed_matrix_test_np[:,i]
        if method == "pearson":
            corr_matrix = np.corrcoef(x, y)
            corr = corr_matrix[0, 1]
        elif method == "spearman":
            corr, _ = spearmanr(x, y)    # np.corrcoef does not support Spearman correlation
        else:
            raise ValueError("Invalid method, choose either 'pearson' or 'spearman'")
        corr = np.nanmean(corr) if not np.isnan(corr).all() else 0
        correlation_coefficients.append(corr)

    return correlation_coefficients


def mean_cor(adata, data_impt_reshape, label, sample="gene"):
    """
    Calculate mean correlation.
        adata : AnnData. Input adata.
        data_impt_reshape : numpy array. Data imputed reshape.
        label : str. Label.
        sample : str. Sample.
    Returns:
        mean_pearson_corr : float. Mean Pearson correlation coefficient.
    """
    if isinstance(adata.X, np.ndarray):
        matrix1 = np.array(adata.X)
    else:
        matrix1 = np.array(adata.X.todense())
    matrix2 = np.array(data_impt_reshape)
    print("matrix1: ", matrix1.shape)
    print("matrix2: ", matrix2.shape)

    mean_pearson_corr = calculate_correlation_infer(matrix1, matrix2, method="pearson", sample=sample)
    print(f"Mean Pearson correlation coefficient--{label}: {mean_pearson_corr:.4f}")
    mean_spearman_corr = calculate_correlation_infer(matrix1, matrix2, method="spearman", sample=sample)
    print(f"Mean Spearman correlation coefficient--{label}: {mean_spearman_corr:.4f}")
    cosine_sim = calculate_cosine_similarity_col(matrix1, matrix2)
    cosine_sim_per_sample = np.diag(cosine_sim)
    mean_cosine_similarity = np.mean(cosine_sim_per_sample)   
    print(f"Mean cosine similarity--{label}: {mean_cosine_similarity:.4f}")
    
    return mean_pearson_corr, mean_spearman_corr, mean_cosine_similarity


#############################
# Inference Correlation  
#############################
def calculate_correlation_infer(matrix_tensor_test_np, reconstructed_matrix_test_np, 
                                method="pearson", sample="spot"):
    """
    Calculate correlation.
        matrix_tensor_test_np : numpy array. Matrix tensor test.
        reconstructed_matrix_test_np : numpy array. Reconstructed matrix test.
        method : str. Method.
        sample : str. Sample.
    Returns:
        mean_corr : float. Mean correlation.
    """
    # Check for NaN values in the input matrices
    if np.isnan(matrix_tensor_test_np).any() or np.isnan(reconstructed_matrix_test_np).any():
        print("Warning: Input matrices contain NaN. Please handle them before calculating.")
        return np.nan

    correlation_coefficients = []

    if sample == "spot":
        loop_range = matrix_tensor_test_np.shape[0]
        data_index = 0
    elif sample == "gene":
        loop_range = matrix_tensor_test_np.shape[1]
        data_index = 1
    else:
        raise ValueError("Invalid sample type, choose either 'spot' or 'gene'")

    for i in range(loop_range):
        # Check if the row/column is constant in both input matrices
        matrix_slice = (
            matrix_tensor_test_np[i] if data_index == 0 
            else matrix_tensor_test_np[:, i]
        )
        reconstructed_slice = (
            reconstructed_matrix_test_np[i] if data_index == 0 
            else reconstructed_matrix_test_np[:, i]
        )
        if np.std(matrix_slice) == 0 or np.std(reconstructed_slice) == 0:
            continue

        if method == "pearson":
            corr = np.corrcoef(matrix_tensor_test_np[i] if data_index==0 else matrix_tensor_test_np[:,i], 
                               reconstructed_matrix_test_np[i] if data_index==0 else reconstructed_matrix_test_np[:,i])[0,1]
        elif method == "spearman":
            corr, _ = spearmanr(matrix_tensor_test_np[i] if data_index==0 else matrix_tensor_test_np[:,i], 
                                reconstructed_matrix_test_np[i] if data_index==0 else reconstructed_matrix_test_np[:,i])
        else:
            raise ValueError("Invalid method, choose either 'pearson' or 'spearman'")
        correlation_coefficients.append(corr)
        
    mean_corr = np.nanmean(correlation_coefficients) if sample == "gene" else np.mean(correlation_coefficients)
    return mean_corr


#############################
# cosine_similarity
#############################
def calculate_cosine_similarity_row(rep_query_adata, rep_ref_adata_image_reshape):
    """
    Calculate cosine similarity row.
        rep_query_adata : torch.Tensor. Query adata.
        rep_ref_adata_image_reshape : torch.Tensor. Reference adata image reshape.
    Returns:
        cosine_sim : numpy array. Cosine similarity.
    """
    if isinstance(rep_query_adata, torch.Tensor):
        rep_query_adata = rep_query_adata.numpy()
    if isinstance(rep_ref_adata_image_reshape, torch.Tensor):
        rep_ref_adata_image_reshape = rep_ref_adata_image_reshape.numpy()
    cosine_sim = cosine_similarity(rep_query_adata, rep_ref_adata_image_reshape)
    return cosine_sim

def calculate_cosine_similarity_col(rep_query_adata, rep_ref_adata_image_reshape):
    if isinstance(rep_query_adata, torch.Tensor):
        rep_query_adata = rep_query_adata.numpy()
    if isinstance(rep_ref_adata_image_reshape, torch.Tensor):
        rep_ref_adata_image_reshape = rep_ref_adata_image_reshape.numpy()
    rep_query_adata_T = rep_query_adata.T
    rep_ref_adata_image_reshape_T = rep_ref_adata_image_reshape.T
    cosine_sim = cosine_similarity(rep_query_adata_T, rep_ref_adata_image_reshape_T)
    return cosine_sim

def compute_corr(expression_gt, matched_spot_expression_pred, top_k=50, qc_idx=None):
    """
    Compute correlation.
        expression_gt : numpy array. Expression ground truth.
        matched_spot_expression_pred : numpy array. Matched spot expression predicted.
        top_k : int. Top k.
        qc_idx : list. QC index.
    Returns:
        corr : float. Correlation.
    """
    ## cells are in columns, genes are in rows
    if qc_idx is not None:
        expression_gt = expression_gt[:, qc_idx]
        matched_spot_expression_pred = matched_spot_expression_pred[:, qc_idx]
    mean = np.mean(expression_gt, axis=1)
    top_genes_idx = np.argpartition(mean, -top_k)[-top_k:]
    corr = [np.corrcoef(expression_gt[i, :], matched_spot_expression_pred[i, :])[0, 1] for i in top_genes_idx]
    return np.mean(corr)


#############################
# 2026.08.07 Tutorial evaluation (Visium / Visium HD)
#############################
# Evaluate within-spot infer / impute against measured expression.
#
# * evaluate_visium()   — NPC notebook Section 3
# * evaluate_visiumhd() — CRC16 notebook Sections 3 + 3.6 + 4
#
# CLI::
#
#     python -m FineST.evaluation --platform visium
#     python -m FineST.evaluation --platform visiumhd
#
# API::
#
#     import FineST as fst
#     fst.evaluate_visium(data_root='FineST_tutorial_data')
#     fst.evaluate_visiumhd(data_root='FineST_tutorial_data_VisiumHD')
#############################


def _resolve_path(system_path, path):
    import os
    return path if os.path.isabs(path) else os.path.join(system_path, path)


def _resolve_h5ad(system_path, path):
    """Resolve h5ad path; accept legacy notebook names like ``SaveData/_adata_imput.h5ad``."""
    import os

    full = _resolve_path(system_path, path)
    if os.path.isfile(full):
        return full
    directory, basename = os.path.split(full)
    if basename.startswith('adata_'):
        legacy = os.path.join(directory, f'_{basename}')
        if os.path.isfile(legacy):
            return legacy
    return full


def _spot_expression_matrix(adata_spot):
    if hasattr(adata_spot.X, 'todense'):
        return np.array(adata_spot.X.todense())
    return np.array(adata_spot.X)


def _ensure_spatial(adata, reference_adata):
    """Attach ``obsm['spatial']`` and copy ``uns['spatial']`` from a reference."""
    if 'spatial' not in adata.obsm and 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        adata.obsm['spatial'] = np.stack([adata.obs['x'].values, adata.obs['y'].values], axis=-1)
    if hasattr(reference_adata, 'uns') and 'spatial' in reference_adata.uns:
        adata.uns['spatial'] = reference_adata.uns['spatial']
    return adata


def _load_step1_outputs(system_path, presets):
    import os
    import scanpy as sc

    save_dir = _resolve_path(system_path, presets['save_data_path'])
    adata_count = sc.read_h5ad(_resolve_h5ad(system_path, presets['save_adata_count']))
    adata_norml = sc.read_h5ad(_resolve_h5ad(system_path, presets['save_adata_norml']))
    adata_infer_spot = sc.read_h5ad(_resolve_h5ad(system_path, presets['save_adata_infer_spot']))
    adata_imput_spot = sc.read_h5ad(_resolve_h5ad(system_path, presets['save_adata_imput_spot']))
    gene_hv = list(adata_infer_spot.var_names)
    return {
        'save_dir': save_dir,
        'adata_count': adata_count,
        'adata_norml': adata_norml,
        'adata_infer_spot': adata_infer_spot,
        'adata_imput_spot': adata_imput_spot,
        'gene_hv': gene_hv,
    }


def evaluate_vs_input(
    adata_count,
    adata_norml,
    adata_infer_spot,
    adata_imput_spot,
    gene_hv,
    genes,
    fig_dir,
    logger,
    gene_expr_marker='s',
    gene_expr_s=50,
    imput_cor_hist=False,
    prefix='',
):
    """Compare infer / impute spot-level outputs with measured input ``adata``."""
    import os
    from .plottings import cor_hist, gene_expr_compare, mean_cor_box, sele_gene_cor

    os.makedirs(fig_dir, exist_ok=True)
    infer_reshape = _spot_expression_matrix(adata_infer_spot)
    imput_reshape = _spot_expression_matrix(adata_imput_spot)
    tag = f'{prefix}_' if prefix else ''

    for gene in genes:
        gene_expr_compare(
            adata_count,
            gene,
            infer_reshape,
            gene_hv,
            marker=gene_expr_marker,
            s=gene_expr_s,
            save_path=os.path.join(fig_dir, f'{gene}_expr_infer_vs_input{tag}.pdf'),
        )

    mean_cor_box(
        adata_count,
        infer_reshape,
        logger,
        save_path=os.path.join(fig_dir, f'Boxplot_infer_cor_count{tag}.pdf'),
    )
    mean_cor_box(
        adata_norml,
        infer_reshape,
        logger,
        save_path=os.path.join(fig_dir, f'Boxplot_infer_cor_norml{tag}.pdf'),
    )
    cor_hist(
        adata_count,
        adata_infer_spot.to_df(),
        fig_size=(5, 4),
        trans=False,
        format='svg',
        save_path=os.path.join(fig_dir, f'Hist_infer_cor_count{tag}.svg'),
    )

    for gene in genes:
        gene_expr_compare(
            adata_count,
            gene,
            imput_reshape,
            gene_hv,
            marker=gene_expr_marker,
            s=gene_expr_s,
            save_path=os.path.join(fig_dir, f'{gene}_expr_imput_vs_input{tag}.pdf'),
        )

    gene_tag = '_'.join(genes)
    sele_gene_cor(
        adata_count,
        imput_reshape,
        gene_hv,
        gene=genes,
        ylabel='FineST Expression',
        size=5,
        save_path=os.path.join(fig_dir, f'{gene_tag}_cor_imput_vs_input{tag}.pdf'),
    )
    mean_cor_box(
        adata_count,
        imput_reshape,
        logger,
        save_path=os.path.join(fig_dir, f'Boxplot_imput_cor_count{tag}.pdf'),
    )
    if imput_cor_hist:
        cor_hist(
            adata_count,
            adata_imput_spot.to_df(),
            fig_size=(5, 4),
            trans=False,
            format='svg',
            save_path=os.path.join(fig_dir, f'Hist_imput_cor_count{tag}.svg'),
        )


def evaluate_visium(
    system_path='./',
    data_root='FineST_tutorial_data',
    save_data_path=None,
    figure_save_path=None,
    genes=None,
):
    """Evaluate within-spot infer / impute vs measured Visium spots (NPC notebook Section 3)."""
    import os
    from .paths import normalize_data_root, tutorial_path_presets
    from .utils import setup_logger

    presets = tutorial_path_presets(normalize_data_root(data_root))
    save_data_path = save_data_path or presets['save_data_path']
    figure_save_path = figure_save_path or presets['figure_save_path']
    genes = genes or ['CD70', 'CD27']

    presets = {**presets, 'save_data_path': save_data_path}
    fig_dir = _resolve_path(system_path, figure_save_path)
    os.makedirs(fig_dir, exist_ok=True)
    logger = setup_logger(fig_dir)

    artifacts = _load_step1_outputs(system_path, presets)
    logger.info('Loaded Step 1 outputs from %s', artifacts['save_dir'])

    evaluate_vs_input(
        artifacts['adata_count'],
        artifacts['adata_norml'],
        artifacts['adata_infer_spot'],
        artifacts['adata_imput_spot'],
        artifacts['gene_hv'],
        genes,
        fig_dir,
        logger,
    )
    logger.info('Visium evaluation complete. Figures saved to %s', fig_dir)
    return artifacts


def evaluate_visiumhd_vs_8um(
    system_path,
    presets,
    fig_dir,
    logger,
    genes,
    adata_count,
    adata_infer,
    adata_infer_spot,
    adata_imput,
    adata_imput_spot,
    gene_hv,
    save_adata_imput_8um_path=None,
    save_adata_impt_align_8um_path=None,
):
    """Evaluate infer / impute against native Visium HD 8 µm bins (Sections 3.6 + 4)."""
    import os
    import pandas as pd
    from . import datasets
    from .plottings import cor_hist, gene_expr_compare, mean_cor_box, sele_gene_cor
    from .processData import adata_preprocess, reshape2adata

    adata_8um_raw = datasets.CRC08um()
    adata_8um_raw = adata_8um_raw[:, gene_hv]
    adata_8um = adata_preprocess(adata_8um_raw, min_cells=1, normalize=False)
    logger.info('Loaded native 8 µm reference: %s', adata_8um.shape)

    adata_imput = _ensure_spatial(adata_imput, adata_8um)
    adata_infer = _ensure_spatial(adata_infer, adata_8um)

    adata_imput_align, _, _ = align_adata_fst2hd(adata_imput, adata_8um)
    logger.info('Aligned imputed sub-bins to 8 µm: %s', adata_imput_align.shape)

    gene_tag = '_'.join(genes)
    sele_gene_cor(
        adata_8um,
        adata_imput_align.to_df(),
        gene_hv,
        gene=genes,
        ylabel='FineST Expression',
        size=5,
        save_path=os.path.join(fig_dir, f'{gene_tag}_cor_imput_8um.pdf'),
    )
    mean_cor_box(
        adata_8um,
        np.array(adata_imput_align.to_df()),
        logger,
        save_path=os.path.join(fig_dir, 'Boxplot_imput_8um_cor_count.pdf'),
    )
    for gene in genes:
        gene_expr_compare(
            adata_8um,
            gene,
            adata_imput_align.X,
            gene_hv,
            marker='s',
            s=0.1,
            save_path=os.path.join(fig_dir, f'{gene}_expr_imput_8um.pdf'),
        )

    adata_df_imput_align = reshape2adata(adata_8um, adata_imput_align, gene_hv)
    imput_8um_path = _resolve_path(
        system_path, save_adata_imput_8um_path or presets['save_adata_imput_8um']
    )
    align_8um_path = _resolve_path(
        system_path, save_adata_impt_align_8um_path or presets['save_adata_impt_align_8um']
    )
    os.makedirs(os.path.dirname(imput_8um_path), exist_ok=True)
    adata_imput.write_h5ad(imput_8um_path)
    adata_df_imput_align.write_h5ad(align_8um_path)
    logger.info('Saved %s and %s', imput_8um_path, align_8um_path)

    infer_reshape = _spot_expression_matrix(adata_infer_spot)
    imput_reshape = _spot_expression_matrix(adata_imput_spot)
    cor_hist(
        adata_count,
        pd.DataFrame(infer_reshape, index=adata_count.obs_names, columns=gene_hv),
        fig_size=(5, 4),
        trans=False,
        format='svg',
        save_path=os.path.join(fig_dir, 'Hist_infer_cor_16um.svg'),
    )
    adata_infer_align, _, _ = align_adata_fst2hd(adata_infer, adata_8um)
    cor_hist(
        adata_8um,
        adata_infer_align.to_df(),
        fig_size=(5, 4),
        trans=False,
        format='svg',
        save_path=os.path.join(fig_dir, 'Hist_infer_cor_8um.svg'),
    )
    cor_hist(
        adata_count,
        pd.DataFrame(imput_reshape, index=adata_count.obs_names, columns=gene_hv),
        max_step=0.02,
        min_step=0.01,
        fig_size=(5, 4),
        trans=False,
        format='svg',
        save_path=os.path.join(fig_dir, 'Hist_imput_cor_16um.svg'),
    )
    cor_hist(
        adata_8um,
        adata_imput_align.to_df(),
        fig_size=(5, 4),
        trans=False,
        format='svg',
        save_path=os.path.join(fig_dir, 'Hist_imput_cor_8um.svg'),
    )

    return {
        'adata_8um': adata_8um,
        'adata_imput_align': adata_imput_align,
        'adata_infer_align': adata_infer_align,
        'adata_df_imput_align': adata_df_imput_align,
    }


def evaluate_visiumhd(
    system_path='./',
    data_root='FineST_tutorial_data_VisiumHD',
    save_data_path=None,
    figure_save_path=None,
    genes=None,
    eval_input=True,
    eval_8um=True,
    adata_infer_path=None,
    adata_imput_path=None,
    save_adata_imput_8um_path=None,
    save_adata_impt_align_8um_path=None,
):
    """Evaluate Visium HD within-bin results vs input 16 µm bins and native 8 µm bins."""
    import os
    import scanpy as sc
    from .paths import normalize_data_root, visiumhd_path_presets
    from .utils import setup_logger

    presets = visiumhd_path_presets(normalize_data_root(data_root))
    save_data_path = save_data_path or presets['save_data_path']
    figure_save_path = figure_save_path or presets['figure_save_path']
    genes = genes or ['SPP1', 'COL1A1']
    presets = {**presets, 'save_data_path': save_data_path}

    fig_dir = _resolve_path(system_path, figure_save_path)
    os.makedirs(fig_dir, exist_ok=True)
    logger = setup_logger(fig_dir)

    artifacts = _load_step1_outputs(system_path, presets)
    logger.info('Loaded Step 1 outputs from %s', artifacts['save_dir'])

    results = dict(artifacts)

    if eval_input:
        evaluate_vs_input(
            artifacts['adata_count'],
            artifacts['adata_norml'],
            artifacts['adata_infer_spot'],
            artifacts['adata_imput_spot'],
            artifacts['gene_hv'],
            genes,
            fig_dir,
            logger,
            gene_expr_s=50,
            imput_cor_hist=True,
            prefix='16um',
        )

    if eval_8um:
        adata_infer = sc.read_h5ad(
            _resolve_h5ad(system_path, adata_infer_path or presets['save_adata_infer'])
        )
        adata_imput = sc.read_h5ad(
            _resolve_h5ad(system_path, adata_imput_path or presets['save_adata_imput'])
        )
        results.update(
            evaluate_visiumhd_vs_8um(
                system_path,
                presets,
                fig_dir,
                logger,
                genes,
                artifacts['adata_count'],
                adata_infer,
                artifacts['adata_infer_spot'],
                adata_imput,
                artifacts['adata_imput_spot'],
                artifacts['gene_hv'],
                save_adata_imput_8um_path=save_adata_imput_8um_path,
                save_adata_impt_align_8um_path=save_adata_impt_align_8um_path,
            )
        )

    logger.info('Visium HD evaluation complete. Figures saved to %s', fig_dir)
    return results


def evaluate_visiumhd_8um(*args, **kwargs):
    """Backward-compatible alias for ``evaluate_visiumhd(..., eval_input=False, eval_8um=True)``."""
    kwargs.setdefault('eval_input', False)
    kwargs.setdefault('eval_8um', True)
    return evaluate_visiumhd(*args, **kwargs)


def _str2bool(v):
    return str(v).lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def parse_eval_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate FineST within-spot imputation (Visium or Visium HD).'
    )
    parser.add_argument(
        '--platform',
        type=str,
        choices=['visium', 'visiumhd'],
        default='visiumhd',
        help='visium = NPC tutorial; visiumhd = CRC16 Visium HD tutorial',
    )
    parser.add_argument('--system_path', type=str, default='./', help='Package root path')
    parser.add_argument('--data_root', type=str, default=None, help='Tutorial data root')
    parser.add_argument('--save_data_path', type=str, default=None, help='SaveData directory')
    parser.add_argument('--figure_save_path', type=str, default=None, help='Figures output directory')
    parser.add_argument('--genes', type=str, nargs='+', default=None, help='Marker genes')
    parser.add_argument(
        '--eval_input',
        type=str,
        default='true',
        help='Compare with measured input bins/spots (true/false)',
    )
    parser.add_argument(
        '--eval_8um',
        type=str,
        default='true',
        help='Visium HD only: compare with native 8 µm bins (true/false)',
    )
    parser.add_argument('--adata_infer_path', type=str, default=None)
    parser.add_argument('--adata_imput_path', type=str, default=None)
    parser.add_argument('--save_adata_imput_8um_path', type=str, default=None)
    parser.add_argument('--save_adata_impt_align_8um_path', type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_eval_args(argv)

    common = dict(
        system_path=args.system_path,
        save_data_path=args.save_data_path,
        figure_save_path=args.figure_save_path,
    )
    if args.genes is not None:
        common['genes'] = args.genes

    if args.platform == 'visium':
        evaluate_visium(
            data_root=args.data_root or 'FineST_tutorial_data',
            **common,
        )
    else:
        evaluate_visiumhd(
            data_root=args.data_root or 'FineST_tutorial_data_VisiumHD',
            eval_input=_str2bool(args.eval_input),
            eval_8um=_str2bool(args.eval_8um),
            adata_infer_path=args.adata_infer_path,
            adata_imput_path=args.adata_imput_path,
            save_adata_imput_8um_path=args.save_adata_imput_8um_path,
            save_adata_impt_align_8um_path=args.save_adata_impt_align_8um_path,
            **common,
        )


if __name__ == '__main__':
    main()

#############################