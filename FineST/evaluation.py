import numpy as np
import  logging
logging.getLogger().setLevel(logging.INFO)
from .utils import *
from .loadData import *
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import torch


#############################
# 2024.11.16 align 8um adata
#############################
def align_adata_fst2hd(adata_impt, adata_8um):
    """
    Uses cKDTree to find the closest point in 'adata_impt' for each spatial point in 'adata_8um'.
    Creates 'adata_impt_align' dataset with observation names matching 'adata_8um' based on closest points indices.
    Finally, it converts both datasets into DataFrames and prints their shapes.

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
def calculate_correlation(matrix_tensor_test_np, reconstructed_matrix_test_np, method="pearson", sample="spot"):

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
            corr, _ = spearmanr(x, y)  # np.corrcoef does not support Spearman correlation
        else:
            raise ValueError("Invalid method, choose either 'pearson' or 'spearman'")
        corr = np.nanmean(corr) if not np.isnan(corr).all() else 0
        correlation_coefficients.append(corr)

    return correlation_coefficients


def mean_cor(adata, data_impt_reshape, label, sample="gene"):

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


###########################################################################################
# Inference Correlation  
###########################################################################################
def calculate_correlation_infer(matrix_tensor_test_np, reconstructed_matrix_test_np, method="pearson", sample="spot"):
    # Check for NaN values in the input matrices
    if np.isnan(matrix_tensor_test_np).any() or np.isnan(reconstructed_matrix_test_np).any():
        print("Warning: The input matrices contain NaN values. Please handle them before calculating correlations.")
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
        if np.std(matrix_tensor_test_np[i] if data_index==0 else matrix_tensor_test_np[:,i]) == 0 or np.std(reconstructed_matrix_test_np[i] if data_index==0 else reconstructed_matrix_test_np[:,i]) == 0:
            continue

        if method == "pearson":
            # Use NumPy's corrcoef function to compute the correlation coefficient
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


###########################################################################################
# cosine_similarity row (default)  
###########################################################################################
def calculate_cosine_similarity_row(rep_query_adata, rep_ref_adata_image_reshape):
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
    ## cells are in columns, genes are in rows
    if qc_idx is not None:
        expression_gt = expression_gt[:, qc_idx]
        matched_spot_expression_pred = matched_spot_expression_pred[:, qc_idx]

    mean = np.mean(expression_gt, axis=1)
    top_genes_idx = np.argpartition(mean, -top_k)[-top_k:]

    corr = [np.corrcoef(expression_gt[i, :], matched_spot_expression_pred[i, :])[0, 1] for i in top_genes_idx]

    return np.mean(corr)