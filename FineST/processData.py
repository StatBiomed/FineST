import pandas as pd
import scanpy as sc
import time
from scipy.sparse import csr_matrix
from .utils import *
from .inference import *
import pickle
import json
from pathlib import Path
import anndata
import numpy as np
from scipy.spatial import cKDTree
import torch
import pickle


###################################################
# 2025.01.16 add embeds_convert from istar
###################################################
def istar_embeds_convert(hist_emb, locs, current_shape, image_embedings='sub', k=16):
    """
    Processes the embeddings and calculates the nearest pixel locations.
    Parameters:
        hist_emb (dict): Dictionary containing the 'sub' key with the embeddings.
        locs (numpy.ndarray): The locations to be processed.
        current_shape (numpy.ndarray): The current shape.
        target_shape (numpy.ndarray): The target shape.
    Returns:
        numpy.ndarray: The ordered locations.
        numpy.ndarray: The ordered images.
    """
    def load_pickle(filename, verbose=True):
        with open(filename, 'rb') as file:
            x = pickle.load(file)
        if verbose:
            print(f'Pickle loaded from {filename}')
        return x

    ## form (W-H) to (H-W)
    locs = locs[['y','x']]    
    embs = load_pickle(str(hist_emb))

    ## Reshape the 3D array into the shape (width, height, depth)
    if image_embedings=='sub':
        embs_sub = np.array(embs['sub'])
        embs = embs_sub.transpose(1, 2, 0)
    elif image_embedings=='cls':
        embs_cls = np.array(embs['cls'])
        embs = embs_cls.transpose(1, 2, 0)
    elif image_embedings=='cls_sub':
        embs = np.concatenate([embs['cls'], embs['sub']])
        embs = embs.transpose(1, 2, 0)
    elif image_embedings=='cls_sub_rgb':
        embs = np.concatenate([embs['cls'], embs['sub'], embs['rgb']])
        embs = embs.transpose(1, 2, 0)
    
    ## Rescale locations
    target_shape = embs.shape[:2]
    rescale_factor = current_shape // target_shape
    locs = locs.astype(float)
    locs /= rescale_factor
    locs = locs.round().astype(int)

    ## convert (width, height, depth) to (width*height, depth)
    imgs_pixel = []
    locs_pixel = []
    for i in range(embs.shape[0]):        # height
        for j in range(embs.shape[1]):    # width
            embeddings = embs[i, j, :]
            if not np.isnan(embeddings).any():  # Check if any value in embeddings is NaN
                imgs_pixel.append(embeddings)
                locs_pixel.append([i+1, j+1])

    locs_pixel = np.array(locs_pixel)
    imgs_pixel = np.array(imgs_pixel)

    ## convert (width*height, depth) to (#spots, depth)
    tree = cKDTree(locs_pixel)
    _, closest_points_indices = tree.query(locs, k)

    locs_order = locs_pixel[closest_points_indices]
    imgs_order = imgs_pixel[closest_points_indices]
    imgs_order_2d = torch.from_numpy(imgs_order.reshape(imgs_order.shape[0]*imgs_order.shape[1], -1))
                                     
    return locs_order, imgs_order, imgs_order_2d


###################################################
# 2025.01.16 add patch_size calculate from adata
# Return x_dir： patch_size at x/weidth direction 
###################################################

def patch_size(adata, p=16, dir='x'):
    """
    Computes the absolute differences of the sorted spatial data in adata.
    Parameters:
        adata: anndata object which contains the spatial data
        p: int, number of rows to select after sorting (default is 16)
        dir: str, direction to sort by, either 'x' or 'y' (default is 'x')
    Returns:
        differences: pandas Series, the computed absolute differences
    """
    if dir == 'x':    # fix hight
        spatial_test = pd.DataFrame(adata.obsm['spatial']).sort_values(by=1)[:p]
        differences = spatial_test[0].diff().abs()
    elif dir == 'y':   # fix weidth
        spatial_test = pd.DataFrame(adata.obsm['spatial']).sort_values(by=0)[:p]
        differences = spatial_test[1].diff().abs()
    else:
        print("Invalid direction. Please choose either 'x' or 'y'.")

    differences = differences.dropna()
    return differences


########################################
# 2025.01.15 For using istar img feature
########################################
def position_order_adata_istar(position, obs_names, dataset_class='Visium16'):
    ## Filter rows and set new index
    position_order = position[position[position.columns[-5]] == 1]
    position_order = position_order.set_index(position_order.columns[-6])

    ## Reorder index and drop column
    position_order = position_order.reindex(obs_names)
    position_order = position_order.drop(columns=[position.columns[-5]])

    ## Rename and reorder columns
    if dataset_class == 'Visium16' or dataset_class == 'Visium64':
        position_order.columns = ['array_col', 'array_row', 'pixel_x', 'pixel_y']
    elif dataset_class == 'VisiumHD':
        position_order.columns = ['array_col', 'array_row', 'pixel_x', 'pixel_y']
    elif dataset_class == 'VisiumHD_MS64':
        position_order.columns = ['array_col', 'array_row', 'pixel_y', 'pixel_x']
    
    position_order = position_order[['pixel_y','pixel_x', 'array_row', 'array_col']]

    return position_order

########################################
# 2024.12.2 add json file
########################################
def json_load(json_path):
    path_to_visium_bundle = Path(str(json_path)).expanduser()
    with open(path_to_visium_bundle / "scalefactors_json.json") as file:
        visium_scale_factors = json.load(file)
    return visium_scale_factors

def parquet2csv(parquet_path, parquet_name='tissue_positions.parquet'):
    os.chdir(str(parquet_path))
    positions = pd.read_parquet(parquet_name)    
    positions.set_index('barcode', inplace=True)
    ## inverse pxl_row_in_fullres and pxl_col_in_fullres
    positions.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']
    position_tissue = positions[positions['in_tissue'] == 1]
    return position_tissue


######################################
# 2024.11.13 add for Visium position
######################################
def filter_pos_list(filename):
    """
    Reads CSV file, renames the columns, and filters the rows where 'in_tissue' is 1.
        filename: str, The name of the CSV file to read.
        Returns a DataFrame.
    """
    position = pd.read_csv(filename, header=None)
    position.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 
                        'pxl_row_in_fullres', 'pxl_col_in_fullres']
    position_tissue = position[position['in_tissue'] == 1]
    return position_tissue


######################################
# 2024.11.13 add for Interpolate spot
# faster version
######################################
def inter_spot(position, direction):
    """
    Returns a DataFrame of midpoints between adjacent points in the specified direction.
        position : DataFrame
        direction: str, either 'x' or 'y'
    """
    ## Order position according to array_row and array_col 
    position_ordered = position.sort_values(['array_col', 'array_row'], 
                                            ascending=[True, True]).reset_index(drop=True)

    mid_points_list = []

    ## Iterate over position_ordered
    for _, row in position_ordered.iterrows():
        if direction == 'x':
            # Find the next point in the 'x' direction
            next_row = position_ordered[
                (position_ordered['array_row'] == row['array_row'] + 2) & 
                (position_ordered['array_col'] == row['array_col'])
            ]

            if not next_row.empty:
                next_row = next_row.iloc[0]
                mid_pxl_row = (row['pxl_row_in_fullres'] + next_row['pxl_row_in_fullres']) / 2
                mid_pxl_col = (row['pxl_col_in_fullres'] + next_row['pxl_col_in_fullres']) / 2

                mid_points_list.append({
                    'array_row': row['array_row'] + 1,
                    'array_col': row['array_col'],
                    'pxl_row_in_fullres': mid_pxl_row,
                    'pxl_col_in_fullres': mid_pxl_col
                })

        elif direction == 'y' and row['array_col'] < 127:
            ## Find the nearest points in the horizontal direction
            nearest_rows = position_ordered[
                position_ordered['array_col'] == row['array_col'] + 1
            ].copy()
            nearest_rows['distance'] = np.abs(nearest_rows['array_row'] - row['array_row'])
            nearest_rows = nearest_rows.nsmallest(2, 'distance')

            ## Compute the midpoints
            for _, nearest_row in nearest_rows.iterrows():
                mid_points_list.append({
                    'array_row': (row['array_row'] + nearest_row['array_row']) / 2,
                    'array_col': (row['array_col'] + nearest_row['array_col']) / 2,
                    'pxl_row_in_fullres': (row['pxl_row_in_fullres'] + nearest_row['pxl_row_in_fullres']) / 2,
                    'pxl_col_in_fullres': (row['pxl_col_in_fullres'] + nearest_row['pxl_col_in_fullres']) / 2
                })

    ## Create DataFrame from list of midpoints
    position_add = pd.DataFrame(mid_points_list)

    if direction == 'y':
        position_add = position_add.drop_duplicates()    # Remove duplicates for 'y' direction

    return position_add


def final_pos_list(position_x, position_y, position=None):
    """
    Returns DataFrame: the concatenation of position, 
    position_x, and position_y, sorted by 'array_col' and 'array_row'.
        position_x, position_y : DataFrame
    """
    ## Concatenate position (if provided), position_x, and position_y
    if position is not None:
        position_final = pd.concat([
            position[['array_row', 'array_col', 
                      'pxl_row_in_fullres', 'pxl_col_in_fullres']], 
            position_x, 
            position_y
        ], ignore_index=True)
    else:
        position_final = pd.concat([position_x, position_y], ignore_index=True)

    # #Sort position_final by 'array_col' and 'array_row'
    position_final = position_final.sort_values(['array_col', 'array_row'], 
                                                ascending=[True, True]).reset_index(drop=True)

    return position_final


######################################
# 2024.11.12 add for pathway analysis
######################################
def clean_save_adata(adata, filename):
    adata_save = adata.copy()

    ## List of keys to remove
    keys_to_remove = ['single_cell', 'mean', 'num_pairs',
                      # 'ligand', 'receptor',  'geneInter'                      
                      'global_I', 'global_stat', 'global_res', 'local_z'
                      # 'local_stat', 'local_z_p', 
                      # 'selected_spots',
                      ]

    for key in keys_to_remove:
        if key in adata_save.uns:
            del adata_save.uns[key]

    ## Update problematic elements in adata_save.uns and save them as pickle files
    for key, value in adata_save.uns.items():
        ## Save the original value as a pickle file
        with open(f"{key}.pkl", "wb") as f:
            pickle.dump(value, f)
        ## Update the value in adata with the filename of the pickle file
        adata_save.uns[key] = f"{key}.pkl"

    adata_save.write_h5ad(filename)
    return adata_save


def Load_clean_save_adata(adata):
    keys = ["local_z_p", "local_stat", "geneInter", "ligand", "receptor", "selected_spots",
            'histology_results_binary', 'histology_results_continu']
    for key in keys:
        with open(adata.uns[key], "rb") as f:
            adata.uns[key] = pickle.load(f)
    return adata


######################################
# 2024.11.11 add for all spot: Visium
######################################
def get_allspot_coors(input_coord_all):
    
    tensor_1 = input_coord_all[0][0]
    tensor_2 = input_coord_all[0][1]

    input_coord_all_concat = torch.stack((tensor_1, tensor_2))
    spatial_loc = input_coord_all_concat.T.numpy()

    ## Find unique rows and their counts
    unique_rows, counts = np.unique(spatial_loc, axis=0, return_counts=True)
    ## Check if there are any duplicate rows
    duplicate_rows = (counts > 1).any()
    print("Are there any duplicate rows? :", duplicate_rows)
    return spatial_loc


def adata_LR(adata, file_path):
    LRgene = pd.read_csv(file_path)
    adata.var_names_make_unique()
    if isinstance(adata.X, np.ndarray):
        adata_matrix = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    else:
        adata_matrix = pd.DataFrame(adata.X.A, index=adata.obs_names, columns=adata.var_names)
    available_genes = [gene for gene in LRgene['LR gene'].tolist() if gene in adata_matrix.columns]
    adataLR_matrix = adata_matrix[available_genes]
    adata._n_vars = adataLR_matrix.shape[1]
    adata.X = adataLR_matrix.values
    adata.var = adata.var.loc[available_genes]
    adata.var_names = adataLR_matrix.columns
    return adata


#####################################
# 2024.12.23: Add 
#####################################
def adata_preprocess(adata, keep_raw=False, normalize=True, 
                     min_cells=10, target_sum=None, n_top_genes=None, species='human'):
    """
    Preprocesses AnnData object for single-cell RNA-seq data.
    Parameters:
        adata (anndata.AnnData): The annotated data matrix of shape n_obs x n_vars. 
        keep_raw (bool, optional): If True, a copy of the original data is saved. Default is False.
        min_cells (int, optional): Minimum number of cells expressed. Default is 10.
        target_sum (float, optional): If not None, normalize total counts per cell with this value. 
                                    If None, after normalization, each cell has a total count 
                                    equal to the median of the counts_per_cell before normalization. 
                                    Default is None.
        n_top_genes (int, optional): Number of highly-variable genes to keep. 
                                    If n_top_genes is not None, this number is kept as 
                                    highly-variable genes. Default is None.
        species (str, optional): The species of the dataset. If not 'human', certain steps are skipped.
    Returns:
        adata (anndata.AnnData): The processed annotated data matrix.
    """

    ## Set mitochondrial gene prefix based on species
    if species == 'human':
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
    elif species == 'mouse':
        adata.var["mt"] = adata.var_names.str.startswith("mt-")
    else:
        raise ValueError("Unsupported species. Please specify 'human' or 'mouse'.")

    ## Calculate QC metrics if there are mitochondrial genes
    if adata.var["mt"].any():
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if keep_raw:
        adata = adata.copy()     # del adata.raw   

    if normalize:
        if target_sum is not None:
            sc.pp.normalize_total(adata, target_sum=target_sum)
        else:
            sc.pp.normalize_total(adata)

        sc.pp.log1p(adata)

    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)

    return adata


def adata2matrix(adata, gene_hv):
    ## Access the matrix and convert it to a dense matrix
    if isinstance(adata.X, np.ndarray):
        matrix = pd.DataFrame(adata.X)
    else:
        matrix = pd.DataFrame(adata.X.todense())
    matrix.columns = gene_hv
    spotID = np.array(pd.DataFrame(adata.obs['in_tissue']).index)
    matrix.insert(0, '', spotID)   
    matrix = matrix.set_index(matrix.columns[0])
    print(matrix.shape)
    return matrix

#
###############################################
# 2024.11.02 update 
# 2024.12.18 add VisiumSC
###############################################
def get_image_coord(file_paths, dataset_class):
    data = []
    file_paths.sort() 
    for file_path in file_paths:
        parts = file_path.split('_')
        if dataset_class == 'Visium' or dataset_class == 'VisiumSC':
            part_3 = int(parts[-2])
            part_4 = int(parts[-1].split('.')[0])
        elif dataset_class == 'VisiumHD':
            part_3 = parts[-2]
            part_4 = parts[-1].split('.pth')[0]
        else:
            print("Invalid dataset_class. Please use 'Visium', 'VisiumSC' or 'VisiumHD'")
            return
        data.append([part_3, part_4])
    df = pd.DataFrame(data, columns=['pixel_y', 'pixel_x'])
    return df[['pixel_x', 'pixel_y']]
    

def get_image_coord_all(file_paths):
    file_paths.sort()
    data = []
    for file_path in file_paths:
        parts = file_path.split('_')
        data.append([parts[-2], parts[-1].split('.pth')[0]])
        # if dataset_class == 'Visium' or dataset_class == 'VisiumSC':
        #     data.append([parts[-2], parts[-1].split('.pth')[0]])
    return data


def image_coord_merge(df, position, dataset_class):
    def merge_dfs(df, position):
        merged_df = pd.merge(df, position, on=['pixel_x', 'pixel_y'], how='left')
        cols = merged_df.columns.tolist()
        cols.remove('pixel_x')
        cols.remove('pixel_y')
        merged_df = merged_df[cols + ['pixel_x', 'pixel_y']]
        col_x = merged_df.columns[-4]
        col_y = merged_df.columns[-3]
        return merged_df.rename(columns={col_x: 'x', col_y: 'y'})

    #######################################################
    # 2024.12.04 postion doesn't match image
    #######################################################
    def merge_dfs_HD(df, position):
        position['pxl_col_in_fullres'] = pd.to_numeric(position['pxl_col_in_fullres'], errors='coerce').round(6)
        position['pxl_row_in_fullres'] = pd.to_numeric(position['pxl_row_in_fullres'], errors='coerce').round(6)
        position = position.rename(columns={'pxl_col_in_fullres': 'pixel_x', 'pxl_row_in_fullres': 'pixel_y'})

        df['pixel_x'] = df['pixel_x'].astype('float64').round(6)
        df['pixel_y'] = df['pixel_y'].astype('float64').round(6)

        merged_df = pd.merge(df, position, on=['pixel_x', 'pixel_y'], how='left')
        cols = merged_df.columns.tolist()
        cols.remove('pixel_x')
        cols.remove('pixel_y')
        merged_df = merged_df[cols + ['pixel_x', 'pixel_y']]
        col_x = merged_df.columns[-4]
        col_y = merged_df.columns[-3]
        return merged_df.rename(columns={col_x: 'x', col_y: 'y'})

    #     in_df = position['pixel_x'].isin(df['pixel_x']) & position['pixel_y'].isin(df['pixel_y'])
    #     merged_df = position[in_df].reset_index(drop=True)
    #     merged_df = merged_df.rename(columns={'array_row': 'x', 'array_col': 'y'})
    #     return merged_df

    ## Use dataset_class to decide which function to call
    if dataset_class == 'Visium' or dataset_class=='VisiumSC':
        result = merge_dfs(df, position)
    elif dataset_class == 'VisiumHD':
        result = merge_dfs_HD(df, position)
    else:
        raise ValueError(f"Unknown dataset_class: {dataset_class}")

    ## Check if the merge was successful
    if result.empty:
        raise ValueError("The merging resulted in an empty DataFrame. Please check your input data.")

    # ## For Visium
    # result = result.drop_duplicates(subset=result.columns[0], keep='first')

    return result


def sort_matrix(matrix, position_image, spotID_order, gene_hv):
    ## Reset the index of the matrix and rename the first column
    position_image_first_col = position_image.columns[0]
    matrix = matrix.reset_index().rename(columns={matrix.index.name: position_image_first_col})
    
    ## Merge position_image and matrix based on the first column
    sorted_matrix = pd.merge(position_image[[position_image_first_col]], matrix, 
                             on=position_image_first_col, how="left")
    
    # ################################################
    # # different: delete the same row For VisiumHD
    # ################################################
    # sorted_matrix = sorted_matrix.drop_duplicates(subset=position_image_first_col, keep='first')
    
    matrix_order = np.array(sorted_matrix.set_index(position_image_first_col))
    
    ## Convert matrix_order to DataFrame and set the index and column names
    matrix_order_df = pd.DataFrame(matrix_order)
    matrix_order_df.index = spotID_order
    matrix_order_df.columns = gene_hv
    
    return matrix_order, matrix_order_df


def update_adata_coord(adata, matrix_order, position_image, 
                       spotID_order=None, gene_hv=None, dataset_class='Visium16'):
    if dataset_class in ['Visium16', 'Visium64']:
        adata.X = csr_matrix(matrix_order, dtype=np.float32)
        adata.obs_names = matrix_order.index    # order by image feature name 
        adata.obsm['spatial'] = np.array(position_image.loc[:, ['pixel_y', 'pixel_x']])
        adata.obs['array_row'] = np.array(position_image.loc[:, 'y'])
        adata.obs['array_col'] = np.array(position_image.loc[:, 'x'])

    elif dataset_class == 'VisiumHD':
        sparse_matrix = csr_matrix(matrix_order, dtype=np.float32)
        ## construct new adata (reduce 97 coords)
        adata_redu = sc.AnnData(X=sparse_matrix, 
                                obs=pd.DataFrame(index=spotID_order), 
                                var=pd.DataFrame(index=gene_hv))
        adata_redu.X = csr_matrix(matrix_order, dtype=np.float32)
        adata_redu.obsm['spatial'] = np.array(position_image.loc[:, ['pixel_y', 'pixel_x']])
        adata_redu.obs['array_row'] = np.array(position_image.loc[:, 'y'])
        adata_redu.obs['array_col'] = np.array(position_image.loc[:, 'x'])
        ## add another objects
        adata_redu.var = adata.var
        adata_redu.uns = adata.uns
        adata = adata_redu.copy()
    else:
        raise ValueError("Invalid dataset_class. Expected 'Visium16', 'Visium64' or 'VisiumHD'.")
    
    return adata


def update_st_coord(position_image):
    position_order = pd.DataFrame({
        "pixel_y": position_image.loc[:, 'pixel_y'],
        "pixel_x": position_image.loc[:, 'pixel_x'],
        "array_row": position_image.loc[:, 'y'],
        "array_col": position_image.loc[:, 'x']
    })
    return position_order


###########################################
# 2025.01.06 sdjust weight and optimal nbs
###########################################
def impute_adata(adata, adata_spot, C2, gene_hv, dataset_class, weight_exponent=1):
    '''
    Prepare impute_adata: Fill gene expression using nbs
        adata_know: adata (original) 1331 × 596
        adata_spot: all subspot 21296 × 596
    '''
    adata_know = adata.copy()
    adata_know.obs[["x", "y"]] = adata.obsm['spatial']
    # adata_spot.obsm['spatial'] = adata_spot.obs[["x", "y"]].values
    sudo = pd.DataFrame(C2, columns=["x", "y"])
    sudo_adata = sc.AnnData(np.zeros((sudo.shape[0], len(gene_hv))), obs=sudo, var=adata.var)

    ## set ‘split_num’, according 'dataset_class'
    if dataset_class == 'Visium16':
        k_nbs = 6
        split_num = 16
    elif dataset_class == 'Visium64':
        k_nbs = 6
        split_num = 64
    elif dataset_class == 'VisiumSC':
        k_nbs = 6
        split_num = 1
    elif dataset_class == 'VisiumHD':
        k_nbs = 4
        split_num = 4
    else:
        raise ValueError('Invalid dataset_class. Only "Visium16", "Visium64", "VisiumSC" and "VisiumHD" are supported.')

    ## Impute_adata
    start_time = time.time()

    nearest_points = find_nearest_point(adata_spot.obsm['spatial'], adata_know.obsm['spatial'])
    nbs, nbs_indices = find_nearest_neighbors(nearest_points, adata_know.obsm['spatial'], k=k_nbs)
    distances = calculate_euclidean_distances(adata_spot.obsm['spatial'], nbs)

    ## Iterate over each point in sudo_adata
    for i in range(sudo_adata.shape[0]):
        dis_tmp = (distances[i] + 0.1) / np.min(distances[i] + 0.1)
        weights = ((1 / (dis_tmp ** weight_exponent)) / ((1 / (dis_tmp ** weight_exponent)).sum()))
        
        if isinstance(adata_know.X, np.ndarray):
            sudo_adata.X[i, :] = np.dot(weights, adata_know.X[nbs_indices[i]])
        else:
            sudo_adata.X[i, :] = np.dot(weights, adata_know.X[nbs_indices[i]].todense())

        # sudo_adata.X[i, :] = np.dot(weights, adata_know.X[nbs_indices[i]].todense())
        # sudo_adata.X[i, :] = np.dot(weights, adata_know.X[nbs_indices[i]].todense() / split_num)

    print("--- %s seconds ---" % (time.time() - start_time))

    ## add other objects to adata
    sudo_adata.obsm['spatial'] = adata_spot.obsm['spatial']
    sudo_adata.uns['spatial'] = adata.uns['spatial']

    return sudo_adata

######################################
# 2025.02.08: add scale 
######################################
def weight_adata(adata_spot, sudo_adata, gene_hv, w=0.5, do_scale=False):
    """
    Combine inferred super-resolved gene expression data with imputed data, and optionally scale the result.
    Parameters:
        adata_spot (sc.AnnData): Inferred super-resolved gene expression data with high resolution.
        sudo_adata (sc.AnnData): Imputed data using k-nearest neighbors within spots.
        gene_hv (list): List of highly variable genes.
        w (float, optional): Weight for combining the two datasets. Defaults to 0.5.
        do_scale (bool, optional): Whether to scale the combined data. Defaults to False.
    Returns:
        sc.AnnData: Combined and optionally scaled AnnData object.
        torch.Tensor: The combined data as a PyTorch tensor.
    """

    ## Optionally scale the combined data
    if do_scale:
        weight_impt_data = w * scale(adata_spot.X) + (1 - w) * scale(sudo_adata.X)
    else:
        weight_impt_data = w * adata_spot.X + (1 - w) * sudo_adata.X

    ## Convert the combined data to a PyTorch tensor
    data_impt = torch.tensor(weight_impt_data)

    ## Create a new AnnData object with the combined data
    adata_impt = sc.AnnData(X=pd.DataFrame(weight_impt_data))
    adata_impt.var_names = gene_hv
    adata_impt.obs = adata_spot.obs

    ## Add other necessary objects to the new AnnData object
    adata_impt.obsm['spatial'] = sudo_adata.obsm['spatial']
    adata_impt.uns['spatial'] = sudo_adata.uns['spatial']

    return adata_impt, data_impt


def reshape2adata(adata, adata_impt_all_reshape, gene_hv, spatial_loc_all=None):

    if isinstance(adata_impt_all_reshape, torch.Tensor):
        adata_impt_spot = sc.AnnData(X = adata_impt_all_reshape.numpy())
    elif isinstance(adata_impt_all_reshape, anndata.AnnData):
        adata_impt_spot = sc.AnnData(X = adata_impt_all_reshape.to_df())

    adata_impt_spot.var_names = gene_hv

    if adata_impt_all_reshape.shape[0] == adata.shape[0]:
        adata_impt_spot.obs_names = adata.obs_names
        adata_impt_spot.obsm['spatial'] = adata.obsm['spatial'] 
        adata_impt_spot.obs = adata.obs
        adata_impt_spot.var = adata.var
    else: 
        adata_impt_spot.obs['x'] = spatial_loc_all[:,0]
        adata_impt_spot.obs['y'] = spatial_loc_all[:,1]
        adata_impt_spot.obsm['spatial'] = adata_impt_spot.obs[["x", "y"]].values
    
    ## Check if adata has 'uns' and 'spatial' key
    if hasattr(adata, 'uns') and 'spatial' in adata.uns:
        adata_impt_spot.uns['spatial'] = adata.uns['spatial']

    return adata_impt_spot