import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import fdrcorrection
from threadpoolctl import threadpool_limits
from .utils import *
from itertools import zip_longest
import anndata as ann
import scipy
import pyreadr
import requests





#######################################
# 2024.11.28 Add code of LR-TF
# , datahost='builtin' -- !!
#######################################
def extract_tf(species, datahost='package'):
    """
    find overlapping LRs from CellChatDB
    :param species: support 'human', 'mouse' and 'zebrafish'
    :param datahost: the host of the ligand-receptor data. 
                    'builtin' for package built-in otherwise from figshare
    :return: LR_TF (containing comprehensive info from CellChatDB) dataframe
    """
    
    if datahost == 'package':

        if species in ['mouse', 'human']:
            # datapath = '/mnt/lingyu/nfs_share2/Python/FineST/FineST/FineST/datasets/TF_data/%s-' %(species)
            # datapath = './datasets/TF_data/%s-' %(species)
            datapath = './FineST/datasets/TF_data/%s-' %(species)
        else:
            raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
        
        LR_TF = pyreadr.read_r(datapath + 'TF_PPRhuman.rda')['TF_PPRhuman']

    else:
        if species == 'mouse':
            url = 'https://figshare.com/ndownloader/files/50860644'
        elif species == 'human':
            url = 'https://figshare.com/ndownloader/files/50860650'
        else:
            raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
        
        # specify where to download the file
        # download_path = '/mnt/lingyu/nfs_share2/Python/FineST/FineST/FineST/datasets/TF_data/temp.rda'
        # download_path = './datasets/TF_data/'
        download_path = './FineST/datasets/TF_data/'
        
        # download the file
        r = requests.get(url)
        with open(download_path, 'wb') as f:
            f.write(r.content)

        LR_TF = pyreadr.read_r(download_path)['TF_PPRhuman']

        # remove the downloaded file after use
        os.remove(download_path)

        # if species == 'mouse':
        #     LR_TF = pyreadr.read_r('https://figshare.com/ndownloader/files/50860644')['TF_PPRhuman']
        #     # TF_TG = pyreadr.read_r('https://figshare.com/ndownloader/files/50860656')
        # elif species == 'human':
        #     LR_TF = pyreadr.read_r('https://figshare.com/ndownloader/files/50860650')['TF_PPRhuman']
        #     # TF_TG = pyreadr.read_r('https://figshare.com/ndownloader/files/50860647')
        # else:
        #     raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
        
    return LR_TF


def top_pattern_LR2TF(tmp, ligand_list, receptor_list, top_num=20):
    """
    The function takes in a DataFrame and two lists of ligands and receptors respectively, 
    filter the DataFrame based on the lists and return the top rows sorted by 'value' column.
    Args:
    tmp : a DataFrame to process
    ligand_list : a list of ligands to filter on
    receptor_list : a list of receptors to filter on
    top_num : the number of top rows to return, defaults to 20

    Returns:
    tmp_df : a DataFrame after processing
    """

    #################################################
    # Ligand or Receptor contain pattern LR external
    #################################################
    # # Filter the DataFrame based on the ligand_list and receptor_list
    # lData = [tmp[tmp['Ligand'].str.contains(ligand, na=False)] for ligand in ligand_list]
    # rData = [tmp[tmp['Receptor'].str.contains(receptor, na=False)] for receptor in receptor_list]
    # print("Ligand in R2TFdatabase:", len(lData))
    # print("Receptor in R2TFdatabase:", len(rData))
    
    # # Concatenate the filtered DataFrame and drop duplicates
    # fData = pd.concat(lData + rData).drop_duplicates()
    # print("Ligand or Receptor in R2TFdatabase:", fData.shape[0])
    
    #################################################
    # Ligand or Receptor only contain pattern LR 
    #################################################
    # Filter the DataFrame based on the ligand_list and receptor_list
    fData = tmp[tmp['Ligand'].isin(ligand_list) & tmp['Receptor'].isin(receptor_list)]
    print("Ligand and Receptor in R2TFdatabase:", fData.shape[0])

    # Sort the DataFrame by 'value' column and return the top rows
    tmp_df = fData.sort_values(by='value', ascending=False).head(top_num)

    # rename colname
    subdf = pd.DataFrame(tmp_df.rename(columns={"Ligand": "Ligand_symbol", 
                                                "Receptor": "Receptor_symbol", 
                                                "tf": "TF", "value": "value"}) )    
    
    return subdf


def pattern_LR2TF2TG(histology_results, pattern_num, R_TFdatabase, TF_TGdatabase):
    """
    The function takes in a DataFrame, checks if column 'g' contains two '_', if so, it splits the 'g' column value into two new rows
    Args:
    histology_results : a DataFrame to process
    pattern_num : the pattern number to filter on, defaults to 0
    R_TFdatabase : the DataFrame containing receptor to TF mapping

    Returns:
    tmp : a DataFrame after processing
    """
    rows = []

    for i, row in histology_results.iterrows():
        # check if column 'g' contains two '_'
        if row['g'].count('_') == 2:
            # split the 'g' column value into three parts
            gene1, gene2, gene3 = row['g'].split('_')

            # create two new rows, gene1_gene2 and gene1_gene3 respectively
            new_row1 = row.copy()
            new_row1['g'] = gene1 + '_' + gene2
            new_row2 = row.copy()
            new_row2['g'] = gene1 + '_' + gene3

            # add the new rows to the result DataFrame
            rows.append(new_row1)
            rows.append(new_row2)
        else:
            # if column 'g' does not contain two '_', add the original row to the result DataFrame directly
            rows.append(row)

    # after the loop, create DataFrame at once
    p0_results = pd.DataFrame(rows, columns=histology_results.columns)
    
    LRp0 = p0_results[p0_results['pattern']==pattern_num]['g']

    Lp0 = [gene for pair in LRp0 for gene in pair.split('_')[0:1]]
    print("This pattern contain %s unique ligand", len(set(Lp0)))

    Rp0 = [gene for pair in LRp0 for gene in pair.split('_')[1:]]
    print("This pattern contain %s unique receptor", len(set(Rp0)))

    R_TFdata_df = R_TFdatabase[R_TFdatabase['receptor'].isin(Rp0)]
    ligand = Lp0
    receptor = Rp0

    result = pd.concat([pd.DataFrame(ligand), pd.DataFrame(receptor)], axis=1)
    result.columns = ['ligand', 'receptor']
    result = result.dropna()

    comm = result.merge(R_TFdata_df, on='receptor', how='left')
    comm = comm.dropna()

    tf_comm = [gene for gene in comm['tf']]
    print("This pattern contain %s unique tf", len(set(tf_comm)))
    
    R_TFdata_TG_df = TF_TGdatabase[TF_TGdatabase['tf'].isin(tf_comm)]
    R_TFdata_TG_df
    comm_all = comm.merge(R_TFdata_TG_df, on='tf', how='right')
    comm_all = comm_all.dropna().drop_duplicates()
    comm_all
    
    tmp = comm_all[["ligand","receptor","tf", "target", "tf_PPR"]]
    tmp = tmp.rename(columns={"ligand": "Ligand", "receptor": "Receptor", "tf": "tf", "target": "Target", "tf_PPR": "value"})
    tmp = tmp.drop_duplicates()
    
    return tmp


def pattern_LR2TF(histology_results, pattern_num, R_TFdatabase):
    """
    The function takes in a DataFrame, checks if column 'g' contains two '_', if so, it splits the 'g' column value into two new rows
    Args:
    histology_results : a DataFrame to process
    pattern_num : the pattern number to filter on, defaults to 0
    R_TFdatabase : the DataFrame containing receptor to TF mapping

    Returns:
    tmp : a DataFrame after processing
    """
    rows = []

    for i, row in histology_results.iterrows():
        # check if column 'g' contains two '_'
        if row['g'].count('_') == 2:
            # split the 'g' column value into three parts
            gene1, gene2, gene3 = row['g'].split('_')

            # create two new rows, gene1_gene2 and gene1_gene3 respectively
            new_row1 = row.copy()
            new_row1['g'] = gene1 + '_' + gene2
            new_row2 = row.copy()
            new_row2['g'] = gene1 + '_' + gene3

            # add the new rows to the result DataFrame
            rows.append(new_row1)
            rows.append(new_row2)
        else:
            # if column 'g' does not contain two '_', add the original row to the result DataFrame directly
            rows.append(row)

    # after the loop, create DataFrame at once
    p0_results = pd.DataFrame(rows, columns=histology_results.columns)
    
    LRp0 = p0_results[p0_results['pattern']==pattern_num]['g']

    Lp0 = [gene for pair in LRp0 for gene in pair.split('_')[0:1]]
    print("This pattern contain %s unique ligand", len(set(Lp0)))

    Rp0 = [gene for pair in LRp0 for gene in pair.split('_')[1:]]
    print("This pattern contain %s unique receptor", len(set(Rp0)))

    R_TFdata_df = R_TFdatabase[R_TFdatabase['receptor'].isin(Rp0)]
    ligand = Lp0
    receptor = Rp0

    result = pd.concat([pd.DataFrame(ligand), pd.DataFrame(receptor)], axis=1)
    result.columns = ['ligand', 'receptor']
    result = result.dropna()

    comm = result.merge(R_TFdata_df, on='receptor', how='left')
    comm = comm.dropna()

    tmp = comm[["ligand","receptor","tf","tf_PPR"]]
    tmp = tmp.rename(columns={"ligand": "Ligand", "receptor": "Receptor", "tf": "tf", "tf_PPR": "value"})
    tmp = tmp.drop_duplicates()
    
    return tmp


#######################################
# 2024.11.17 Add code of SpatialDM
#######################################
def LRpair_gene(df):
    """
    df : DataFrame

    ## see the unique gene of sig LR pairs

    Returns a DataFrame of unique elements from  'Ligand0', 'Ligand1', 'Receptor0', 
    'Receptor1', and 'Receptor2' columns of the DataFrame where 'selected' is True.
    """
    # Filter the DataFrame
    filtered_df = df[df['selected'] == True]

    # Get the unique elements
    unique_elements = set(filtered_df['Ligand0'].tolist() + 
                          filtered_df['Ligand1'].tolist() + 
                          filtered_df['Receptor0'].tolist() + 
                          filtered_df['Receptor1'].tolist() + 
                          filtered_df['Receptor2'].tolist())
    unique_elements_df = pd.DataFrame(unique_elements)

    return unique_elements_df


def anno_LRpair(adata_impt_all):
    """
    adata_impt_all : AnnData object

    Returns a DataFrame resulting from merging 
    adata_impt_all.uns['global_res'].sort_values(by='fdr')  with 
    adata_impt_all.uns['geneInter'] on 'Ligand0' and 'interaction_name'.
    """

    # Create a DataFrame from 'geneInter'
    geneInter_df = adata_impt_all.uns['geneInter']
    spa_coexp_pair = adata_impt_all.uns['global_res'].sort_values(by='fdr')  

    # Merge the dataframes
    merged_df = pd.merge(spa_coexp_pair, geneInter_df, how='left', 
                         left_index=True, right_index=True)

    # Keep only the columns of interest
    final_df = merged_df[['Ligand0', 'Ligand1', 'Receptor0', 'Receptor1', 
                          'Receptor2', 'z_pval', 'z', 'fdr', 'selected', 
                          'evidence', 'annotation']]

    return final_df


#######################################
# 2024.11.11.Adjust code of SpatialDM
#######################################

def _Euclidean_to_RBF(X, l, singlecell):
    """Convert Euclidean distance to RBF distance"""
    from scipy.sparse import issparse
    if issparse:
        rbf_d = X
        rbf_d[X.nonzero()] = np.exp(-X[X.nonzero()].A**2 / (2 * l ** 2))
    else:
        rbf_d = np.exp(- X**2 / (2 * l ** 2))
    
    # At single-cell resolution, no within-spot communications
    if singlecell:

        ###################
        # old for SpatialDM
        ###################
        # np.fill_diagonal(rbf_d, 0)

        rbf_d_dense = rbf_d.toarray()  # or rbf_d.todense()
        np.fill_diagonal(rbf_d_dense, 0)

    else:
        rbf_d.setdiag(np.exp(-X.diagonal()**2 / (2 * l ** 2)))

    return rbf_d


#######################################
# 2024.11.11.Adjust code of SpatialDM
#######################################
def weight_matrix(adata_impt_all, l, cutoff, single_cell=False, n_nearest_neighbors=6):

    """
    Compute weight matrix based on radial basis kernel, more efficient than SpatialDM.
    cutoff & n_neighbors are two alternative options to restrict signaling range.
    :param l: radial basis kernel parameter, need to be customized for optimal weight gradient and \
    to restrain the range of signaling before downstream processing.
    :param cutoff: (for secreted signaling) minimum weight to be kept from the rbf weight matrix. \
    Weight below cutoff will be made zero
    :param n_neighbors: (for secreted signaling) number of neighbors per spot from the rbf weight matrix.
    :param n_nearest_neighbors: (for adjacent signaling) number of neighbors per spot from the rbf \
    weight matrix.
    Non-neighbors will be made 0
    :param single_cell: if single cell resolution, diagonal will be made 0.
    :return: secreted signaling weight matrix: adata.obsp['weight'], \
            and adjacent signaling weight matrix: adata.obsp['nearest_neighbors']
    """

    adata_impt_all.uns['single_cell'] = single_cell
    if isinstance(adata_impt_all.obsm['spatial'], pd.DataFrame):
        X_loc = adata_impt_all.obsm['spatial'].values
    else:
        X_loc = adata_impt_all.obsm['spatial']

    n_neighbors = n_nearest_neighbors * 31

    #####################################
    # large neighborhood for W (5 layers)
    #####################################
    nnbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='euclidean').fit(X_loc)
    nbr_d = nnbrs.kneighbors_graph(X_loc, mode='distance')
    rbf_d = _Euclidean_to_RBF(nbr_d, l, single_cell)

    #####################################
    # small neighborhood for RBF
    #####################################
    nnbrs0 = NearestNeighbors(n_neighbors=n_nearest_neighbors, 
                              algorithm='ball_tree', metric='euclidean').fit(X_loc)
    nbr_d0 = nnbrs0.kneighbors_graph(X_loc, mode='distance')
    rbf_d0 = _Euclidean_to_RBF(nbr_d0, l, single_cell)

    #########################################################################
    # NOTE: add more info about cutoff, n_neighbors and n_nearest_neighbors
    #########################################################################
    if cutoff:
        #####################################
        # not efficient -- old code
        #####################################
        # rbf_d[rbf_d < cutoff] = 0
        
        #####################################################################
        # more efficient -- new code
        # https://seanlaw.github.io/2019/02/27/set-values-in-sparse-matrix/
        #####################################################################
        nonzero_mask = np.array(rbf_d[rbf_d.nonzero()] < cutoff)[0]
        rows = rbf_d.nonzero()[0][nonzero_mask]
        cols = rbf_d.nonzero()[1][nonzero_mask]
        rbf_d[rows, cols] = 0
    
    # elif n_neighbors:
    #     nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rbf_d)
    #     knn = nbrs.kneighbors_graph(rbf_d).toarray()
    #     rbf_d = rbf_d * knn

    adata_impt_all.obsp['weight'] = rbf_d * adata_impt_all.shape[0] / rbf_d.sum()
    adata_impt_all.obsp['nearest_neighbors'] = rbf_d0 * adata_impt_all.shape[0] / rbf_d0.sum()

    return adata_impt_all


#######################################
# 2024.11.11.Adjust code of SpatialDM
#######################################
def pathway_analysis(sample=None,
                    all_interactions=None,
                    groups=None, cut_off=None,  # LLY add
        interaction_ls=None, name=None, dic=None):
    """
    Compute enriched pathways for a list of pairs or a dic of SpatialDE results.
    :param sample: spatialdm obj
    :param ls: a list of LR interaction names for the enrichment analysis
    :param path_name: str. For later recall sample.path_summary[path_name]
    :param dic: a dic of SpatialDE results (See tutorial)
    """
    if interaction_ls is not None:
        dic = {name: interaction_ls}
    if sample is not None:
        ## the old one
        all_interactions = sample.uns['geneInter']
        ## Load the original object from a pickle file when needed
        # with open(sample.uns["geneInter"], "rb") as f:
        #     all_interactions = pickle.load(f)

    # print(all_interactions)
    df = pd.DataFrame(all_interactions.groupby('pathway_name').interaction_name)
    df = df.set_index(0)
    total_feature_num = len(all_interactions)
    result = []
    result_pattern = []
    for n,ls in dic.items():
        qset = set([x.upper() for x in ls]).intersection(all_interactions.index)
        query_set_size = len(qset)
        for modulename, members in df.iterrows():
            module_size = len(members.values[0])
            overlap_features = qset.intersection(members.values[0])
            overlap_size = len(overlap_features)
    
            negneg = total_feature_num + overlap_size - module_size - query_set_size
            # Fisher's exact test
            p_FET = scipy.stats.fisher_exact([[overlap_size, query_set_size - overlap_size],
                                        [module_size - overlap_size, negneg]], 'greater')[1]

            result.append((p_FET, modulename, module_size, overlap_size, overlap_features, n))
            result_pattern.append((p_FET, modulename, module_size, overlap_size, overlap_features, n, query_set_size, negneg))  # 将结果添加到列表中

    result = pd.DataFrame(result).set_index(1)
    result.columns = ['fisher_p', 'pathway_size', 'selected', 'selected_inters', 'name']

    result_pattern = pd.DataFrame(result_pattern).set_index(1)  # 将结果转换为数据框并设置索引为第二列
    result_pattern.columns = ['fisher_p', 'module_size', 'overlap_size', 'selected_inters', 'name', 'query_set_size', 'negneg']  # 设置数据框的列名
    
    if sample is not None:
        sample.uns['pathway_summary'] = result    

    #################################################################################
    # copy form Function dot_path(), get all pathway information
    #################################################################################
    pathway_res = result[result.selected >= cut_off]
    if groups is not None:
        pathway_res = pathway_res.loc[pathway_res.name.isin(groups)]

    #################################################################################
    # copy form Function dot(), get pattern visulization: selected pathway 
    #################################################################################
    for i, name in enumerate(pathway_res.name.unique()):
        result1 = pathway_res.loc[pathway_res.name == name]
        result1 = result1.sort_values('selected', ascending=False)
    
    # return result, confusion_matrix, pathway_res, result1, result_pattern, result_pattern_mat
    return result, pathway_res, result1, result_pattern


# def extract_lr(adata, species, mean='algebra', min_cell=0, datahost='builtin'):
#     """
#     find overlapping LRs from CellChatDB
#     :param adata: AnnData object
#     :param species: support 'human', 'mouse' and 'zebrafish'
#     :param mean: 'algebra' (default) or 'geometric'
#     :param min_cell: for each selected pair, the spots expressing ligand or receptor should be larger than the min,
#     respectively.
#     :param datahost: the host of the ligand-receptor data. 'builtin' for package built-in otherwise from figshare
#     :return: ligand, receptor, geneInter (containing comprehensive info from CellChatDB) dataframes \
#             in adata.uns
#     """
#     if mean=='geometric':
#         from scipy.stats.mstats import gmean
#     adata.uns['mean'] = mean

#     if datahost == 'package':
#         if species in ['mouse', 'human', 'zerafish']:
#             datapath = './datasets/LR_data/%s-' %(species)
#         else:
#             raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
        
#         import pkg_resources
#         stream1 = pkg_resources.resource_stream(__name__, datapath + 'interaction_input_CellChatDB.csv.gz')
#         geneInter = pd.read_csv(stream1, index_col=0, compression='gzip')

#         stream2 = pkg_resources.resource_stream(__name__, datapath + 'complex_input_CellChatDB.csv')
#         comp = pd.read_csv(stream2, header=0, index_col=0)
#     else:
#         if species == 'mouse':
#             geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638919', index_col=0)
#             comp = pd.read_csv('https://figshare.com/ndownloader/files/36638916', header=0, index_col=0)
#         elif species == 'human':
#             geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638943', header=0, index_col=0)
#             comp = pd.read_csv('https://figshare.com/ndownloader/files/36638940', header=0, index_col=0)
#         elif species == 'zebrafish':
#             geneInter = pd.read_csv('https://figshare.com/ndownloader/files/38756022', header=0, index_col=0)
#             comp = pd.read_csv('https://figshare.com/ndownloader/files/38756019', header=0, index_col=0)
#         else:
#             raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
        
#     geneInter = geneInter.sort_values('annotation')
#     ligand = geneInter.ligand.values
#     receptor = geneInter.receptor.values
#     geneInter.pop('ligand')
#     geneInter.pop('receptor')

#     ## NOTE: the following for loop needs speed up
#     t = []
#     for i in range(len(ligand)):
#         for n in [ligand, receptor]:
#             l = n[i]
#             if l in comp.index:
#                 n[i] = comp.loc[l].dropna().values[pd.Series \
#                     (comp.loc[l].dropna().values).isin(adata.var_names)]
#             else:
#                 n[i] = pd.Series(l).values[pd.Series(l).isin(adata.var_names)]
#         if (len(ligand[i]) > 0) * (len(receptor[i]) > 0):
#             if mean=='geometric':
#                 meanL = gmean(adata[:, ligand[i]].X, axis=1)
#                 meanR = gmean(adata[:, receptor[i]].X, axis=1)
#             else:
#                 meanL = adata[:, ligand[i]].X.mean(axis=1)
#                 meanR = adata[:, receptor[i]].X.mean(axis=1)
#             if (sum(meanL > 0) >= min_cell) * \
#                     (sum(meanR > 0) >= min_cell):
#                 t.append(True)
#             else:
#                 t.append(False)
#         else:
#             t.append(False)
#     ind = geneInter[t].index
#     adata.uns['ligand'] = pd.DataFrame.from_records(zip_longest(*pd.Series(ligand[t]).values)).transpose()
#     adata.uns['ligand'].columns = ['Ligand' + str(i) for i in range(adata.uns['ligand'].shape[1])]
#     adata.uns['ligand'].index = ind
#     adata.uns['receptor'] = pd.DataFrame.from_records(zip_longest(*pd.Series(receptor[t]).values)).transpose()
#     adata.uns['receptor'].columns = ['Receptor' + str(i) for i in range(adata.uns['receptor'].shape[1])]
#     adata.uns['receptor'].index = ind
#     adata.uns['num_pairs'] = len(ind)
#     adata.uns['geneInter'] = geneInter.loc[ind]
#     if adata.uns['num_pairs'] == 0:
#         raise ValueError("No effective RL. Please have a check on input count matrix/species.")
#     return

# def spatialdm_global(adata, n_perm=1000, specified_ind=None, method='z-score', nproc=1):
#     """
#         global selection. 2 alternative methods can be specified.
#     :param n_perm: number of times for shuffling receptor expression for a given pair, default to 1000.
#     :param specified_ind: array containing queried indices for quick test/only run selected pair(s).
#     If not specified, selection will be done for all extracted pairs
#     :param method: default to 'z-score' for computation efficiency.
#         Alternatively, can specify 'permutation' or 'both'.
#         Two approaches should generate consistent results in general.
#     :param nproc: default to 1. Please decide based on your system.
#     :return: 'global_res' dataframe in adata.uns containing pair info and Moran p-values
#     """
#     if specified_ind is None:
#         specified_ind = adata.uns['geneInter'].index.values  # default to all pairs
#     else:
#         adata.uns['geneInter'] = adata.uns['geneInter'].loc[specified_ind]
#     total_len = len(specified_ind)
#     adata.uns['ligand'] = adata.uns['ligand'].loc[specified_ind]#.values
#     adata.uns['receptor'] = adata.uns['receptor'].loc[specified_ind]#.values
#     adata.uns['global_I'] = np.zeros(total_len)
#     adata.uns['global_stat'] = {}
#     if method in ['z-score', 'both']:
#         adata.uns['global_stat']['z']={}
#         adata.uns['global_stat']['z']['st'] = globle_st_compute(adata)
#         adata.uns['global_stat']['z']['z'] = np.zeros(total_len)
#         adata.uns['global_stat']['z']['z_p'] = np.zeros(total_len)
#     if method in ['both', 'permutation']:
#         adata.uns['global_stat']['perm']={}
#         adata.uns['global_stat']['perm']['global_perm'] = np.zeros((total_len, n_perm)).astype(np.float16)

#     if not (method in ['both', 'z-score', 'permutation']):
#         raise ValueError("Only one of ['z-score', 'both', 'permutation'] is supported")

#     with threadpool_limits(limits=nproc, user_api='blas'):
#         pair_selection_matrix(adata, n_perm, specified_ind, method)

#     adata.uns['global_res'] = pd.concat((adata.uns['ligand'], adata.uns['receptor']),axis=1)
#     # adata.uns['global_res'].columns = ['Ligand1', 'Ligand2', 'Ligand3', 'Receptor1', 'Receptor2', 'Receptor3', 'Receptor4']
#     if method in ['z-score', 'both']:
#         adata.uns['global_stat']['z']['z_p'] = np.where(np.isnan(adata.uns['global_stat']['z']['z_p']),
#                                                       1, adata.uns['global_stat']['z']['z_p'])
#         adata.uns['global_res']['z_pval'] = adata.uns['global_stat']['z']['z_p']
#         adata.uns['global_res']['z'] = adata.uns['global_stat']['z']['z']

#     if method in ['both', 'permutation']:
#         adata.uns['global_stat']['perm']['global_p'] = 1 - (adata.uns['global_I'] \
#                              > adata.uns['global_stat']['perm']['global_perm'].T).sum(axis=0) / n_perm
#         adata.uns['global_res']['perm_pval'] = adata.uns['global_stat']['perm']['global_p']
#     return

# def sig_pairs(adata, method='z-score', fdr=True, threshold=0.1):
#     """
#         select significant pairs
#     :param method: only one of 'z-score' or 'permutation' to select significant pairs.
#     :param fdr: True or False. If fdr correction will be done for p-values.
#     :param threshold: 0-1. p-value or fdr cutoff to retain significant pairs. Default to 0.1.
#     :return: 'selected' column in global_res containing whether or not a pair should be retained
#     """
#     adata.uns['global_stat']['method'] = method
#     if method == 'z-score':
#         _p = adata.uns['global_res']['z_pval'].values
#     elif method == 'permutation':
#         _p = adata.uns['global_res']['perm_pval'].values
#     else:
#         raise ValueError("Only one of ['z-score', 'permutation'] is supported")
#     if fdr:
#         _p = fdrcorrection(_p)[1]
#         adata.uns['global_res']['fdr'] = _p
#     adata.uns['global_res']['selected'] = (_p < threshold)

# def spatialdm_local(adata, n_perm=1000, method='z-score', specified_ind=None,
#                     nproc=1, scale_X=True):
#     """
#         local spot selection
#     :param n_perm: number of times for shuffling neighbors partner for a given spot, default to 1000.
#     :param method: default to 'z-score' for computation efficiency.
#         Alternatively, can specify 'permutation' or 'both' (recommended for spot number < 1000, multiprocesing).
#     :param specified_ind: array containing queried indices in sample pair(s).
#     If not specified, local selection will be done for all sig pairs
#     :param nproc: default to 1.
#     :return: 'local_stat' & 'local_z_p' and/or 'local_perm_p' in adata.uns.
#     """
#     adata.uns['local_stat'] = {}
#     if (int(n_perm / nproc) != (n_perm / nproc)):
#         raise ValueError("n_perm should be divisible by nproc")
#     if type(specified_ind) == type(None):
#         specified_ind = adata.uns['global_res'][
#             adata.uns['global_res']['selected']].index  # default to global selected pairs
#     # total_len = len(specified_ind)
#     ligand = adata.uns['ligand'].loc[specified_ind]
#     receptor = adata.uns['receptor'].loc[specified_ind]
#     ind = ligand.index
#     adata.uns['local_stat']['local_I'] = np.zeros((adata.shape[0], len(ind)))
#     adata.uns['local_stat']['local_I_R'] = np.zeros((adata.shape[0], len(ind)))
#     N = adata.shape[0]
#     if method in ['both', 'permutation']:
#         adata.uns['local_stat']['local_permI'] = np.zeros((len(ind), n_perm, N))
#         adata.uns['local_stat']['local_permI_R'] = np.zeros((len(ind), n_perm, N))
#     if method in ['both', 'z-score']:
#         adata.uns['local_z'] = np.zeros((len(ind), adata.shape[0]))
#         adata.uns['local_z_p'] = np.zeros((len(ind), adata.shape[0]))

#     ## different approaches
#     with threadpool_limits(limits=nproc, user_api='blas'):
#         spot_selection_matrix(adata, ligand, receptor, ind, n_perm, method, scale_X)


# def sig_spots(adata, method='z-score', fdr=True, threshold=0.1):
#     """
#         pick significantly co-expressing spots
#     :param method: one of the methods from spatialdm_local, default to 'z-score'.
#     :param fdr: True or False, default to True
#     :param threshold: p-value or fdr cutoff to retain significant pairs. Default to 0.1.
#     :return:  1) 'selected_spots' in adata.uns: a binary frame of which spots being selected for each pair;
#      2) 'n_spots' in adata.uns['local_stat']: number of selected spots for each pair.
#     """
#     if method == 'z-score':
#         _p = adata.uns['local_z_p']
#     if method == 'permutation':
#         _p = adata.uns['local_perm_p']
#     if fdr:
#         _fdr = fdrcorrection(np.hstack(_p.values))[1].reshape(_p.shape)
#         _p.loc[:,:] = _fdr
#         adata.uns['local_stat']['local_fdr'] = _p
#     adata.uns['selected_spots'] = (_p < threshold)
#     adata.uns['local_stat']['n_spots'] = adata.uns['selected_spots'].sum(1)
#     adata.uns['local_stat']['local_method'] = method
#     return

# def drop_uns_na(adata, global_stat=False, local_stat=False):
#     adata.uns['geneInter'] = adata.uns['geneInter'].fillna('NA')
#     adata.uns['global_res'] = adata.uns['global_res'].fillna('NA')
#     adata.uns['ligand'] = adata.uns['ligand'].fillna('NA')
#     adata.uns['receptor'] = adata.uns['receptor'].fillna('NA')
#     adata.uns['local_stat']['n_spots'] = pd.DataFrame(adata.uns['local_stat']['n_spots'], columns=['n_spots'])
#     if global_stat and ('global_stat' in adata.uns.keys()):
#         adata.uns.pop('global_stat')
#     if local_stat and ('local_stat' in adata.uns.keys()):
#         adata.uns.pop('local_stat')

# def restore_uns_na(adata):
#     adata.uns['geneInter'] = adata.uns['geneInter'].replace('NA', np.nan)
#     adata.uns['global_res'] = adata.uns['global_res'].replace('NA', np.nan)
#     adata.uns['ligand'] = adata.uns['ligand'].replace('NA', np.nan)
#     adata.uns['receptor'] = adata.uns['receptor'].replace('NA', np.nan)
#     adata.uns['local_stat']['n_spots'] =  adata.uns['local_stat']['n_spots'].n_spots

# def write_spatialdm_h5ad(adata, filename=None):
#     if filename is None:
#         filename = 'spatialdm_out.h5ad'
#     elif not filename.endswith('h5ad'):
#         filename = filename+'.h5ad'
#     drop_uns_na(adata)
#     adata.write(filename)

# def read_spatialdm_h5ad(filename):
#     adata = ann.read_h5ad(filename)
#     restore_uns_na(adata)
#     return adata


