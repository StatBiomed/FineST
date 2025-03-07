import os
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


def topLRpairs(adata, spa_coexp_pair, num):
    """
    Get the top ligand-receptor pairs based on the global intensity from adata.

    Parameters:
    adata (AnnData): The annotated data matrix.
    num (int): The number of top pairs to return.

    Returns:
    TopLRpair (numpy.ndarray): The top ligand-receptor pairs.
    """
    TopLRpair=adata.uns['ligand'].index[np.argsort(np.log1p(adata.uns['global_I']))[(spa_coexp_pair.shape[0]-num):spa_coexp_pair.shape[0]]].tolist()
    
    return TopLRpair

#######################################
# 2024.11.28 Add code of LR-TF
# , datahost='builtin' -- !!
#######################################
def extract_tf(species, datahost='package'):
    """
    Find overlapping LRs from CellChatDB
    Parameters:
    species: support 'human', 'mouse' and 'zebrafish'
    datahost: the host of the ligand-receptor data. 
                'builtin' for package built-in otherwise from figshare
    Return: LR_TF (containing comprehensive info from CellChatDB) dataframe
    """
    
    if datahost == 'package':

        if species in ['mouse', 'human']:
            # datapath = '/mnt/lingyu/nfs_share2/Python/FineST/FineST/FineST/datasets/TF_data/%s-' %(species)
            # datapath = './datasets/TF_data/%s-' %(species)
            datapath = './FineST/datasets/TF_data/%s-' %(species)
        else:
            raise ValueError("Species type: {} is not supported currently. Please check.".format(species))
        
        filepath = datapath + 'TF_PPRhuman.rda'
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist!")
        
        LR_TF = pyreadr.read_r(filepath)['TF_PPRhuman']

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
        download_path = './FineST/datasets/TF_data/human-TF_PPRhuman.rda'
        
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
    Input DataFrame and two lists of ligands and receptors respectively, 
    filter the DataFrame based on the lists and return the top rows sorted by 'value' column.
    Parameters:
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
    ## Filter the DataFrame based on the ligand_list and receptor_list
    fData = tmp[tmp['Ligand'].isin(ligand_list) & tmp['Receptor'].isin(receptor_list)]
    print("Ligand and Receptor in R2TFdatabase:", fData.shape[0])
    ## Sort the DataFrame by 'value' column and return the top rows
    tmp_df = fData.sort_values(by='value', ascending=False).head(top_num)
    subdf = pd.DataFrame(tmp_df.rename(columns={"Ligand": "Ligand_symbol", 
                                                "Receptor": "Receptor_symbol", 
                                                "tf": "TF", "value": "value"}) )    
    
    return subdf


def pattern_LR2TF2TG(histology_results, pattern_num, R_TFdatabase, TF_TGdatabase):
    """
    Input DataFrame, checks if column 'g' contains '_', then splits 'g' column into two new rows
    parameters:
        histology_results : a DataFrame to process
        pattern_num : the pattern number to filter on, defaults to 0
        R_TFdatabase : the DataFrame containing receptor to TF mapping
    Returns:
        tmp : a DataFrame after processing
    """
    rows = []

    for i, row in histology_results.iterrows():
        if row['g'].count('_') == 2:
            gene1, gene2, gene3 = row['g'].split('_')

            ## create two new rows, gene1_gene2 and gene1_gene3 respectively
            new_row1 = row.copy()
            new_row1['g'] = gene1 + '_' + gene2
            new_row2 = row.copy()
            new_row2['g'] = gene1 + '_' + gene3
            rows.append(new_row1)
            rows.append(new_row2)
        else:
            ## if column 'g' does not contain two '_', add the original row to DataFrame directly
            rows.append(row)

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
    Input DataFrame, checks if column 'g' contains two '_', then splits the 'g' column value into two rows
    Parameters:
        histology_results : a DataFrame to process
        pattern_num : the pattern number to filter on, defaults to 0
        R_TFdatabase : the DataFrame containing receptor to TF mapping
    """
    rows = []

    for i, row in histology_results.iterrows():
        ## check if column 'g' contains two '_'
        if row['g'].count('_') == 2:
            ## split the 'g' column value into three parts
            gene1, gene2, gene3 = row['g'].split('_')

            ## create two new rows, gene1_gene2 and gene1_gene3 respectively
            new_row1 = row.copy()
            new_row1['g'] = gene1 + '_' + gene2
            new_row2 = row.copy()
            new_row2['g'] = gene1 + '_' + gene3

            ## add the new rows to the result DataFrame
            rows.append(new_row1)
            rows.append(new_row2)
        else:
            ## if column 'g' does not contain two '_', add the original row to the result DataFrame directly
            rows.append(row)

    ## after the loop, create DataFrame at once
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
    The unique gene of sig LR pairs
    Returns a DataFrame of unique elements from  'Ligand0', 'Ligand1', 'Receptor0', 
    'Receptor1', and 'Receptor2' columns of the DataFrame where 'selected' is True.
    """
    filtered_df = df[df['selected'] == True]
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
    geneInter_df = adata_impt_all.uns['geneInter']
    spa_coexp_pair = adata_impt_all.uns['global_res'].sort_values(by='fdr')  

    ## Merge the dataframes
    merged_df = pd.merge(spa_coexp_pair, geneInter_df, how='left', 
                         left_index=True, right_index=True)

    ## Keep only the columns of interest
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
    if issparse(X):
        rbf_d = X
        rbf_d[X.nonzero()] = np.exp(-X[X.nonzero()].A**2 / (2 * l ** 2))
    else:
        rbf_d = np.exp(- X**2 / (2 * l ** 2))
    
    ## At single-cell resolution, no within-spot communications
    if singlecell:

        ###################
        # old for SpatialDM
        ###################
        # np.fill_diagonal(rbf_d, 0)

        ###################
        # update v1
        ###################
        # rbf_d_dense = rbf_d.toarray()  # or rbf_d.todense()
        # np.fill_diagonal(rbf_d_dense, 0)

        ####################
        # update v2 25.01.31
        ####################
        # At single-cell resolution, no within-spot communications
        if singlecell:
            rbf_d.setdiag(0)    

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
    Paramwters:
        l: radial basis kernel parameter, need to be customized for optimal weight gradient and \
            to restrain the range of signaling before downstream processing.
        cutoff: (for secreted signaling) minimum weight to be kept from the rbf weight matrix, and \
            weight below cutoff will be made zero
        n_neighbors: (for secreted signaling) number of neighbors per spot from the rbf weight matrix.
        n_nearest_neighbors: (for adjacent signaling) number of neighbors per spot from rbf weight matrix. \
            Non-neighbors will be made 0
        ingle_cell: if single cell resolution, diagonal will be made 0.
    Return: secreted signaling weight matrix: adata.obsp['weight'], \
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
    Parameters:
        sample: spatialdm obj
        ls: a list of LR interaction names for the enrichment analysis
        path_name: str. For later recall sample.path_summary[path_name]
        dic: a dic of SpatialDE results (See tutorial)
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
            result_pattern.append((p_FET, modulename, module_size, overlap_size, overlap_features, n, query_set_size, negneg))  

    result = pd.DataFrame(result).set_index(1)
    result.columns = ['fisher_p', 'pathway_size', 'selected', 'selected_inters', 'name']

    result_pattern = pd.DataFrame(result_pattern).set_index(1)    # Convert result to dataframe, set the index to the second column
    result_pattern.columns = ['fisher_p', 'module_size', 'overlap_size', 'selected_inters', 'name', 'query_set_size', 'negneg']   
    
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



