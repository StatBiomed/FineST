import anndata
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree
import random
import numpy as np
from tqdm import tqdm
import torch
import time
from .utils import *


#######################################################################
## 2024.9.16 LLY add some function for Train and Test
#######################################################################
def extract_test_data(data_loader):
    reduced_expression_list = []
    reduced_image_list = []
    reduced_coord_list = []
    reduced_row_list = []
    reduced_col_list = []

    for batch in tqdm(data_loader):
        reduced_expression_list.append(batch['reduced_expression'])
        reduced_image_list.append(batch['image'])
        reduced_coord_list.append(batch['spatial_coords'])  
        reduced_row_list.append(batch['array_row'])  
        reduced_col_list.append(batch['array_col']) 

    print("***** batch_size=adata.shape[0] doen't effect *****")
    input_spot_test = torch.cat(reduced_expression_list)
    print(input_spot_test.shape)
    input_image_test = torch.cat(reduced_image_list)
    print(input_image_test.shape)

    input_coord_test = reduced_coord_list
    print(len(input_coord_test))
    input_row_test = reduced_row_list
    print(len(input_row_test))
    input_col_test = reduced_col_list
    print(len(input_col_test))
    print("***** *****")
    
    print("Finished extractting test data")    
    return input_spot_test, input_image_test, input_coord_test, input_row_test, input_col_test


#######################################################################
## 2024.9.16 LLY add some function for Load between spot data
#######################################################################
def extract_test_data_image_between_spot(data_loader):
    reduced_image_list = []
    reduced_coord_list = []

    for batch in tqdm(data_loader):
        reduced_image_list.append(batch['image'])
        reduced_coord_list.append(batch['spatial_coords'])  

    print("***** batch_size=adata.shape[0] *****")
    input_image_test = torch.cat(reduced_image_list)
    print(input_image_test.shape)

    input_coord_test = reduced_coord_list
    print(len(input_coord_test))
    print("***** *****")
    
    print("Finished extractting image_between_spot data")    
    return input_image_test, input_coord_test
    

def loadBatchData(train_image_mat, train_matrix_mat, train_coors_mat, batch_size, pos_info):
    '''
    Generate batch training data   
    '''
    
    train_pos_dist = pos_info['pos dist']
    train_pos_ind = pos_info['pos ind']
    
    train_index_list = list(range(train_image_mat.shape[0]))
    random.shuffle(train_index_list)

    
    train_data_size = train_image_mat.shape[0]

    half_batch_size =  int(batch_size/2)
    batch_num = train_data_size//half_batch_size
    
    for i in range(batch_num):
        
        start = i*half_batch_size
        end = start + half_batch_size
        
        tmp_index_list =  list(train_index_list[start:end])
       
        pos_peer_index = []

        neighbor_index = np.zeros((batch_size, batch_size))
        
        count = 0
        pos_index_list = []
        for j in tmp_index_list:
             
            cur_pos_peer_index = np.copy(train_pos_ind[j])
            ## shummin           
            # random.shuffle(cur_pos_peer_index)
            # pos_index_list.append(cur_pos_peer_index[0])
            
            ## when only select itself, adjust this
            # random.shuffle(cur_pos_peer_index)
            pos_index_list.append(cur_pos_peer_index)
            
            neighbor_index[count][half_batch_size+count] = 1 
            neighbor_index[half_batch_size+count][count] = 1
 
            count += 1
     
        tmp_index_list.extend(pos_index_list)
        cur_index_list = np.asarray(tmp_index_list)
        cur_batch_mat = np.take(train_image_mat.cpu(), cur_index_list, axis=0)
        cur_matrix_mat = np.take(train_matrix_mat.cpu(), cur_index_list, axis=0)
        
        yield cur_batch_mat, cur_matrix_mat, neighbor_index, cur_index_list        

    pass


#################################################################
# 2024.09.16 NameError: name 'checkNeighbors' is not defined
# 2024.11.02 Adjust parameters
#################################################################
def checkNeighbors(cur_adata, neighbor_k, tree_type='KDTree', leaf_size=2):
    '''
    parameter:
        tree_type: BallTree, cKDTree, KDTree (fast -> low)
        leaf_size: defalt 'leaf_size=2'
    Return 'dist' and 'ind' of positive samples.    
    '''
    print("checkNeighbors.............")
    
    cur_coor = np.column_stack((cur_adata.obs['array_row'].values, cur_adata.obs['array_col'].values))

    # start = time.time()
    if tree_type == 'BallTree':
        cur_coor_tree = BallTree(cur_coor, leaf_size=leaf_size)
    elif tree_type == 'cKDTree':
        cur_coor_tree = cKDTree(cur_coor, leafsize=leaf_size)
    elif tree_type == 'KDTree':
        cur_coor_tree = KDTree(cur_coor, leaf_size=leaf_size)
    else:
        raise ValueError('Invalid tree_type. Only "BallTree, KDTree and cKDTree" is supported.')
    # end = time.time()
    # print(f"Time to build KDTree: {end - start}")

    # start = time.time()
    location_dist, location_ind  = cur_coor_tree.query(cur_coor, k=(neighbor_k+1))
    # end = time.time()
    # print(f"Time to query KDTree: {end - start}")

    # Reshape the results to 2D if they are 1D
    if len(location_dist.shape) == 1:
        location_dist = location_dist[:, None]
    if len(location_ind.shape) == 1:
        location_ind = location_ind[:, None]

    ## Need to consider the selected location itself
    location_dist = location_dist[:,0]
    location_ind = location_ind[:,0]

    ## shumin
    # location_dist = location_dist[:,1:]
    # location_ind = location_ind[:,1:]
    
    return location_dist, location_ind



#################################################################
# 2024.9.16 NameError: name 'loadTrainTestData' is not defined
#################################################################
def loadTrainTestData(train_loader, neighbor_k, tree_type='KDTree', leaf_size=2, dataset_class='Visium'):
    
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    matrix_data = []
    image_data = []
    spatial_coords_list = []
    array_row_list = []
    array_col_list = []

    for batch in tqdm_object:
        # Load data
        matrix_data.append(batch["reduced_expression"].clone().detach().cuda())
        image_data.append(batch["image"].clone().detach().cuda())

        # Process each batch's spatial_coords
        spatial_coords = batch["spatial_coords"]
        combined_coords = torch.stack((spatial_coords[0], spatial_coords[1]), dim=1)
        spatial_coords_list.append(combined_coords)

        array_row = batch["array_row"]
        array_row_list.append(array_row)
        array_col = batch["array_col"]
        array_col_list.append(array_col)

    # Matrix data
    matrix_tensor = torch.cat(matrix_data).to(device)
    # Coord data
    spatial_coords_list_all = torch.cat(spatial_coords_list).to(device)
    array_row_list_all = torch.cat(array_row_list).to(device)
    array_col_list_all = torch.cat(array_col_list).to(device)
    # Image data
    image_tensor = torch.cat(image_data).to(device)
    print('image_tensor:', image_tensor.shape)
    image_tensor = image_tensor.view(image_tensor.shape[0] * image_tensor.shape[1], image_tensor.shape[2])
    inputdata_reshaped, latent_image_reshape = reshape_latent_image(image_tensor, dataset_class)
    latent_representation_image_arr = latent_image_reshape.cpu().detach().numpy()

    # Create adata_latent object
    adata_latent = anndata.AnnData(X=latent_representation_image_arr)
    adata_latent.obsm['spatial'] = np.array(spatial_coords_list_all.cpu())
    adata_latent.obs['array_row'] = np.array(array_row_list_all.cpu())
    adata_latent.obs['array_col'] = np.array(array_col_list_all.cpu())

    train_genes = adata_latent.var_names

    # Generate training data representation, training coordinate matrix, and positive sample information
    cur_train_data_mat = inputdata_reshaped
    cur_train_coors_mat = np.column_stack((adata_latent.obs['array_row'], adata_latent.obs['array_col']))
    cur_train_matrix_mat = matrix_tensor

    # Generate positive pair information
    pos_dist, pos_ind = checkNeighbors(adata_latent, neighbor_k, tree_type, leaf_size)
    cur_pos_info = {'pos dist': pos_dist, 'pos ind': pos_ind}

    return cur_train_data_mat, cur_train_matrix_mat, cur_train_coors_mat, cur_pos_info











