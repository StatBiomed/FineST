import logging
logging.getLogger().setLevel(logging.INFO)
from .utils import *
from .loadData import *
import numpy as np
from scipy.spatial import cKDTree


def perform_inference_image(model, test_loader, dataset_class='Visium64'):
    print("device",device)    

    ########################
    # for whole dataset
    ########################    
    print("***** Begin perform_inference: ******")
    
    input_spot_all, input_image_all, input_coord_all, _, _ = extract_test_data(test_loader)
            
    ## input image and matrix
    matrix_profile = input_spot_all.to(device)
    image_profile = input_image_all.to(device)
    ## reshape image
    image_profile_reshape = image_profile.view(-1, image_profile.shape[2])  # [1331, 256, 384] --> [1331*256, 384]
    input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
    
    ## useful model
    representation_matrix = model.matrix_encoder(matrix_profile)
    reconstructed_matrix = model.matrix_decoder(representation_matrix)
    projection_matrix = model.matrix_projection(representation_matrix)  
    representation_image = model.image_encoder(input_image_exp) 
    reconstruction_iamge = model.image_decoder(representation_image)
    projection_image = model.image_projection(representation_image)

    ## reshape
    _, representation_image_reshape = reshape_latent_image(representation_image, dataset_class)
    _, projection_image_reshape = reshape_latent_image(projection_image, dataset_class)

    ## cross decoder
    reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
    _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped, dataset_class)
    
    ########################  
    # convert
    ######################## 
    ## matrix
    matrix_profile = matrix_profile.cpu().detach().numpy() 
    reconstructed_matrix = reconstructed_matrix.cpu().detach().numpy() 
    reconstruction_iamge_reshapef2 = reconstruction_iamge_reshapef2.cpu().detach().numpy() 
    ## latent space
    representation_image_reshape = representation_image_reshape.cpu().detach().numpy() 
    representation_matrix = representation_matrix.cpu().detach().numpy() 
    ## latent space -- peojection
    projection_image_reshape = projection_image_reshape.cpu().detach().numpy() 
    projection_matrix = projection_matrix.cpu().detach().numpy() 
    ## image
    input_image_exp = input_image_exp.cpu().detach().numpy() 
    reconstruction_iamge = reconstruction_iamge.cpu().detach().numpy()
    # ## tensor recon_f2
    # reconstructed_matrix_reshaped = reconstructed_matrix_reshaped.cpu().detach().numpy()
    
    return (matrix_profile, 
            reconstructed_matrix, 
            reconstruction_iamge_reshapef2, 
            representation_image_reshape,
            representation_matrix,
            projection_image_reshape,
            projection_matrix,
            input_image_exp,
            reconstruction_iamge,
            reconstructed_matrix_reshaped,
            input_coord_all)


def perform_inference_image_between_spot(model, test_loader, dataset_class='Visium'):
    print("device",device)    

    ########################
    # for whole dataset
    ########################    
    print("***** Begin perform_inference: ******")
    
    input_image_all, input_coord_all = extract_test_data_image_between_spot(test_loader)   
            
    ## input image
    image_profile = input_image_all.to(device)
    ## reshape image
    image_profile_reshape = image_profile.view(-1, image_profile.shape[2])  # [adata.shape[0], 256, 384] --> [adata.shape[0]*256, 384]
    input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
    ## useful model
    representation_image = model.image_encoder(input_image_exp) 
    ## cross decoder
    reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
    _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped, dataset_class)
    ## reshape
    _, representation_image_reshape = reshape_latent_image(representation_image, dataset_class)
    
    ######################## 
    # convert
    ########################  
    ## matrix
    representation_image_reshape = representation_image_reshape.cpu().detach().numpy() 
    reconstruction_iamge_reshapef2 = reconstruction_iamge_reshapef2.cpu().detach().numpy() 
    
    return (reconstruction_iamge_reshapef2, 
            reconstructed_matrix_reshaped,
            representation_image_reshape,
            input_image_exp,
            input_coord_all)


########################
# imputation.py
########################
def find_nearest_point(adata_spot, adata_know):
    ## Find the nearest point in adata_know for each point in adata_spot
    nearest_points = []
    for point in adata_spot:
        distances = np.linalg.norm(adata_know - point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_points.append(adata_know[nearest_index])
    return np.array(nearest_points)

##################################################################################
# using 6 neighborhood： cKDTree is more faster
# Find k nearest neighbors for each point in nearest_points within adata_know
##################################################################################

## Function 1: Using cKDTree
def find_nearest_neighbors(nearest_points, adata_know, k=6):
    nbs = []
    nbs_indices = []
    tree = cKDTree(adata_know)
    for point in nearest_points:
        dist, indices = tree.query(point, k+1)
        nbs.append(adata_know[indices])
        nbs_indices.append(indices)
    return np.array(nbs), np.array(nbs_indices)

# ## Function 2: Using NearestNeighbors
# from sklearn.neighbors import NearestNeighbors
# def find_nearest_neighbors(nearest_points, adata_know, k=6):
#     nbs = []
#     nbs_indices = []
#     nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='euclidean').fit(adata_know)
#     for point in nearest_points:
#         distances, indices = nbrs.kneighbors([point])
#         nbs.append(adata_know[indices][0])
#         nbs_indices.append(indices[0])
#     return np.array(nbs), np.array(nbs_indices)

## Calculate Euclidean distances between each point in adata_spot and its nearest neighbors
def calculate_euclidean_distances(adata_spot, nbs):
    distances = []
    for point, neighbors in zip(adata_spot, nbs):
        dist = np.linalg.norm(neighbors - point, axis=1)
        distances.append(dist)
    return np.array(distances)


#################################################
# 2025.01.24: add the infer model im main code
#################################################
def infer_model_fst(model, test_loader, logger, 
                    dataset_class='Visium64'): 
    model.to(device)

    logger.info("Running inference task...")
    start_infer_time = time.time()
    
    (matrix_profile, reconstructed_matrix, recon_ref_adata_image_f2, 
    _, _, _, _, _, _, reconstructed_matrix_reshaped,
    input_coord_all) = perform_inference_image(model, test_loader, dataset_class=dataset_class)
    
    print("--- %s seconds for inference within spots ---" % (time.time() - start_infer_time))
    print("Reconstructed_matrix_reshaped shape: ", reconstructed_matrix_reshaped.shape)
    logger.info("Running inference task DONE!")
    
    return (matrix_profile, reconstructed_matrix, recon_ref_adata_image_f2, 
            reconstructed_matrix_reshaped, input_coord_all)