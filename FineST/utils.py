import numpy as np
import random
import pandas as pd
import torch
import scanpy as sc
from anndata import AnnData
import logging
import os
## for configure_logging
import sys


###########################################################
# 2025.02.08 Form iStar
#            Found infer-smooth not at same scale, so norm
#            Normalized infer and smooth data, sepratelly
###########################################################
def scale(cnts):
    """
    First performs column-wise scaling and then applies a global max scaling.
    Parameters:
        cnts (numpy.ndarray): A two-dimensional count matrix.
    Returns:
        numpy.ndarray: The scaled count matrix.
    """

    cnts = cnts.astype(np.float64)  # Convert to float to avoid integer division issues

    # ## Calculate the minimum and maximum values for each column
    # cnts_min = cnts.min(axis=0)
    # cnts_max = cnts.max(axis=0)

    # ## Apply Min-Max normalization to each column
    # # cnts -= cnts_min
    # # cnts /= (cnts_max - cnts_min) + 1e-12  
    # ## Apply column-wise scaling & global scaling to [0, 1]
    # cnts /= (cnts_max - cnts_min) + 1e-12  # Adding a small constant to avoid division by zero
    
    cnts /= cnts.max()

    return cnts

###########################################################
# 2024.11.20 Form SpatialScope
#            created for StarDist_nuclei_segmente.py
###########################################################
def configure_logging(logger_name):
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name+'.log'
    importer_logger = logging.getLogger('importer_logger')
    importer_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    importer_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    importer_logger.addHandler(sh)
    return importer_logger


## set the random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


## set the logging
def setup_logger(model_save_folder):
        
    level =logging.INFO

    log_name = 'model.log'
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(model_save_folder + log_name)
    logger.setLevel(level)
    
    fileHandler = logging.FileHandler(os.path.join(model_save_folder, log_name), mode = 'a')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    return logger


## set the device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


## define function
def reshape_latent_image(inputdata, dataset_class='Visium64'):   

    ## set ‘split_num’, according 'dataset_class'
    if dataset_class == 'Visium16':
        split_num = 16
    elif dataset_class == 'Visium64':
        split_num = 64
    elif dataset_class == 'VisiumSC':
        split_num = 1
    elif dataset_class == 'VisiumHD':
        split_num = 4
    else:
        raise ValueError('Invalid dataset_class. Only "Visium16", "Visium64", "VisiumSC" and "VisiumHD" are supported.')            

    ## [adata.shape[0]*256, 384]  -->  [adata.shape[0], 384]
    inputdata_reshaped = inputdata.view(int(inputdata.shape[0]/split_num), 
                                        split_num, inputdata.shape[1]) # [adata.shape[0], 256, 384]
    average_inputdata_reshaped = torch.sum(inputdata_reshaped, dim=1) / inputdata_reshaped.size(1)
    return inputdata_reshaped, average_inputdata_reshaped



###############################################
# 2024.11.02 adjusted: add parameter： dataset
###############################################
class DatasetCreatImageBetweenSpot(torch.utils.data.Dataset):
    def __init__(self, image_paths, spatial_pos_path, dataset_class):
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        
        ## Load .pth file
        self.images = []
        for image_path in image_paths:
            if image_path.endswith('.pth'):
                image_tensor = torch.load(image_path)
                self.images.extend(image_tensor)
        self.image_data = torch.stack(self.images)
        self.image_tensor = self.image_data.view(self.image_data.size(0), -1)  

        ## set ‘split_num’, according 'dataset_class'
        if dataset_class == 'Visium16':
            self.split_num = 16
        elif dataset_class == 'Visium64':
            self.split_num = 64
        elif dataset_class == 'VisiumSC':
            self.split_num = 1
        elif dataset_class == 'VisiumHD':
            self.split_num = 4
        else:
            raise ValueError('Invalid dataset_class. Only "Visium" and "VisiumHD" are supported.')
                
        print("Finished loading all files")

    def __getitem__(self, idx):
        item = {}
        v1 = self.spatial_pos_csv.loc[idx, 0]   
        v2 = self.spatial_pos_csv.loc[idx, 1]  
    
        # Stack the tensors in the list along a new dimension  
        item['image'] = self.image_tensor[idx * self.split_num : (idx + 1) * self.split_num]    
        item['spatial_coords'] = [v1, v2]  

        return item

    def __len__(self):
        return len(self.spatial_pos_csv)
    

################################################
# 2025.01.16 adjust C2，according to split_num
# 2025.02.06 add 'obs' and 'obsm' to adata
################################################
def subspot_coord_expr_adata(recon_mat_reshape_tensor, adata, gene_hv, patch_size=56, 
                             p=None, q=None, dataset_class=None):
    ## Extract x, y coordinates based on the type of `adata`
    def get_x_y(adata, p):
        if isinstance(adata, AnnData):
            return adata.obsm['spatial'][p][0], adata.obsm['spatial'][p][1]
        else:
            return adata[p][0], adata[p][1]

    NN = recon_mat_reshape_tensor.shape[1]
    N = int(np.sqrt(NN))  # Determine the grid size
    ################
    # IMPORTANT
    ################
    pixel_step = patch_size / (2*N)  # Calculate the half of pixel step size
    print('pixel_step (half of patch_size):', pixel_step)
    all_spot_all_variable = np.zeros((recon_mat_reshape_tensor.shape[0] * recon_mat_reshape_tensor.shape[1], 
                                      recon_mat_reshape_tensor.shape[2]))
    C2 = np.zeros((recon_mat_reshape_tensor.shape[0] * recon_mat_reshape_tensor.shape[1], 2), dtype=int)
    first_spot_first_variable = None

    ## Set `split_num` according to `dataset_class`
    if dataset_class == 'Visium16':
        split_num = 16
    elif dataset_class == 'Visium64':
        split_num = 64
    elif dataset_class == 'VisiumSC':
        split_num = 1
    elif dataset_class == 'VisiumHD':
        split_num = 4
    else:
        raise ValueError('Invalid dataset_class. Only "Visium16", '
                 '"Visium64", "VisiumSC" and "VisiumHD" are supported.')

    if p is None and q is None:

        if split_num not in [1, 4, 16, 64]:
            raise ValueError("split_num must be 1, 4, 16, or 64")
        
        for p_ in range(recon_mat_reshape_tensor.shape[0]):
            x, y = get_x_y(adata, p_)
            C = np.zeros((NN, 2), dtype=int)

            ##############################
            ## 2025.01.06 old code
            ## from left-down to right-up
            ##############################
            for k in range(1, split_num + 1):
                s = k % N
                if s == 0:
                    i = N
                    j = k // N
                else:
                    i = s
                    j = (k - i) // N + 1

                if split_num == 4:
                    C[k - 1, 0] = x - pixel_step + (i - 1) * (2*pixel_step)
                    C[k - 1, 1] = y - pixel_step + (j - 1) * (2*pixel_step)
                elif split_num == 16:
                    C[k - 1, 0] = x - pixel_step - 1 * (2*pixel_step) + (i - 1) * (2*pixel_step)
                    C[k - 1, 1] = y - pixel_step - 1 * (2*pixel_step) + (j - 1) * (2*pixel_step)
                elif split_num == 64:
                    C[k - 1, 0] = x - pixel_step - 3 * (2*pixel_step) + (i - 1) * (2*pixel_step)
                    C[k - 1, 1] = y - pixel_step - 3 * (2*pixel_step) + (j - 1) * (2*pixel_step)
                elif split_num == 1:
                    C[k - 1, 0] = x
                    C[k - 1, 1] = y

            C2[p_ * split_num:(p_ + 1) * split_num, :] = C

        for q_ in range(recon_mat_reshape_tensor.shape[2]):
            all_spot_all_variable[:, q_] = recon_mat_reshape_tensor[:, :, q_].flatten().cpu().detach().numpy()

    else:
        x, y = get_x_y(adata, p)

        ## Select the information of the pth spot and the qth variable
        first_spot_first_variable = recon_mat_reshape_tensor[p, :, q].cpu().detach().numpy()

        ## Initialize C as a zero matrix of integer type
        C = np.zeros((NN, 2), dtype=int)

        #########################################
        ## 2025.01.06 adjust patch orgnization
        ## from left-up to right-down
        #########################################
        for k in range(1, split_num + 1):
            s = k % N
            if s == 0:
                i = N
                j = k // N
            else:
                i = s
                j = (k - i) // N + 1

            if split_num == 4:
                C[k - 1, 0] = x - pixel_step + (i - 1) * (2*pixel_step)
                C[k - 1, 1] = y - pixel_step + (j - 1) * (2*pixel_step)
            elif split_num == 16:
                C[k - 1, 0] = x - pixel_step - 1 * (2*pixel_step) + (i - 1) * (2*pixel_step)
                C[k - 1, 1] = y - pixel_step - 1 * (2*pixel_step) + (j - 1) * (2*pixel_step)
            elif split_num == 64:
                C[k - 1, 0] = x - pixel_step - 3 * (2*pixel_step) + (i - 1) * (2*pixel_step)
                C[k - 1, 1] = y - pixel_step - 3 * (2*pixel_step) + (j - 1) * (2*pixel_step)


    ## Establish new anndata in sub-spot level
    adata_spot = sc.AnnData(X=pd.DataFrame(all_spot_all_variable))
    adata_spot.var_names = gene_hv
    adata_spot.obs["x"] = C2[:, 0]
    adata_spot.obs["y"] = C2[:, 1]
    ## add other objects to adata
    adata_spot.obsm['spatial'] = adata_spot.obs[["x", "y"]].values
    # adata_spot.uns['spatial'] = adata.uns['spatial']
    
    return first_spot_first_variable, C, all_spot_all_variable, C2, adata_spot