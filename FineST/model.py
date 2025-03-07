import torch
from torch import nn
import torch.nn.functional as F
from .utils import *
from .loadData import *
import os
import json
import numpy as np
from torch. utils.data import DataLoader, Dataset, TensorDataset
import glob
import logging
logging.getLogger().setLevel(logging.INFO)


#############################
# 2025.01.08 adjust for istar
#############################
def build_loaders(batch_size, image_embed_path, spatial_pos_path, reduced_mtx_path, 
                  image_clacss='HIPT', dataset_class='Visium'):
    setup_seed(666)

    ## set image_paths, according 'image_clacss'
    if image_clacss == 'istar':
    
        dataset = DatasetCreat_istar(
            image_paths=image_embed_path,
            spatial_pos_path=spatial_pos_path,
            reduced_mtx_path=reduced_mtx_path,   
            dataset_class=dataset_class
        )
    elif image_clacss == 'HIPT' or image_clacss == 'Virchow2':
        import glob
        image_paths = glob.glob(str(image_embed_path))
        image_paths.sort()

        dataset = DatasetCreat(
            image_paths=image_paths,
            spatial_pos_path=spatial_pos_path,
            reduced_mtx_path=reduced_mtx_path,   
            dataset_class=dataset_class
        )
    else:
        raise ValueError('Invalid image_class. Only "istar", "HIPT" and "Virchow2" are supported.')    

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("train/test split completed")
    print(len(train_dataset), len(test_dataset))

    ## set ‘pin_memory’, according 'dataset_class'
    if dataset_class == 'Visium16':
        pin_memory=False
    elif dataset_class == 'Visium64':
        pin_memory=False
    elif dataset_class == 'VisiumHD':
        pin_memory=False
    else:
        raise ValueError('Invalid dataset_class. Only "Visium16", "Visium64", and "VisiumHD" are supported.')

    ## Set up distributed sampler
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory, drop_last=True)
    print("***** Finished building loaders *****")
    return train_loader, test_loader


def build_loaders_inference(batch_size, image_embed_path, spatial_pos_path, reduced_mtx_path, 
                            image_clacss='HIPT', dataset_class='Visium'):
    setup_seed(666)

    ## set image_paths, according 'image_clacss'
    if image_clacss == 'istar':
    
        dataset = DatasetCreat_istar(
            image_paths=image_embed_path,
            spatial_pos_path=spatial_pos_path,
            reduced_mtx_path=reduced_mtx_path,   
            dataset_class=dataset_class
        )
    elif image_clacss == 'HIPT' or image_clacss == 'Virchow2':
        import glob
        image_paths = glob.glob(str(image_embed_path))
        image_paths.sort()

        dataset = DatasetCreat(
            image_paths=image_paths,
            spatial_pos_path=spatial_pos_path,
            reduced_mtx_path=reduced_mtx_path,   
            dataset_class=dataset_class
        )
    else:
        raise ValueError('Invalid image_class. Only "istar" and "HIPT" are supported.')  
    
    ## set ‘pin_memory’, according 'dataset_class'
    if dataset_class == 'Visium16':
        pin_memory=False
    elif dataset_class == 'Visium64':
        pin_memory=False
    elif dataset_class == 'VisiumHD':
        pin_memory=False
    else:
        raise ValueError('Invalid dataset_class. Only "Visium16", "Visium64", and "VisiumHD" are supported.')

    all_dataset = DataLoader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory, drop_last=False)
    print("***** Finished building loaders_inference *****")
    return all_dataset


def build_loaders_inference_allimage(batch_size, file_paths_spot, spatial_pos_path, 
                                     dataset_class='Visium16', file_paths_between_spot=None):
    setup_seed(666)

    if dataset_class == 'Visium16' or dataset_class == 'Visium64':
        if file_paths_between_spot is None:
            raise ValueError("file_paths_between_spot must be provided for Visium dataset")
        print("***** Building loaders_inference between spot *****")
        file_paths_spot = glob.glob(file_paths_spot)
        file_paths_between_spot = glob.glob(file_paths_between_spot)
        image_paths = file_paths_spot + file_paths_between_spot
        image_paths.sort()
    elif dataset_class == 'VisiumSC':
        print("***** Building loaders_inference sc image *****")
        image_paths = glob.glob(file_paths_spot)
        image_paths.sort()
    else:
        raise ValueError("Unknown dataset_class: {}".format(dataset_class))

    dataset = DatasetCreatImageBetweenSpot(
        image_paths=image_paths,
        spatial_pos_path=spatial_pos_path,
        dataset_class=dataset_class
    )
    
    all_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    # all_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    print("***** Finished building loaders_inference *****")
    return all_dataset



###############################################
# 2024.11.02 adjusted: add parameter： dataset
# 2025.01.08 design for istar image embeddings
###############################################
class DatasetCreat_istar(torch.utils.data.Dataset):
    def __init__(self, image_paths, spatial_pos_path, reduced_mtx_path, dataset_class):
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        ## Load expression matrix:（Transport to: cell x features）
        ## if exits colnames and rownames，use 'allow_pickle=True'
        self.reduced_matrix = np.load(reduced_mtx_path).T   
        self.image_tensor = torch.load(image_paths)
        
        ## set ‘split_num’, according 'dataset_class'
        if dataset_class == 'Visium16':
            self.split_num = 16
        elif dataset_class == 'Visium64':
            self.split_num = 64
        elif dataset_class == 'VisiumHD':
            self.split_num = 4
        else:
            raise ValueError('Invalid dataset_class. Only "Visium" and "VisiumHD" are supported.')
        print("Finished loading all files")

    def __getitem__(self, idx):
        item = {}
        v1 = self.spatial_pos_csv.loc[idx, 0]   
        v2 = self.spatial_pos_csv.loc[idx, 1]  
        v3 = self.spatial_pos_csv.loc[idx, 2]   
        v4 = self.spatial_pos_csv.loc[idx, 3]  
    
        ## Stack the tensors in the list along a new dimension  
        item['image'] = self.image_tensor[idx * self.split_num : (idx + 1) * self.split_num]  
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx, :]).float() 
        item['spatial_coords'] = [v1, v2]  
        item['array_row'] = v3   
        item['array_col'] = v4  
        return item

    def __len__(self):
        return len(self.reduced_matrix)
    

###############################################
# 2024.11.02 adjusted: add parameter： dataset
###############################################
class DatasetCreat(torch.utils.data.Dataset):
    def __init__(self, image_paths, spatial_pos_path, reduced_mtx_path, dataset_class):
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        ## Load expression matrix:（Transport to: cell x features）
        ## if exits colnames and rownames，use 'allow_pickle=True'
        self.reduced_matrix = np.load(reduced_mtx_path).T    
        
        ## Load .pth files
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
        elif dataset_class == 'VisiumHD':
            self.split_num = 4
        else:
            raise ValueError('Invalid dataset_class. Only "Visium" and "VisiumHD" are supported.')                
        print("Finished loading all files")

    def __getitem__(self, idx):
        item = {}
        v1 = self.spatial_pos_csv.loc[idx, 0]   
        v2 = self.spatial_pos_csv.loc[idx, 1]  
        v3 = self.spatial_pos_csv.loc[idx, 2]   
        v4 = self.spatial_pos_csv.loc[idx, 3]  
    
        ## Stack the tensors in the list along a new dimension  
        item['image'] = self.image_tensor[idx * self.split_num : (idx + 1) * self.split_num]    
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx, :]).float() 
        item['spatial_coords'] = [v1, v2]  
        item['array_row'] = v3   
        item['array_col'] = v4  
        return item

    def __len__(self):
        return len(self.reduced_matrix)
    

## 2023.10.26 need add CAE loss  
# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, embeddings, labels):

#         batch_size = embeddings.shape[0]
        
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         sim_matrix = torch.exp(torch.matmul(embeddings, embeddings.T) / self.temperature)
       
#         pos_mask = (labels == 1).type(torch.bool)

#         neg_mask = ~pos_mask
#         neg_mask.fill_diagonal_(False)  # remove diagonal items

#         pos_similarities = (sim_matrix * pos_mask).sum(dim=1)
#         neg_similarities = (sim_matrix * neg_mask).sum(dim=1)
        
#         loss = -torch.mean(torch.log(pos_similarities / neg_similarities))

#         return loss


###############################
# CoSimLoss Loss： used
###############################
def CoSimLoss(Y_hat: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    cos_by_col = nn.CosineSimilarity(dim=1)
    cos_by_row = nn.CosineSimilarity(dim=0)
    loss_col = (1 - cos_by_col(Y_hat, Y)).mean()
    loss_row = (1 - cos_by_row(Y_hat, Y)).mean()
    imp_loss = loss_col + loss_row
    return imp_loss

#######################################
# 2025.01.06  from Hist2ST
# ZINB_loss Loss： add
# https://github.com/biomed-AI/Hist2ST
#######################################
# def ZINBLoss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
#     eps = 1e-10
#     if isinstance(scale_factor,float):
#         scale_factor=np.full((len(mean),),scale_factor)
#     scale_factor = scale_factor[:, None]
#     mean = mean * scale_factor

#     t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
#     t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
#     nb_final = t1 + t2

#     nb_case = nb_final - torch.log(1.0-pi+eps)
#     zero_nb = torch.pow(disp/(disp+mean+eps), disp)
#     zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
#     result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

#     if ridge_lambda > 0:
#         ridge = ridge_lambda*torch.square(pi)
#         result += ridge
#     result = torch.mean(result)
#     return result


#######################################
# 2025.01.06  from Hist2ST
# ZINB_loss Loss： add
# https://github.com/inoue0426/scVGAE/
#######################################
def ZINBLoss(y_true, y_pred_mean, y_pred_disp, y_pred_pi, eps=1e-10):
    """
    Compute the ZINB Loss.
        y_true: Ground truth data.
        y_pred_mean: \mu Predicted mean from the model.
        y_pred_disp: \theta Dispersion parameter.
        y_pred_pi: \pi Zero-inflation probability.
        eps: Small constant to prevent log(0).
    """

    ## Negative Binomial Loss
    nb_terms = (
        -torch.lgamma(y_true + y_pred_disp)
        + torch.lgamma(y_true + 1)
        + torch.lgamma(y_pred_disp)
        - y_pred_disp * torch.log(y_pred_disp + eps)
        + y_pred_disp * torch.log(y_pred_disp + y_pred_mean + eps)
        - y_true * torch.log(y_pred_mean + y_pred_disp + eps)
        + y_true * torch.log(y_pred_mean + eps)
    )

    ## Zero-Inflation
    zero_inflated = torch.log(y_pred_pi + (1 - y_pred_pi) * torch.pow(1 + y_pred_mean / y_pred_disp, -y_pred_disp))

    result = -torch.sum(
        torch.log(y_pred_pi + (1 - y_pred_pi) * torch.pow(1 + y_pred_mean / y_pred_disp, -y_pred_disp))
        * (y_true < eps).float()
        + (1 - (y_true < eps).float()) * nb_terms
    )

    return torch.round(result)
    

###############################
# PearsonCorrelationLoss： v1
###############################
def compute_pearson_corr(x, y, dim):
    x_mean = torch.mean(x, dim=dim, keepdim=True)
    y_mean = torch.mean(y, dim=dim, keepdim=True)
    x_std = torch.std(x, dim=dim, keepdim=True)
    y_std = torch.std(y, dim=dim, keepdim=True)

    ## Compute covariance of x and y along the specified dimension
    cov = torch.mean((x - x_mean) * (y - y_mean), dim=dim, keepdim=True)
    pearson_corr = cov / (x_std * y_std)
    loss = (1 - pearson_corr) / 2
    return torch.mean(loss)

def PearsonCorrelationLoss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ## Compute Pearson correlation loss along rows
    row_loss = compute_pearson_corr(x, y, dim=1)
    ## Compute Pearson correlation loss along columns
    col_loss = compute_pearson_corr(x, y, dim=0)
    combined_loss = row_loss + col_loss
    return combined_loss


#####################################################################
# ContrastiveLoss Loss: used
# 2025.01.03: Adjist weight parameter
#####################################################################
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, w1=1, w2=1, w3=1, w4=1, w5=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        # self.w5 = w5

    def forward(
        self,
        embeddings_iamge,
        embeddings_matrix,
        labels,
        input_image_exp,
        reconstruction_iamge,
        reconstructed_matrix_all,
        reconstruction_iamge_reshapef2,
        input_matrix_exp,
        # recon_mat_mean, recon_mat_disp, recon_mat_pi
    ): 

        embeddings_iamge = F.normalize(embeddings_iamge, p=2, dim=1)
        embeddings_matrix = F.normalize(embeddings_matrix, p=2, dim=1)
        sim_matrix = torch.exp(torch.matmul(embeddings_iamge, embeddings_matrix.T) / self.temperature)
        
        pos_mask = (labels == 1).type(torch.bool)
        neg_mask = ~pos_mask

        pos_similarities = (sim_matrix * pos_mask).sum(dim=1)
        neg_similarities = (sim_matrix * neg_mask).sum(dim=1)

        #######################################################################
        ## 2023.12.20 adjust loss 

        # loss = (-self.w1*torch.mean(torch.log(pos_similarities / neg_similarities))            # contrast loss
        #         + self.w2*PearsonCorrelationLoss(reconstruction_iamge, input_image_exp)        # image loss
        #         + self.w3*PearsonCorrelationLoss(reconstructed_matrix_all, input_matrix_exp)   # matirx loss
        #         + self.w4*PearsonCorrelationLoss(reconstruction_iamge_reshapef2, input_matrix_exp))   # cross loss

        #######################################################################
        ## 2023.12.15 adjust loss 
        loss = (-self.w1*torch.mean(torch.log(pos_similarities / neg_similarities))     # contrast loss
                + self.w2*CoSimLoss(reconstruction_iamge, input_image_exp)              # image loss
                + self.w3*CoSimLoss(reconstructed_matrix_all, input_matrix_exp)         # matirx loss
                + self.w4*CoSimLoss(reconstruction_iamge_reshapef2, input_matrix_exp))  # cross loss
                # + self.w4*CoSimLoss(reconstruction_iamge_reshapef2, reconstructed_matrix_all))   # cross loss

        ##################################
        # 2025.01.06 add ZINB loss
        ##################################
        # - z_mean, z_dropout, z_dispersion: Outputs from the model, used for ZINB Loss calculation.
        # ZINBLoss(x_original, z_mean, z_dispersion, z_dropout)

        #######################################################################       
        ## 2023.12.15 adjust loss 
        # loss = (-self.w1*torch.mean(torch.log(pos_similarities / neg_similarities))       # contrast loss
        #         + self.w2*F.mse_loss(reconstruction_iamge, input_image_exp)               # image loss
        #         + self.w3*F.mse_loss(reconstructed_matrix_all, input_matrix_exp)          # matirx loss
        #         + self.w4*F.mse_loss(reconstruction_iamge_reshapef2, input_matrix_exp))   # cross loss
        #         # + self.w4*F.mse_loss(reconstruction_iamge_reshapef2, reconstructed_matrix_all)   # cross loss
        #         # + self.w5*ZINBLoss(input_matrix_exp, recon_mat_mean, recon_mat_disp, recon_mat_pi) ) # ZINB loss
        
        return loss
    

###################
# ProjectionHead  
###################
class ProjectionHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   
        # self.gelu = nn.GELU()  
        self.relu = nn.ReLU(inplace=True)  
        self.fc2 = nn.Linear(hidden_size, output_size)   

    def forward(self, x):
        x = self.fc1(x)  
        # x = self.gelu(x)   # BLEEP
        x = self.relu(x)  
        x = self.fc2(x)   
        return x
      
class ELU(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta
    def forward(self, x):
        return self.activation(x) + self.beta

##############################
# Spatial data embedding  
##############################
class Autoencoder_matrix(nn.Module):    # 2 layers
    def __init__(self, input_dim, latent_dim, n_encoder_hidden_matrix, negative_slope=0.01):
        super(Autoencoder_matrix, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_encoder_hidden_matrix),
            nn.LeakyReLU(negative_slope),    # GELU()  BLEEP
            nn.Linear(n_encoder_hidden_matrix, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_encoder_hidden_matrix),
            nn.LeakyReLU(negative_slope),
            nn.Linear(n_encoder_hidden_matrix, input_dim),
            # nn.Sigmoid(),
            # nn.Softplus(),     # xfuse   positive
            ELU(alpha=0.01, beta=0.01),    # iSTAR     positive
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


##############################
# 2025.01.06 Adjust, add ZINB
# Spatial data embedding  
##############################
# class Autoencoder_matrix(nn.Module):    # 2 layers
#     def __init__(self, input_dim, latent_dim, n_encoder_hidden_matrix, negative_slope=0.01):
#         super(Autoencoder_matrix, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, n_encoder_hidden_matrix),
#             nn.LeakyReLU(negative_slope),    # GELU()  BLEEP
#             nn.Linear(n_encoder_hidden_matrix, latent_dim)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, n_encoder_hidden_matrix),
#             nn.LeakyReLU(negative_slope),
#             nn.Linear(n_encoder_hidden_matrix, input_dim),
#             # nn.Sigmoid(),
#             # nn.Softplus(),     # xfuse   positive
#             ELU(alpha=0.01, beta=0.01),    # iSTAR     positive
#         )

#         self.decoder_mean = nn.Sequential(
#             nn.Linear(latent_dim, n_encoder_hidden_matrix),
#             nn.LeakyReLU(negative_slope),
#             nn.Linear(n_encoder_hidden_matrix, input_dim),
#             nn.Softplus()
#             # ELU(alpha=0.01, beta=0.01),    # iSTAR     positive
#         )
#         self.decoder_disp = nn.Sequential(
#             nn.Linear(latent_dim, n_encoder_hidden_matrix),
#             nn.LeakyReLU(negative_slope),
#             nn.Linear(n_encoder_hidden_matrix, input_dim),
#             nn.Softplus()
#             # ELU(alpha=0.01, beta=0.01),    # iSTAR     positive
#         )
#         self.decoder_pi = nn.Sequential(
#             nn.Linear(latent_dim, n_encoder_hidden_matrix),
#             nn.LeakyReLU(negative_slope),
#             nn.Linear(n_encoder_hidden_matrix, input_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)

#         decoded_mean = self.decoder_mean(encoded)
#         decoded_disp = self.decoder_disp(encoded)
#         decoded_pi = self.decoder_pi(encoded)

#         return decoded, decoded_mean, decoded_disp, decoded_pi
    

###############################
# Patch image embedding  
###############################
class Autoencoder_image(nn.Module):
    def __init__(self, input_dim, latent_dim, n_encoder_hidden_image, negative_slope=0.01):
        super(Autoencoder_image, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_encoder_hidden_image),
            nn.LeakyReLU(negative_slope),
            nn.Linear(n_encoder_hidden_image, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_encoder_hidden_image),
            nn.LeakyReLU(negative_slope),
            nn.Linear(n_encoder_hidden_image, input_dim),
            # nn.Softplus(),     # xfuse   positivate
            ELU(alpha=0.01, beta=0.01),    # iSTAR     positivate
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

########################
# FineSTModel    
########################
class FineSTModel(nn.Module):
    def __init__(
        self, 
        n_input_matrix=578, 
        n_input_image=384, 
        n_encoder_hidden_matrix=512,
        n_encoder_hidden_image=256,
        n_encoder_hidden=512,  # not use
        n_encoder_latent=128, 
        n_projection_hidden=256, 
        n_projection_output=128, 
        n_encoder_layers=2, 
        dropout_rate=0, 
        negative_slope=0.01
    ):    
        super(FineSTModel,self).__init__()     
        self.Autoencoder_matrix = Autoencoder_matrix(n_input_matrix, n_encoder_latent, n_encoder_hidden_matrix) 
        self.matrix_encoder = self.Autoencoder_matrix.encoder
        self.matrix_decoder = self.Autoencoder_matrix.decoder
        self.Autoencoder_image = Autoencoder_image(n_input_image, n_encoder_latent, n_encoder_hidden_image)  
        self.image_encoder = self.Autoencoder_image.encoder
        self.image_decoder = self.Autoencoder_image.decoder
        self.image_projection = ProjectionHead(n_encoder_latent, n_projection_hidden, n_projection_output)
        self.matrix_projection = ProjectionHead(n_encoder_latent, n_projection_hidden, n_projection_output)
                        
    def forward(self, x, y):              
        representation_matrix = self.matrix_encoder(x) 
        reconstruction_matrix = self.matrix_decoder(representation_matrix) 
        projection_matrix = self.matrix_projection(representation_matrix)       
        representation_image = self.image_encoder(y) 
        reconstruction_image = self.image_decoder(representation_image) 
        projection_image = self.image_projection(representation_image)      
        return (representation_matrix, reconstruction_matrix, projection_matrix, 
                representation_image, reconstruction_image, projection_image)


#################################
# 2025.01.06 adjust, add ZINB
# FineSTModel   
#################################
# class FineSTModel(nn.Module):
#     def __init__(
#         self, 
#         n_input_matrix=578, 
#         n_input_image=384, 
#         n_encoder_hidden_matrix=512,
#         n_encoder_hidden_image=256,
#         n_encoder_hidden=512,  # not use
#         n_encoder_latent=128, 
#         n_projection_hidden=256, 
#         n_projection_output=128, 
#         n_encoder_layers=2, 
#         dropout_rate=0, 
#         negative_slope=0.01
#     ):
#         super(FineSTModel,self).__init__()     
#         self.Autoencoder_matrix = Autoencoder_matrix(n_input_matrix, n_encoder_latent, n_encoder_hidden_matrix) 
#         self.matrix_encoder = self.Autoencoder_matrix.encoder
#         self.matrix_decoder = self.Autoencoder_matrix.decoder
#         self.matrix_decoder_mean = self.Autoencoder_matrix.decoder_mean
#         self.matrix_decoder_disp = self.Autoencoder_matrix.decoder_disp
#         self.matrix_decoder_pi = self.Autoencoder_matrix.decoder_pi
#         self.Autoencoder_image = Autoencoder_image(n_input_image, n_encoder_latent, n_encoder_hidden_image)  
#         self.image_encoder = self.Autoencoder_image.encoder
#         self.image_decoder = self.Autoencoder_image.decoder
#         self.image_projection = ProjectionHead(n_encoder_latent, n_projection_hidden, n_projection_output)
#         self.matrix_projection = ProjectionHead(n_encoder_latent, n_projection_hidden, n_projection_output)

#     def forward(self, x, y):              
#         representation_matrix = self.matrix_encoder(x) 
#         reconstruction_matrix = self.matrix_decoder(representation_matrix) 

#         recon_mat_mean = self.matrix_decoder_mean(representation_matrix) 
#         recon_mat_disp = self.matrix_decoder_disp(representation_matrix) 
#         recon_mat_pi = self.matrix_decoder_pi(representation_matrix) 

#         projection_matrix = self.matrix_projection(representation_matrix)       
#         representation_image = self.image_encoder(y) 
#         reconstruction_image = self.image_decoder(representation_image) 
#         projection_image = self.image_projection(representation_image) 

#         return (representation_matrix, 
#                 reconstruction_matrix, 
#                 projection_matrix, 
#                 representation_image, 
#                 reconstruction_image, 
#                 projection_image,
#                 recon_mat_mean, recon_mat_disp, recon_mat_pi)


def load_model(dir_name, parameter_file_path, params, gene_hv):    
    save_folder = os.path.join(dir_name, "epoch_"+str(params["training_epoch"])+".pt")
    if(dev=="cpu"):
        checkpoint = torch.load(save_folder,map_location="cpu")
    else:
        checkpoint = torch.load(save_folder)
    
    model_state_dict = checkpoint['model_state_dict']
    
    ## load parameter settings
    with open(parameter_file_path,"r") as json_file:
        params = json.load(json_file)
    params['n_input_matrix'] = len(gene_hv)
    # params['n_input_image'] = 384
    
    ## init the model
    model = FineSTModel(n_input_matrix=params['n_input_matrix'],
                              n_input_image=params['n_input_image'],
                              n_encoder_hidden_matrix=params["n_encoder_hidden_matrix"],
                              n_encoder_hidden_image=params["n_encoder_hidden_image"],
                              n_encoder_latent=params["n_encoder_latent"],
                              n_projection_hidden=params["n_projection_hidden"],
                              n_projection_output=params["n_projection_output"],
                              n_encoder_layers=params["n_encoder_layers"]).to(device)    
    ## load model states
    model.load_state_dict(model_state_dict)    
    return model
