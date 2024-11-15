import os
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch



# from FineST.datasets import dataset
# from FineST.utils import *        
# from FineST.loadData import * 
# from FineST.model import * 
# from FineST.train import * 
# from FineST.inference import * 
# from FineST.evaluation import * 
# from FineST.imputation import * 

print(torch.__version__)
# print("FineST version: %s" %fst.__version__)

## From GPU2
path = '/mnt/lingyu/nfs_share2/Python/'
os.chdir(str(path) + 'FineST/FineST/')

from FineST.datasets import dataset
from FineST.utils import *        
from FineST.loadData import * 
from FineST.model import * 
from FineST.train import * 
from FineST.inference import * 
from FineST.processData import *
from FineST.evaluation import * 
from FineST.downloadData import * 
from FineST.HIPT_image_feature_extract import * 
from FineST.SparseAEH import *
from FineST.SpatialDM import *
from FineST.plottings import *


setup_seed(666)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

import matplotlib.colors as clr
cnt_color = clr.LinearSegmentedColormap.from_list('magma', ["#000003",  "#3b0f6f",  "#8c2980",   "#f66e5b", "#fd9f6c", "#fbfcbf"], N=256)

##############################################################
# make logging and save model
##############################################################
import logging
from datetime import datetime
logging.getLogger().setLevel(logging.INFO)

model_folder = str(path) + 'FineST/FineST_local/Finetune/'
dir_name = model_folder + datetime.now().strftime('%Y%m%d%H%M%S%f')
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    logger = setup_logger(dir_name)

print(dir_name)

##############################################################
# load parameter settings
##############################################################

import json
parameter_file_path = str(path) + 'FineST/FineST/Parameter/parameters_NPC_P10125.json'
with open(parameter_file_path,"r") as json_file:
    params = json.load(json_file)

logger.info("Load parameters:\n" + json.dumps(params, indent=2))

## 1. Load Spatial and Image data

1.1 Load spatial data

adata = dataset.NPC()
print(adata)

1.2 Selected LR genes 

LR_gene_path = str(path)+'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv'
print("LR_gene_path:", LR_gene_path)

## Load LR 963 genes and extract their gene expression
adata = adata_LR(adata, LR_gene_path)
adata

1.3 Data preprocess

adata = adata_preprocess(adata)
print(adata)

gene_hv = np.array(adata.var_names)
print("gene_hv length:", len(gene_hv))

1.4 Reload adata with selected gene

matrix = adata2matrix(adata, gene_hv)
matrix

1.5 Order by image file name

## Load image embedding from HIPT
os.chdir(str(path) + 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/')
file_paths = sorted(os.listdir(str(path) + 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/'))
print("Image embedding file:", file_paths[:3])

df = get_image_coord(file_paths, dataset_class="Visium")
print(df.shape)
print(df.head())

## position
os.chdir(str(path)+'FineST/FineST/Dataset/NPC/patient1/')
position = pd.read_csv('tissue_positions_list.csv', header=None)
position = position.rename(columns={position.columns[-2]: 'pixel_x', position.columns[-1]: 'pixel_y'})
print(position.shape)
print(position.head())

## merge position
position_image = image_coord_merge(df, position, dataset = 'Visium')
spotID_order = np.array(position_image[0])
print(position_image.head())

1.6 Order matrix low, spatial corr by image file name

matrix_order, matrix_order_df = sort_matrix(matrix, position_image, spotID_order, gene_hv)
matrix_order_df

## save gene expression mateix
# np.save(str(path)+'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/harmony_matrix.npy', matrix_order_df.T)

adata = update_adata_coord(adata, matrix_order, position_image)
print(adata)

## save the original adata
# patientxy = 'patient1'
# adata.write_h5ad(str(path)+'FineST/FineST_local/Dataset/ImputData/'+str(patientxy)+'/'+str(patientxy)+'_adata_orignal.h5ad')

########################################
# save image position: only run once
########################################
position_order = pd.DataFrame({
    "pixel_y": position_image.loc[:, 'pixel_y'],
    "pixel_x": position_image.loc[:, 'pixel_x'],
    "array_row": position_image.loc[:, 'y'],
    "array_col": position_image.loc[:, 'x']
})
print(position_order)

## save the position data
# position_order.to_csv(str(path)+'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order.csv', index=False, header=False)

gene_selet = 'CD70'
fig, ax1 = plt.subplots(1, 1, figsize=(9, 7))
spatial_loc = adata.obsm['spatial']
scatter_plot = ax1.scatter(adata.obsm['spatial'][:, 0],adata.obsm['spatial'][:, 1], 
                           c=matrix_order_df[gene_selet], cmap=cnt_color, marker='h', s=22) 
ax1.invert_yaxis()
ax1.set_title(str(gene_selet)+' Expression')
fig.colorbar(scatter_plot, ax=ax1)
plt.gcf().set_dpi(100)
plt.show()

## 2 Train and Test model

2.1 Data loader and splitting

image_embed_path_NPC=str(path)+'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/*.pth'
spatial_pos_path = str(path)+'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order.csv'
reduced_mtx_path = str(path)+'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/harmony_matrix.npy'

train_loader, test_loader = build_loaders(batch_size=params['batch_size'], 
                                          image_embed_path=image_embed_path_NPC, 
                                          spatial_pos_path=spatial_pos_path, 
                                          reduced_mtx_path=reduced_mtx_path, dataset_class='Visium')

all_dataset = build_loaders_inference(batch_size=adata.shape[0], 
                                      image_embed_path=image_embed_path_NPC, 
                                      spatial_pos_path=spatial_pos_path, 
                                      reduced_mtx_path=reduced_mtx_path, dataset_class='Visium') 
input_spot_all, input_image_all, input_coord_all, input_row_all, input_col_all = extract_test_data(all_dataset)
print("input_spot_all:", input_spot_all.shape)
print("input_image_all:", input_image_all.shape)
print(input_image_all)
print(input_spot_all)

len(test_loader)

input_spot_test, input_image_test, input_coord_test, input_row_test, input_col_test = extract_test_data(test_loader)
print("input_spot_test:", input_spot_test.shape)
print("input_image_test:", input_image_test.shape)
# print(input_image_test)
# print(input_spot_test)

input_spot_test, input_image_test, input_coord_test, input_row_test, input_col_test = extract_test_data(test_loader)
print("input_spot_test:", input_spot_test.shape)
print("input_image_test:", input_image_test.shape)
print(input_image_test)
print(input_spot_test)

2.2 Train and test model on within spot

# ### load parameter settings
# with open(parameter_file_path,"r") as json_file:
#     params = json.load(json_file)

# ## add params
# params['n_input_matrix'] = len(gene_hv)
# params['n_input_image'] = 384

# ## init the model
# model = CellContrastModel(n_input_matrix=params['n_input_matrix'],
#                           n_input_image=params['n_input_image'],
#                           n_encoder_hidden_matrix=params["n_encoder_hidden_matrix"],
#                           n_encoder_hidden_image=params["n_encoder_hidden_image"],
#                           n_encoder_latent=params["n_encoder_latent"],
#                           n_projection_hidden=params["n_projection_hidden"],
#                           n_projection_output=params["n_projection_output"],
#                           n_encoder_layers=params["n_encoder_layers"]).to(device) 
# l = ContrastiveLoss(temperature=params['temperature'])
# # print(model)

# ## Set optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=params['inital_learning_rate'], momentum=0.9, weight_decay=5e-4)

# ## Load the data
# train_loader, test_loader = build_loaders(batch_size=params['batch_size'], 
#                                           image_embed_path=image_embed_path_NPC, 
#                                           dataset_class='Visium')


# # Train the model for a fixed number of epoch
# logger.info('Begin Training ...')

# start_time = time.time()

# best_loss = float('inf')
# best_epoch = 0
# for epoch in range(params['training_epoch']):
#     logger.info('epoch [{}/{}]'.format(epoch + 1, epoch))
#     print(f"Epoch: {epoch + 1}")
    
#     ######################################################################################
#     # Train the model 
#     ######################################################################################
#     model.train()
#     start_time = time.time()
#     train_loss = train_model(params, model, train_loader, optimizer, epoch, l, 
#                              tree_type='KDTree', leaf_size=2, dataset_class='Visium')  # LOSS
#     print("--- %s seconds ---" % (time.time() - start_time))
    
#     ######################################################################################
#     # Evaluate the model 
#     ######################################################################################
#     model.eval()
#     with torch.no_grad():
#         test_loss = test_model(params, model, test_loader, l, 
#                                tree_type='KDTree', leaf_size=2, dataset_class='Visium') 
    
#     if best_loss > test_loss:
#         best_loss = test_loss
#         best_epoch = epoch

#         save_model(model, dir_name, params, optimizer, train_loss)
#         # torch.save(model.state_dict(), "STFinetune/best.pt")    # BLEEP
#         print("Saved Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))
#         logger.info("Saved Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))

# print("Done!, final loss: {}".format(best_loss))
# print("Best epoch: {}".format(best_epoch))

# print("--- %s seconds ---" % (time.time() - start_time))

# logger.info("Done!, Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))

# logger.info('Finished Training')


## 3. Inference, Imputation and Evaluation on within spot

3.1 Inference: within spot

dir_name

## You can use the 'dir_name' just obtained from the above
## but here we use the trained 'dir_name' before, for paper results repeated

dir_name = str(path)+'FineST/FineST_local/Finetune/20240125140443830148'  
parameter_file_path = str(path)+'FineST/FineST_local/Parameter/parameters_NPC_P10125.json'

# dir_name = '/mnt/lingyu/nfs_share2/Python/NPC/Data/stdata/STFinetune/20240125140443830148'    # 2023.01.25
# parameter_file_path = str(path) + 'cVAE/NPC1py/parameters_16_shuminLRgeneP10125.json'

#####################################################################################  
# main
#####################################################################################

# load params
with open(parameter_file_path,"r") as json_file:
    params = json.load(json_file)

# load models
model = load_model(dir_name, parameter_file_path, params, gene_hv)   
model.to(device)
# print("model", model)

# load all data
test_loader = build_loaders_inference(batch_size=adata.shape[0], 
                                      image_embed_path=image_embed_path_NPC, 
                                      spatial_pos_path=spatial_pos_path, 
                                      reduced_mtx_path=reduced_mtx_path,
                                      dataset_class='Visium')


# inference
logger.info("Running inference tesk...")

start_time = time.time()

(matrix_profile, 
reconstructed_matrix, 
recon_ref_adata_image_f2, 
representation_image_reshape,
representation_matrix,
projection_image_reshape,
projection_matrix,
input_image_exp,
reconstruction_iamge,
reconstructed_matrix_reshaped,
input_coord_all) = perform_inference_image(model, test_loader, dataset_class='Visium')

print("--- %s seconds ---" % (time.time() - start_time))

# print
print(matrix_profile.shape)
print(reconstructed_matrix.shape)
print(recon_ref_adata_image_f2.shape)
print("print(reconstructed_matrix_reshaped.shape)", reconstructed_matrix_reshaped.shape)
print(matrix_profile)
print(reconstructed_matrix)
print(recon_ref_adata_image_f2)
print(input_coord_all)

logger.info("Running inference tesk DONE!")

3.2 Inference: All subspot location

3.2.1 All subspot location

print(reconstructed_matrix_reshaped.shape)
print(reconstructed_matrix_reshaped.shape[0]/16)
reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(reconstructed_matrix_reshaped, dataset_class='Visium')

print(reconstructed_matrix_reshaped_tensor.shape)
print(reconstructed_matrix_reshaped_tensor[0, :, :].shape)
print(reconstructed_matrix_reshaped_tensor[:, :, 0].flatten().shape)

adata.obsm['spatial']

input_coord_all





p = 0
x = spatial_loc[p][0]
y = spatial_loc[p][1]
print(x,y)


NN = reconstructed_matrix_reshaped_tensor.shape[1]
N = int(np.sqrt(NN))
p = 0
x = spatial_loc[p][0]
y = spatial_loc[p][1]


# 初始化为整数类型的零矩阵
C = np.zeros((N**2, 2), dtype=int)

for k in range(1, N**2 + 1):
    s = k % N

    if s == 0:
        i = N
        j = k // N
    else:
        i = s
        j = (k - i) // N + 1

    ## 64-16
    # C[k - 1, 0] = x - 7 - 7 * 14 + (i - 1) * 14
    # C[k - 1, 1] = y - 7 - 7 * 14 + (j - 1) * 14
    ## 64-4  -- h=64/4=16 --  x-(h/2)-(4/2-1)*h
    C[k - 1, 0] = x - 8 - 1 * 16 + (i - 1) * 16
    C[k - 1, 1] = y - 8 - 1 * 16 + (j - 1) * 16

print(C.shape)

# Select the information of the first spot and the first variable
first_spot_first_variable = reconstructed_matrix_reshaped_tensor[0, :, 0].cpu().detach().numpy()
print(first_spot_first_variable.shape)
print(reconstructed_matrix_reshaped_tensor.shape)
print(reconstructed_matrix_reshaped_tensor[0, :, 0])

fig, ax = plt.subplots(figsize=(2.5, 2.5))
scatter3 = ax.scatter(C[:, 0], C[:, 1], c=first_spot_first_variable, marker='o', cmap=cnt_color, s=1800)
ax.invert_yaxis()
ax.set_title("First spot")
plt.show()

NN = reconstructed_matrix_reshaped_tensor.shape[1]
N = int(np.sqrt(NN))

# Initialize variables all_spot_all_variable and C2
all_spot_all_variable = np.zeros((reconstructed_matrix_reshaped_tensor.shape[0]*reconstructed_matrix_reshaped_tensor.shape[1], reconstructed_matrix_reshaped_tensor.shape[2]))
C2 = np.zeros((reconstructed_matrix_reshaped_tensor.shape[0] * reconstructed_matrix_reshaped_tensor.shape[1], 2), dtype=int)

## get the sub spot coord
for p in range(reconstructed_matrix_reshaped_tensor.shape[0]):
    # Calculate the coordinates of the 256 positions of the pth spot
    x = spatial_loc[p][0]
    y = spatial_loc[p][1]

    C = np.zeros((N**2, 2), dtype=int)

    for k in range(1, N**2 + 1):
        s = k % N

        if s == 0:
            i = N
            j = k // N
        else:
            i = s
            j = (k - i) // N + 1

        ## 224
        # C[k - 1, 0] = x - 7 - 7 * 14 + (i - 1) * 14
        # C[k - 1, 1] = y - 7 - 7 * 14 + (j - 1) * 14
        ## 64  -- h=64/4=16 --  x-(h/2) - (4/2-1)*h
        C[k - 1, 0] = x - 8 - 1 * 16 + (i - 1) * 16
        C[k - 1, 1] = y - 8 - 1 * 16 + (j - 1) * 16

    # Add the coordinates of the 16 positions of the pth spot to C2
    C2[p * 16:(p + 1) * 16, :] = C
print(C2.shape)

## get the sub-spot gene expression
for q in range(reconstructed_matrix_reshaped_tensor.shape[2]):
    # Extract the information of the first variable of the pth spot
    all_spot_all_variable[:, q] = reconstructed_matrix_reshaped_tensor[:, :, q].flatten().cpu().detach().numpy()    # the first variable
print(all_spot_all_variable.shape)

# Establish new anndata in sub spot level
data = pd.DataFrame(all_spot_all_variable)
adata_spot = sc.AnnData(X=data)
adata_spot.var_names = gene_hv
adata_spot.obs["x"] = C2[:, 0]
adata_spot.obs["y"] = C2[:, 1]
print(adata_spot)
print(adata_spot.obs["x"])
print(adata_spot.var_names)

3.2.2 Neighbours of all within spot

adata_know, sudo_adata, adata_spot = prepare_impute_adata(adata, adata_spot, C2, gene_hv)
print(adata_know)
print(sudo_adata)

start_time = time.time()

nearest_points = find_nearest_point(adata_spot.obsm['spatial'], adata_know.obsm['spatial'])    
nbs, nbs_indices = find_nearest_neighbors(nearest_points, adata_know.obsm['spatial'], k=6)
distances = calculate_euclidean_distances(adata_spot.obsm['spatial'], nbs)

# Iterate over each point in sudo_adata
for i in range(sudo_adata.shape[0]):
    dis_tmp = (distances[i] + 0.1) / np.min(distances[i] + 0.1)
    k = 1
    weights = ((1 / (dis_tmp ** k)) / ((1 / (dis_tmp ** k)).sum()))
    sudo_adata.X[i, :] = np.dot(weights, adata_know.X[nbs_indices[i]].todense())

print("--- %s seconds ---" % (time.time() - start_time))

print(adata.X.A)
print(adata_spot.X)
print(sudo_adata.X)

3.2.3 Add two inference data with w1 and w2

## imputed results
print(sudo_adata.X.min(), sudo_adata.X.max())
## inferred results
print(adata_spot.X.min(), adata_spot.X.max())

print(sudo_adata)
print(adata_spot)

############################################
## using the weight and add
## adata_spot: inferred results
## sudo_adata: imputed results
############################################

w1 = 0.5
weight_impt_data = w1*adata_spot.X + (1-w1)*sudo_adata.X
adata_impt = sc.AnnData(X = pd.DataFrame(weight_impt_data))

adata_impt.var_names = gene_hv
adata_impt.obs = adata_spot.obs
print(adata_impt)
print(adata_impt.obs['x'])

print(reconstructed_matrix_reshaped.shape)
data_impt = torch.tensor(weight_impt_data)
print(data_impt.shape)
print(data_impt[:10])

_, data_impt_reshape = reshape_latent_image(data_impt, dataset_class='Visium')
print(data_impt_reshape.shape)
print(data_impt_reshape[:10])

3.3 Visualization: selected gene

def plot_gene_data_scale(spatial_loc, genedata, title, ax):
    normalized_data = genedata
    normalized_data = (genedata - genedata.min()) / (genedata.max() - genedata.min())
    scatter = ax.scatter(spatial_loc[:,0], spatial_loc[:,1], c=normalized_data, cmap=cnt_color)   
    ax.invert_yaxis()
    ax.set_title(title)
    return scatter

##########################################################################################
fig, axes = plt.subplots(1, 2, figsize=(22, 8))
gene = "CD70"  

# Orignal test data
orignal_matrix = pd.DataFrame(matrix_profile)
orignal_matrix.columns = gene_hv
genedata1 = orignal_matrix[[gene]].to_numpy()
scatter1 = plot_gene_data_scale(spatial_loc, genedata1, str(gene)+" Expression: Orignal", axes[0])

## Imputed test data
imputed_matrix_test_exp = pd.DataFrame(data_impt_reshape)
imputed_matrix_test_exp.columns = gene_hv
print(imputed_matrix_test_exp.shape)
genedata2 = imputed_matrix_test_exp[[gene]].to_numpy()
scatter2 = plot_gene_data_scale(spatial_loc, genedata2, str(gene)+" Expression: Imputation", axes[1])

# ## Reconstructed_f2 test data
# reconstruction_f2_reshape_pd = pd.DataFrame(recon_ref_adata_image_f2)
# reconstruction_f2_reshape_pd.columns = gene_hv
# print(reconstruction_f2_reshape_pd.shape)
# genedata3 = reconstruction_f2_reshape_pd[[gene]].to_numpy()
# scatter3 = plot_gene_data_scale(spatial_loc, genedata3, str(gene)+" Expression: Reconstructed F2", axes[2])

fig.colorbar(scatter1, ax=axes.ravel().tolist())
plt.show()

3.4 Correlation: selected gene

gene = "CD70"
title = "Predicted Expression"

genedata1 = orignal_matrix[[gene]].to_numpy()
genedata2 = imputed_matrix_test_exp[[gene]].to_numpy()

plot_correlation(genedata1, genedata2, 'Reconstructed Expression', gene, size=5)

3.5 Correlation: Spot

def calculate_statistics_spot(matrix1, matrix2, label):
    print(matrix1.shape)
    print(matrix2.shape)

    ## pearson
    mean_pearson_corr = calculate_correlation_infer(matrix1, matrix2, method="pearson", sample="spot")
    logger.info(f'Mean Pearson correlation coefficient--{label}: {mean_pearson_corr}')

    ## spearman
    mean_spearman_corr = calculate_correlation_infer(matrix1, matrix2, method="spearman", sample="spot")
    logger.info(f'Mean Spearman correlation coefficient--{label}: {mean_spearman_corr}')

    ## cosine_similarity_row
    cosine_sim = calculate_cosine_similarity_row(matrix1, matrix2)
    cosine_sim_per_sample = np.diag(cosine_sim)
    average_cosine_similarity = np.mean(cosine_sim_per_sample)
    logger.info(f'Average cosine similarity--{label}: {average_cosine_similarity}')

## Correlation -- adata.shape[0] sample
logger.info("Running correlation task...")

## reconstructed by f2 (recon_ref_adata_image_f2)
calculate_statistics_spot(matrix_profile, np.array(imputed_matrix_test_exp), 'reconf2')

logger.info("Running correlation task DINE!")

3.6 Correlation: Gene

logger.info("Running Gene Correlation task...")

def calculate_statistics_gene(matrix1, matrix2, label):
    print(matrix1.shape)
    print(matrix2.shape)

    mean_pearson_corr = calculate_correlation_infer(matrix1, matrix2, method="pearson", sample="gene")
    # print(f"Mean Pearson correlation coefficient--{label}: {mean_pearson_corr:.4f}")
    logger.info(f'Mean Pearson correlation coefficient--{label}: {mean_pearson_corr}')
    
    mean_spearman_corr = calculate_correlation_infer(matrix1, matrix2, method="spearman", sample="gene")
    # print(f"Mean Spearman correlation coefficient--{label}: {mean_spearman_corr:.4f}")
    logger.info(f'Mean Spearman correlation coefficient--{label}: {mean_spearman_corr}')

    cosine_sim = calculate_cosine_similarity_col(matrix1, matrix2)
    cosine_sim_per_sample = np.diag(cosine_sim)
    average_cosine_similarity = np.mean(cosine_sim_per_sample)
    
    # print(f"Average cosine similarity--{label}: {average_cosine_similarity:.4f}")
    logger.info(f'Average cosine similarity--{label}: {average_cosine_similarity}')

## reconstructed by f2 
####################################
calculate_statistics_gene(matrix_profile, np.array(imputed_matrix_test_exp), 'reconf2')

logger.info("Running Gene Correlation task DINE!")

3.7 Correlation: Spot and Gene 

plot_correlations(matrix_profile, data_impt_reshape)
# plot_correlations(matrix_profile, data_impt_reshape, save_pdf=True, file_name="correlation_plot.pdf")

## save adata
# patientxy = 'patient1'
# adata_impt.write_h5ad(str(path)+'FineST/FineST/Dataset/ImputData/'+str(patientxy)+'/'+str(patientxy)+'_adata.h5ad')

# print(orignal_matrix.shape)
# print(imputed_matrix_test_exp.shape)
# print(reconstruction_f2_reshape_pd_betwn.shape)

# ## add data in spot
# data_spot_pixel = pd.concat([imputed_matrix_test_exp, reconstruction_f2_reshape_pd_betwn], axis=0, ignore_index=True)
# print("data_spot_pixel shape:", data_spot_pixel.shape)

# ## add coord in pixel
# cord_spot_pixel = pd.concat([pd.DataFrame(spatial_loc), pd.DataFrame(spatial_loc_add)], axis=0, ignore_index=False)
# cord_spot_pixel.columns = ["x", "y"]
# print("cord_spot_pixel shape:", cord_spot_pixel.shape)

# ## plot
# gene = 'CD70'
# fig, ax1 = plt.subplots(1, 1, figsize=(9, 7))
# plot = ax1.scatter(cord_spot_pixel['x'], cord_spot_pixel['y'], c=data_spot_pixel[gene], cmap=cnt_color, s=1)   
# ax1.set_title(str(gene) + ' Expression: Pixel ALL')
# cbar = fig.colorbar(plot, ax=ax1)
# ax1.invert_yaxis()
# # plt.savefig(str(gene)+"P1_pixel.pdf", format="pdf")
# plt.gcf().set_dpi(50)
# plt.show()

## 4. Inference model for "within spot" and "between spot" 

4.1 Load trained model parameters

dir_name = str(path)+'FineST/FineST_local/Finetune/20240125140443830148'  
parameter_file_path = str(path)+'FineST/FineST/Parameter/parameters_NPC_P10125.json'

4.2 Load within spot & between spot image feature

#############################
# Get file paths
#############################
## add coords for each .pth file
file_paths_spot = os.listdir(str(path) + 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/')
print(len(file_paths_spot))
file_paths_between_spot = os.listdir(str(path) + 'NPC/Data/stdata/ZhuoLiang/LLYtest/NEW_AH_Patient1_pth_64_16/')
print(len(file_paths_between_spot))
## Merge, sort and process file paths
file_paths_all = file_paths_spot + file_paths_between_spot
print(len(file_paths_all))

#########################################################
# Merge, sort and process file paths
#########################################################
data_all = get_image_coord_all(file_paths_all, dataset_class='Visium')
df = pd.DataFrame(data_all, columns=['Part_3', 'Part_4'])
df = df.rename(columns={'Part_3': 'pixel_y', 'Part_4': 'pixel_x'})[['pixel_x', 'pixel_y']]
print(df)     

position_order_between_spot = pd.DataFrame({
    "pixel_y": df.loc[:, 'pixel_y'],
    "pixel_x": df.loc[:, 'pixel_x']
})

## save all spots
# position_order_between_spot.to_csv(str(path)+"FineST/FineST/Dataset/NPC/ContrastP1geneLR/position_order_all.csv", index=False, header=False)

4.3 Load within & between spot gene expression data

import glob
# file_paths_spot = str(path)+'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/*.pth'
file_paths_between_spot = str(path) + 'NPC/Data/stdata/ZhuoLiang/LLYtest/NEW_AH_Patient1_pth_64_16/*.pth'
spatial_pos_path=str(path)+'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order_all.csv'

all_dataset = build_loaders_inference_allimage(batch_size=len(file_paths_all), 
                                               file_paths_spot=image_embed_path_NPC, 
                                               file_paths_between_spot=file_paths_between_spot, 
                                               spatial_pos_path=spatial_pos_path, 
                                               dataset_class='Visium')
print("all_dataset:\n", all_dataset)

input_image_all, input_coord_all = extract_test_data_image_between_spot(all_dataset)
print("input_image_all:", input_image_all.shape)
print("input_coord_all:", len(input_coord_all))
# print(input_image_all)
# print(input_coord_all)

4.4 Load the trained model to infer all spots

#####################################################################################  
# main
#####################################################################################

# load params
with open(parameter_file_path,"r") as json_file:
    params = json.load(json_file)

# load models
model = load_model(dir_name, parameter_file_path, params, gene_hv)   
model.to(device)
# print("model", model)

## load between spots data
test_loader = build_loaders_inference_allimage(batch_size=len(file_paths_all),
                                               file_paths_spot=image_embed_path_NPC, 
                                               file_paths_between_spot=file_paths_between_spot, 
                                               spatial_pos_path=spatial_pos_path, 
                                               dataset_class='Visium')

## inference
logger.info("Running inference tesk between spot...")

start_time = time.time()

(recon_ref_adata_image_f2, 
reconstructed_matrix_reshaped,
representation_image_reshape_between_spot,
input_image_exp_between_spot,
input_coord_all) = perform_inference_image_between_spot(model, test_loader)

print("--- %s seconds ---" % (time.time() - start_time))

## print
print("recon_ref_adata_image_f2:", recon_ref_adata_image_f2.shape)
# print("recon_ref_adata_image_f2:\n", recon_ref_adata_image_f2)
# print("input_coord_all:\n", input_coord_all)

logger.info("Running inference tesk between spot DONE!")

#####################################################################################  
# main
#####################################################################################

# load params
with open(parameter_file_path,"r") as json_file:
    params = json.load(json_file)

# load models
model = load_model(dir_name, parameter_file_path, params, gene_hv)   
model.to(device)
# print("model", model)

## load between spots data
test_loader = build_loaders_inference_allimage(batch_size=len(file_paths_all),
                                               file_paths_spot=image_embed_path_NPC, 
                                               file_paths_between_spot=file_paths_between_spot, 
                                               spatial_pos_path=spatial_pos_path, 
                                               dataset_class='Visium')

## inference
logger.info("Running inference tesk between spot...")

start_time = time.time()

(recon_ref_adata_image_f2, 
reconstructed_matrix_reshaped,
representation_image_reshape_between_spot,
input_image_exp_between_spot,
input_coord_all) = perform_inference_image_between_spot(model, test_loader)

print("--- %s seconds ---" % (time.time() - start_time))

## print
print("recon_ref_adata_image_f2:", recon_ref_adata_image_f2.shape)
# print("recon_ref_adata_image_f2:\n", recon_ref_adata_image_f2)
# print("input_coord_all:\n", input_coord_all)

logger.info("Running inference tesk between spot DONE!")

4.5 Visualization all spots

gene = "CD70"
## Reconstructed_f2 test data
reconstruction_f2_reshape_pd_all = pd.DataFrame(recon_ref_adata_image_f2)
reconstruction_f2_reshape_pd_all.columns = gene_hv
genedata3 = reconstruction_f2_reshape_pd_all[[gene]].to_numpy()
print(genedata3.shape)

reconstructed_matrix_test_arr = recon_ref_adata_image_f2.T
print(reconstructed_matrix_test_arr.shape)  

def process_and_check_duplicates(input_coord_all):
    
    tensor_1 = input_coord_all[0][0]
    tensor_2 = input_coord_all[0][1]

    input_coord_all_concat = torch.stack((tensor_1, tensor_2))
    spatial_loc = input_coord_all_concat.T.numpy()

    # Find unique rows and their counts
    unique_rows, counts = np.unique(spatial_loc, axis=0, return_counts=True)
    # Check if there are any duplicate rows
    duplicate_rows = (counts > 1).any()
    print("Are there any duplicate rows? :", duplicate_rows)
    return spatial_loc
    
spatial_loc_all = process_and_check_duplicates(input_coord_all)
print(spatial_loc_all)

def plot_gene_data_dot(spatial_loc, genedata, title, ax):
    normalized_data = genedata
    # normalized_data = (genedata - genedata.min()) / (genedata.max() - genedata.min())
    scatter = ax.scatter(spatial_loc[:,0], spatial_loc[:,1], c=normalized_data, cmap=cnt_color, s=4)   
    ax.invert_yaxis()
    ax.set_title(title)
    return scatter

##########################################################################################
gene = "CD70"

fig, ax = plt.subplots(figsize=(9, 7))
genedata3 = reconstruction_f2_reshape_pd_all[[gene]].to_numpy()
print("genedata3: ", genedata3.shape)
scatter3 = plot_gene_data_dot(spatial_loc_all, genedata3, str(gene)+' Expression: all subspot', ax) 
fig.colorbar(scatter3, ax=ax)
plt.show()

4.6 Visualization all sub-spots

print(reconstructed_matrix_reshaped.shape)
print(reconstructed_matrix_reshaped.shape[0]/16)
reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(reconstructed_matrix_reshaped)
print(reconstructed_matrix_reshaped_tensor.shape)


################################################################################################
# Given a gene name, find the column index of the gene in the DataFrame
################################################################################################
gene = 'CD70'
column_index = matrix.columns.get_loc(gene)
column_index


def calculate_coordinates(spot_index, N, x, y):
    ## used for all subspot: torch.Size([19732, 16, 1543])
    C = np.zeros((N**2, 2), dtype=int)
    for k in range(1, N**2 + 1):
        s = k % N
        if s == 0:
            i = N
            j = k // N
        else:
            i = s
            j = (k - i) // N + 1
        C[k - 1, 0] = x - 8 - 1 * 16 + (i - 1) * 16
        C[k - 1, 1] = y - 8 - 1 * 16 + (j - 1) * 16

        ## 224
        # C[k - 1, 0] = x - 7 - 7 * 14 + (i - 1) * 14
        # C[k - 1, 1] = y - 7 - 7 * 14 + (j - 1) * 14
        ## 64  -- h=64/4=16 --  x-(h/2) - (4/2-1)*h
        C[k - 1, 0] = x - 8 - 1 * 16 + (i - 1) * 16
        C[k - 1, 1] = y - 8 - 1 * 16 + (j - 1) * 16
    return C


num_variables = int(reconstructed_matrix_reshaped.shape[0]/16)
num_coordinates = int(reconstructed_matrix_reshaped.shape[0])
all_spot_all_variable = np.zeros((num_variables, 16))
C2 = np.zeros((num_coordinates, 2), dtype=int)


NN = reconstructed_matrix_reshaped_tensor.shape[1]
N = int(np.sqrt(NN))

for p in range(num_variables):
    spot_first_variable = reconstructed_matrix_reshaped_tensor[p, :, column_index].cpu().detach().numpy()
    all_spot_all_variable[p, :] = spot_first_variable

    x = spatial_loc_all[p][0]
    y = spatial_loc_all[p][1]
    C = calculate_coordinates(p, N, x, y)

    C2[p * 16:(p + 1) * 16, :] = C

print(C2.shape)
print(C2)

fig, ax1 = plt.subplots(1, 1, figsize=(12.5, 10.5))
plot = ax1.scatter(C2[:, 0], C2[:, 1], c=all_spot_all_variable.flatten(), cmap=cnt_color, s=0.5)
ax1.set_title(str(gene) + ' Expression: Reconstructed_f2 Pixel')
cbar = fig.colorbar(plot, ax=ax1)
ax1.invert_yaxis()
# plt.savefig("P1_pixel.pdf", format="pdf")
plt.gcf().set_dpi(100)
plt.show()

4.7 All subspt location

print(reconstructed_matrix_reshaped_tensor.shape)
print(reconstructed_matrix_reshaped_tensor[0, :, :].shape)
print(reconstructed_matrix_reshaped_tensor[:, :, 0].flatten().shape)

# Initialize variables all_spot_all_variable and C2
all_spot_all_variable_all = np.zeros((reconstructed_matrix_reshaped_tensor.shape[0]*reconstructed_matrix_reshaped_tensor.shape[1], 
                                      reconstructed_matrix_reshaped_tensor.shape[2]))

# get the sub-spot gene expression
for q in range(reconstructed_matrix_reshaped_tensor.shape[2]):
    # Extract the information of the first variable of the pth spot
    all_spot_all_variable_all[:, q] = reconstructed_matrix_reshaped_tensor[:, :, q].flatten().cpu().detach().numpy()    # the first variable
print(all_spot_all_variable_all.shape)

# Create AnnData object
adata_spot_all = sc.AnnData(X=pd.DataFrame(all_spot_all_variable_all))
adata_spot_all.var_names = gene_hv
adata_spot_all.obs["x"], adata_spot_all.obs["y"] = C2[:, 0], C2[:, 1]

print(adata_spot_all)
# print(adata_spot_all.obs["x"])
# print(adata_spot_all.var_names)

gene = "CD70"

adata_spot_all.obs[gene]=adata_spot_all.X[:,adata_spot_all.var.index==gene]
print(adata_spot_all.obs[gene])
fig=sc.pl.scatter(adata_spot_all,alpha=1,x="x",y="y",color=gene,color_map=cnt_color,show=False,size=10) 
fig.set_aspect('equal', 'box')
fig.invert_yaxis()
plt.gcf().set_dpi(100)
fig.figure.show()

## 5. Imputation using measured spot expressiom

5.1 Neighbours of all within spot

adata_know, sudo_adata_all, adata_spot_all = prepare_impute_adata(adata, adata_spot_all, C2, gene_hv)
print(sudo_adata_all)

nearest_points = find_nearest_point(adata_spot_all.obsm['spatial'], adata_know.obsm['spatial'])    
nbs, nbs_indices = find_nearest_neighbors(nearest_points, adata_know.obsm['spatial'])
distances = calculate_euclidean_distances(adata_spot_all.obsm['spatial'], nbs)

## Iterate over each point in sudo_adata_all
start_time = time.time()
for i in range(sudo_adata_all.shape[0]):
    dis_tmp = (distances[i] + 0.1) / np.min(distances[i] + 0.1)
    k = 1
    weights = ((1 / (dis_tmp ** k)) / ((1 / (dis_tmp ** k)).sum()))
    sudo_adata_all.X[i, :] = np.dot(weights, adata_know.X[nbs_indices[i]].todense())

print("--- %s seconds ---" % (time.time() - start_time))

print(adata.X.A)
print(adata_spot_all.X)
print(sudo_adata_all.X)

## save two adata
# patientxy = 'Patient1'
# adata_spot_all.write_h5ad(str(path)+'FineST/FineST/Dataset/ImputData/'+str(patientxy)+'/'+str(patientxy)+'_adata_spot_all.h5ad')
# sudo_adata_all.write_h5ad(str(path)+'FineST/FineST/Dataset/ImputData/'+str(patientxy)+'/'+str(patientxy)+'_sudo_adata_all.h5ad')

## Load two data
# patientxy = 'Patient1'
# os.chdir(str(path)+'FineST/FineST/Dataset/ImputData/'+str(patientxy)+'/')
# adata_spot_all = sc.read_h5ad(filename=str(patientxy)+'_adata_spot_all.h5ad')
# sudo_adata_all = sc.read_h5ad(filename=str(patientxy)+'_sudo_adata_all.h5ad')

5.2 Add inference and impution data

print(sudo_adata_all)
print(adata_spot_all)
# print(sudo_adata_all.obs['x'])
# print(adata_spot_all.obs['x'])

# using the weight and add
w1 = 0.5
weight_impt_data_all = w1*adata_spot_all.X + (1-w1)*sudo_adata_all.X
adata_impt_all = sc.AnnData(X = pd.DataFrame(weight_impt_data_all))

adata_impt_all.var_names = gene_hv
adata_impt_all.obs = adata_spot_all.obs
# print("adata_impt_all: ", adata_impt_all)
# print(adata_impt_all.obs['x'])

weight_impt_data_all_tensor = torch.tensor(weight_impt_data_all)
print(weight_impt_data_all_tensor.shape)
print(weight_impt_data_all_tensor[:10])

_, adata_impt_all_reshape = reshape_latent_image(weight_impt_data_all_tensor)
print(adata_impt_all_reshape.shape)
print(adata_impt_all_reshape[:10])

5.3 save the imputation spot data

adata_impt_spot = sc.AnnData(X = pd.DataFrame(adata_impt_all_reshape.cpu().detach().numpy()))
adata_impt_spot.var_names = gene_hv
adata_impt_spot.obs['x'] = spatial_loc_all[:,0]
adata_impt_spot.obs['y'] = spatial_loc_all[:,1]

print("adata_impt_spot: ", adata_impt_spot)
print(adata_impt_spot.obs['x'])

## save data: 5039 × 596
# patientxy = 'patient1'
# os.chdir(str(path)+'FineST/FineST/Dataset/ImputData/'+str(patientxy)+'/')
# adata_impt_spot.write_h5ad(filename=str(patientxy)+'_adata_all_spot.h5ad')

5.4 Visualization: gene at all spot

def plot_gene_data_single(spatial_loc, genedata, title, ax):
    normalized_data = genedata
    # normalized_data = (genedata - genedata.min()) / (genedata.max() - genedata.min())
    scatter = ax.scatter(spatial_loc[:,0], spatial_loc[:,1], c=normalized_data, cmap=cnt_color, s=8)    
    ax.invert_yaxis()
    ax.set_title(title)
    return scatter

#################################################################################################################
gene = "CD70"

## Imputed all subspot data
imputed_matrix_test_exp_all = pd.DataFrame(adata_impt_all_reshape)
imputed_matrix_test_exp_all.columns = gene_hv
genedata2 = imputed_matrix_test_exp_all[[gene]].to_numpy()
print("genedata2:", genedata2.shape)
fig, ax = plt.subplots(figsize=(9, 7))
scatter2 = plot_gene_data_single(spatial_loc_all, genedata2, str(gene)+" Expression: all subspot Imputation",  ax)
fig.colorbar(scatter2, ax=ax)

plt.gcf().set_dpi(100)
# plt.savefig(str(gene)+"_Expression_Imputation_allsubspot.pdf", format="pdf")

plt.show()

5.5 Visualization: gene at all sub-spot

gene = "CD70"

adata_spot_all.obs[gene]=adata_impt_all.X[:,adata_impt_all.var.index==gene]
print(adata_spot_all.obs[gene])
fig=sc.pl.scatter(adata_impt_all,alpha=1,x="x",y="y",color=gene,color_map=cnt_color,show=False,size=6) 
fig.set_aspect('equal', 'box')
fig.invert_yaxis()
plt.gcf().set_dpi(300)
fig.figure.show()

##  6. Save all subspot imupted data

print(adata_impt_all)

## save adata: 80624 × 596
# patientxy = 'patient1'
# os.chdir(str(path)+'FineST/FineST/Dataset/ImputData/'+str(patientxy)+'/')
# adata_impt_all.write_h5ad(filename=str(patientxy)+'_adata_all.h5ad')

## End





