import os
import time
import warnings
import numpy as np
import torch
import logging
from datetime import datetime
import json
import argparse
import pandas as pd
print("torch version: %s" % torch.__version__)

from FineST.utils import *
from FineST import datasets
from FineST.processData import *
from FineST.model import *
from FineST.train import *
from FineST.plottings import *
from FineST.inference import *

##################
# Basic setting
##################
warnings.filterwarnings('ignore')
setup_seed(666)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    return True

def load_and_process_data(args):
    
    adata = datasets.NPC()
    print(" **** Load the original NPC patient1 adata: **** \n", adata)
    adata = adata_LR(adata, args.LRgene_path)
    adata = adata_preprocess(adata)
    print(" **** Processed NPC patient1 adata: **** \n", adata)
    gene_hv = np.array(adata.var_names)
    print(" **** The length of LR genes: ", len(gene_hv))

    matrix = adata2matrix(adata, gene_hv)
    file_paths = sorted(os.listdir(os.path.join(args.system_path, args.image_embed_path)))
    print(" **** Image embedding file (First 3): **** \n", file_paths[:3])

    position_image = get_image_coord(file_paths, dataset_class=args.dataset_class)
    position = pd.read_csv(os.path.join(args.system_path, args.visium_path), header=None)
    position = position.rename(columns={position.columns[-2]: 'pixel_x', position.columns[-1]: 'pixel_y'})
    position_image = image_coord_merge(position_image, position, dataset_class=args.dataset_class)
    position_order = update_st_coord(position_image)
    print(" **** The coords of image patch: **** \n", position_order.shape)
    print(position_order.head())
    position_order.to_csv(os.path.join(args.system_path, args.spatial_pos_path), index=False, header=False)

    spotID_order = np.array(position_image[0])
    matrix_order, matrix_order_df = sort_matrix(matrix, position_image, spotID_order, gene_hv)
    np.save(os.path.join(args.system_path, args.reduced_mtx_path), matrix_order_df.T)

    adata = update_adata_coord(adata, matrix_order, position_image)
    gene_expr(adata, matrix_order_df, gene_selet=args.gene_selected, 
              save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_orig_gene_expr.pdf'))

    return adata, gene_hv, matrix_order_df

def setup_logging(args):
    logging.getLogger().setLevel(logging.INFO)

    weight_path = os.path.join(args.system_path, args.weight_path)
    dir_name = weight_path + datetime.now().strftime('%Y%m%d%H%M%S%f')

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logger = setup_logger(dir_name)
    print("dir_name: \n", dir_name)

    parame_path = os.path.join(args.system_path, args.parame_path)
    with open(parame_path, "r") as json_file:
        params = json.load(json_file)
    logger.info("Load parameters:\n" + json.dumps(params, indent=2))

    return logger, parame_path, params, dir_name

def train_model_fst(params, model, train_loader, test_loader, optimizer, l, dir_name, logger):
    logger.info('Begin Training ...')
    start_train_time = time.time()

    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(params['training_epoch']):
        logger.info('epoch [{}/{}]'.format(epoch + 1, params['training_epoch']))
        print(f"Epoch: {epoch + 1}")

        # Train the model
        model.train()
        start_time = time.time()
        train_loss = train_model(params, model, train_loader, optimizer, epoch, l,
                                 tree_type='KDTree', leaf_size=2, dataset_class=args.dataset_class)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = test_model(params, model, test_loader, l,
                                   tree_type='KDTree', leaf_size=2, dataset_class=args.dataset_class)

        if best_loss > test_loss:
            best_loss = test_loss
            best_epoch = epoch

            save_model(model, dir_name, params, optimizer, train_loss)
            print("Saved Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))
            logger.info("Saved Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))
    print("--- %s seconds ---" % (time.time() - start_train_time))
    logger.info("Done!, Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))
    logger.info('Finished Training')

    return dir_name

def infer_gene_expr(model, adata, args, gene_hv, logger):
    model.to(device)
    test_loader = build_loaders_inference(batch_size=adata.shape[0], 
                                          image_embed_path=os.path.join(args.system_path,
                                                                        args.image_embed_path,'*.pth'),
                                          spatial_pos_path=args.spatial_pos_path,
                                          reduced_mtx_path=args.reduced_mtx_path,
                                          dataset_class=args.dataset_class)

    logger.info("Running inference task...")

    start_infer_time = time.time()

    (_, _, _, _, _, _, _, _, _, 
     reconstructed_matrix_reshaped, _) = perform_inference_image(model, test_loader, 
                                                                 dataset_class=args.dataset_class)

    print("--- %s seconds for inference within spots ---" % (time.time() - start_infer_time))
    print("Reconstructed_matrix_reshaped shape: ", reconstructed_matrix_reshaped.shape)

    logger.info("Running inference task DONE!")

    reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(reconstructed_matrix_reshaped, 
                                                                   dataset_class=args.dataset_class)
    print(" **** The size of reconstructed tensor data:", reconstructed_matrix_reshaped_tensor.shape)

    (first_spot_first_variable, C, 
     _, _, _) = subspot_coord_expr_adata(reconstructed_matrix_reshaped_tensor,
                                         adata, gene_hv, p=0, q=0, 
                                         dataset_class=args.dataset_class)
    (_, _, 
     all_spot_all_variable, C2, adata_infer) = subspot_coord_expr_adata(reconstructed_matrix_reshaped_tensor,
                                                                        adata, gene_hv, 
                                                                        dataset_class=args.dataset_class)
    print(" **** All_spot_all_variable shape:", all_spot_all_variable.shape)


    return adata_infer, first_spot_first_variable, C, C2

def main(args):
    # Check if required files exist
    required_files = [
        args.LRgene_path, 
        os.path.join(args.system_path, args.visium_path),
        os.path.join(args.system_path, args.image_embed_path),
        os.path.join(args.system_path, args.parame_path)
    ]
    
    for file_path in required_files:
        if not check_file_exists(file_path):
            return

    # Load and process data
    adata, gene_hv, matrix_order_df = load_and_process_data(args)

    # Setup logging
    logger, parame_path, params, dir_name = setup_logging(args)

    # Initialize the model
    params['n_input_matrix'] = len(gene_hv)
    params['n_input_image'] = 384

    model = FineSTModel(
        n_input_matrix=params['n_input_matrix'],
        n_input_image=params['n_input_image'],
        n_encoder_hidden_matrix=params["n_encoder_hidden_matrix"],
        n_encoder_hidden_image=params["n_encoder_hidden_image"],
        n_encoder_latent=params["n_encoder_latent"],
        n_projection_hidden=params["n_projection_hidden"],
        n_projection_output=params["n_projection_output"],
        n_encoder_layers=params["n_encoder_layers"]
    ).to(device)

    l = ContrastiveLoss(temperature=params['temperature'])

    # Set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=params['inital_learning_rate'], 
                                momentum=0.9, weight_decay=5e-4)

    # Load the data
    train_loader, test_loader = build_loaders(
        batch_size=params['batch_size'],
        image_embed_path=os.path.join(args.system_path, args.image_embed_path, '*.pth'),
        spatial_pos_path=args.spatial_pos_path,
        reduced_mtx_path=args.reduced_mtx_path,
        dataset_class=args.dataset_class
    )

    # Train the model if no pre-trained weights are provided
    if args.weight_save_path is None:
        dir_name = train_model_fst(params, model, train_loader, test_loader, optimizer, l, dir_name, logger)
    else:
        dir_name = args.weight_save_path

    # Load the trained model
    model = load_model(dir_name, parame_path, params, gene_hv)

    # Perform inference
    adata_infer, first_spot_first_variable, C, C2 = infer_gene_expr(model, adata, args, gene_hv, logger)

    ######################
    # Impute gene expr.
    ######################
    adata_imput = impute_adata(adata, adata_infer, C2, gene_hv, k=6)
    _, data_impt = weight_adata(adata_infer, adata_imput, gene_hv, w=0.5)
    _, data_impt_reshape = reshape_latent_image(data_impt, dataset_class=args.dataset_class)
    print(" **** data_impt shape:", data_impt.shape)
    print(" **** data_impt_reshape shape:", data_impt_reshape.shape)

    ###################################
    # Evaluate predicted gene expr.
    ###################################
    # subspot_expr(C, first_spot_first_variable, 
    #             save_path=os.path.join(args.figure_save_path, '1st_spot_1st_gene.pdf'))
    # logger.info("Running first_spot_first_variable plot DINE!")

    # gene_expr_compare(adata, args.gene_selected, data_impt_reshape, gene_hv, s=50, 
    #                     save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_pred_gene_expr.pdf'))
    # logger.info("Running gene_expr_compare plot DINE!")

    sele_gene_cor(adata, data_impt_reshape, gene_hv, gene = args.gene_selected, 
                        ylabel='FineST Expression', title = str(args.gene_selected)+' expression', size=5, 
                        save_path=os.path.join(args.figure_save_path, str(args.gene_selected)+'_gene_corr.pdf'))    
    logger.info("Running sele_gene_cor plot DINE!")

    logger.info("Running Gene Correlation task...")
    (pearson_cor_gene, 
    spearman_cor_gene, 
    cosine_sim_gene) = mean_cor(adata, data_impt_reshape, 'reconf2', sample="gene")
    logger.info("Pearson, Spearman, Cosine corr_gene: [{}: {}: {}]".format(pearson_cor_gene, spearman_cor_gene, cosine_sim_gene))
    logger.info("Running Gene Correlation task DINE!")

    # mean_cor_box(adata, data_impt_reshape)
    mean_cor_box(adata, data_impt_reshape, save_path=os.path.join(args.figure_save_path, 'Box_spot_gene_corr.pdf'))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CellContrast Model Training and Inference")
    parser.add_argument('--system_path', type=str, required=True, help='System path for data and weights')
    parser.add_argument('--LRgene_path', type=str, required=True, help='Path to LR genes')
    parser.add_argument('--dataset_class', type=str, required=True, help='Visium or VisiumHD')
    parser.add_argument('--gene_selected', type=str, required=True, help='Marker gene visualization')
    parser.add_argument('--image_embed_path', type=str, required=True, help='Path to image embeddings')
    parser.add_argument('--visium_path', type=str, required=True, help='Path to Visium data')
    parser.add_argument('--weight_path', type=str, default='weights', help='Directory to save weights')
    parser.add_argument('--parame_path', type=str, required=True, help='Path to parameter file')
    parser.add_argument('--spatial_pos_path', type=str, default='spatial_pos.csv', help='Path to save spatial positions')
    parser.add_argument('--reduced_mtx_path', type=str, default='reduced_mtx.npy', help='Path to save reduced matrix')
    parser.add_argument('--figure_save_path', type=str, default='figures', help='Directory to save figures')
    parser.add_argument('--weight_save_path', type=str, default=None, help='Path to pre-trained weights, if available')
    args = parser.parse_args()

    main(args)


## Python Script:

###################
# If haven't train
###################
# python ./FineST/FineST/demo/FineST_Model_Training.py \
#     --system_path '/mnt/lingyu/nfs_share2/Python/' \
#     --weight_path 'FineST/FineST_local/Finetune/' \
#     --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
#     --dataset_class 'Visium' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
#     --image_embed_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/' \
#     --spatial_pos_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order.csv' \
#     --reduced_mtx_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/harmony_matrix.npy' \
#     --figure_save_path 'FineST/FineST_local/Dataset/NPC/Figures/' 

###################
# If haven trained
###################
# python ./FineST/FineST/demo/FineST_train_infer.py \
#     --system_path '/mnt/lingyu/nfs_share2/Python/' \
#     --weight_path 'FineST/FineST_local/Finetune/' \
#     --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
#     --dataset_class 'Visium' \
#     --gene_selected 'CD70' \
#     --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
#     --image_embed_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/' \
#     --spatial_pos_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order.csv' \
#     --reduced_mtx_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/harmony_matrix.npy' \
#     --weight_save_path 'FineST/FineST_local/Finetune/20240125140443830148' \
#     --figure_save_path 'FineST/FineST_local/Dataset/NPC/Figures/' 
