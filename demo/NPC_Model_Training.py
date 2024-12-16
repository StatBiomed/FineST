import os
import time
import warnings
import numpy as np
import torch
import logging
from datetime import datetime
import json
import argparse
print("torch version: %s" % torch.__version__)

from FineST.utils import *
from FineST import datasets
from FineST.processData import *
from FineST.model import *
from FineST.train import *
from FineST.plottings import *
from FineST.inference import *

## Basic setting
warnings.filterwarnings('ignore')
setup_seed(666)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Training a FineST model")

    parser.add_argument('--path', type=str,
                        help='The path of main pathway',
                        default=None)
    parser.add_argument('--model_folder', type=str,
                        help='The path of loggings with .log format',
                        default=None)
    parser.add_argument('--parameter_file_path', type=str,
                        help='The path of model parameters with .json format',
                        default=None)
    parser.add_argument('--LR_gene_path', type=str,
                        help='The path of Ligand-Receptor gene with .csv format (tensor object)',
                        default=None)
    parser.add_argument('--visium_pos_path', type=str,
                        help='The path of visium position with .csv format (dataframe object)',
                        default=None)
    parser.add_argument('--image_embed_path_NPC', type=str,
                        help='The path of image embeddings with .pth format (tensor object)',
                        default=None)
    parser.add_argument('--spatial_pos_path', type=str,
                        help='The path of ordered position with .csv format (dataframe object)',
                        default=None)
    parser.add_argument('--reduced_mtx_path', type=str,
                        help='The path of ordered matrix with .npy format (dataframe object)',
                        default=None)
    parser.add_argument('--dir_name_model', type=str,
                        help='The path of trained parameters with .pt format (dataframe object)',
                        default=None)
    parser.add_argument('--fig_save_path', type=str,
                        help='The path of saved fugure with .pdf format',
                        default=None)

    args = parser.parse_args()

    if not check_file_exists(args.LR_gene_path):
        return
    if not check_file_exists(os.path.join(args.path, args.image_embed_path_NPC)):
        return
    if not check_file_exists(args.spatial_pos_path):
        return
    if not check_file_exists(args.reduced_mtx_path):
        return

    adata = datasets.NPC()
    print(" **** Load the original NPC patient1 adata: \n", adata)
    adata = adata_LR(adata, args.LR_gene_path)
    adata = adata_preprocess(adata)
    print(" **** Processed NPC patient1 adata: \n", adata)
    gene_hv = np.array(adata.var_names)
    print(" **** The length of LR genes:", len(gene_hv))

    matrix = adata2matrix(adata, gene_hv)
    file_paths = sorted(os.listdir(os.path.join(args.path, args.image_embed_path_NPC)))
    print(" **** Image embedding file (First 3):", file_paths[:3])


    ## Image patch position
    position_image = get_image_coord(file_paths, dataset_class="Visium")
    ## ST spot position
    position = pd.read_csv(os.path.join(args.path, args.visium_pos_path), header=None)
    position = position.rename(columns={position.columns[-2]: 'pixel_x', 
                                        position.columns[-1]: 'pixel_y'})
    ## merge position
    position_image = image_coord_merge(position_image, position, dataset = 'Visium')
    position_order = update_st_coord(position_image)
    print(" **** The coords of image patch: \n", position_order.shape)
    print(position_order.head())
    position_order.to_csv(os.path.join(args.path, args.spatial_pos_path), index=False, header=False)

    spotID_order = np.array(position_image[0])
    matrix_order, matrix_order_df = sort_matrix(matrix, position_image, spotID_order, gene_hv)
    np.save(os.path.join(args.path, args.reduced_mtx_path), matrix_order_df.T)

    adata = update_adata_coord(adata, matrix_order, position_image)
    gene_expr(adata, matrix_order_df, gene_selet='CD70', 
              save_path=os.path.join(args.fig_save_path,'original spot gene expr.pdf'))

    logging.getLogger().setLevel(logging.INFO)

    # model_folder = str(path) + 'FineST/FineST_local/Finetune/'
    model_folder = os.path.join(args.path, args.model_folder)
    dir_name = model_folder + datetime.now().strftime('%Y%m%d%H%M%S%f')

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logger = setup_logger(dir_name)
    print("dir_name: \n", dir_name)

    # parameter_file_path = str(path) + 'FineST/FineST/parameter/parameters_NPC_P10125.json'
    parameter_file_path = os.path.join(args.path, args.parameter_file_path)
    with open(parameter_file_path, "r") as json_file:
        params = json.load(json_file)
    logger.info("Load parameters:\n" + json.dumps(params, indent=2))

    ##########################
    # Train
    ##########################
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

    # ## Set optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['inital_learning_rate'],
    #                             momentum=0.9, weight_decay=5e-4)

    # ## Load the data
    # train_loader, test_loader = build_loaders(batch_size=params['batch_size'],
    #                                           image_embed_path=os.path.join(args.path, args.image_embed_path_NPC, '*.pth'),
    #                                           spatial_pos_path=args.spatial_pos_path,
    #                                           reduced_mtx_path=args.reduced_mtx_path,
    #                                           dataset_class='Visium')

    # ## Train the model for a fixed number of epochs
    # logger.info('Begin Training ...')

    # start_train_time = time.time()

    # best_loss = float('inf')
    # best_epoch = 0
    # for epoch in range(params['training_epoch']):
    #     logger.info('epoch [{}/{}]'.format(epoch + 1, params['training_epoch']))
    #     print(f"Epoch: {epoch + 1}")

    #     ######################################################################################
    #     # Train the model
    #     ######################################################################################
    #     model.train()
    #     start_time = time.time()
    #     train_loss = train_model(params, model, train_loader, optimizer, epoch, l,
    #                              tree_type='KDTree', leaf_size=2, dataset_class='Visium')
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
    #         print("Saved Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))
    #         logger.info("Saved Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))

    # print("Done!, final loss: {}".format(best_loss))
    # print("Best epoch: {}".format(best_epoch))

    # print("--- %s seconds ---" % (time.time() - start_train_time))

    # logger.info("Done!, Best epoch & Best Model! Loss: [{}: {}]".format(best_epoch, best_loss))
    # logger.info('Finished Training')


    ## inference
    # model = load_model(dir_name, parameter_file_path, params, gene_hv)   
    model = load_model(args.dir_name_model, parameter_file_path, params, gene_hv)   
    model.to(device)

    test_loader = build_loaders_inference(batch_size=adata.shape[0], 
                                            image_embed_path=os.path.join(args.path, args.image_embed_path_NPC, '*.pth'),
                                            spatial_pos_path=args.spatial_pos_path,
                                            reduced_mtx_path=args.reduced_mtx_path,
                                            dataset_class='Visium')


    logger.info("Running inference tesk...")

    start_infer_time = time.time()

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

    print("--- %s seconds for inference within spots ---" % (time.time() - start_infer_time))
    print("reconstructed_matrix_reshaped shape: ", reconstructed_matrix_reshaped.shape)

    logger.info("Running inference tesk DONE!")


    reconstructed_matrix_reshaped_tensor, _ = reshape_latent_image(reconstructed_matrix_reshaped, 
                                                                   dataset_class='Visium')
    print(" **** The size of reconstructed tensor data:", reconstructed_matrix_reshaped_tensor.shape)

    (first_spot_first_variable, C,
    _, _, _) = subspot_coord_expr_adata(reconstructed_matrix_reshaped_tensor,
                                        adata, gene_hv, p=0, q=0, dataset_class="Visium")
    # subspot_expr(C, first_spot_first_variable, save_path=os.path.join(args.fig_save_path, '1st_spot_1st_gene.pdf'))


    (_, _, all_spot_all_variable,
    C2, adata_infer) = subspot_coord_expr_adata(reconstructed_matrix_reshaped_tensor,
                                                adata, gene_hv, dataset_class="Visium")
    print(" **** All_spot_all_variable shape:", all_spot_all_variable.shape)
    print(adata_infer)

    ## impute 
    adata_imput = impute_adata(adata, adata_infer, C2, gene_hv, k=6)
    _, data_impt = weight_adata(adata_infer, adata_imput, gene_hv, w=0.5)
    _, data_impt_reshape = reshape_latent_image(data_impt, dataset_class='Visium')
    print(" **** data_impt shape:", data_impt.shape)
    print(" **** data_impt_reshape shape:", data_impt_reshape.shape)
    # gene_expr_compare(adata, "CD70", data_impt_reshape, gene_hv, s=50, 
    #                     save_path=os.path.join(args.fig_save_path, 'CD70_gene_expr.pdf'))


    ## evaluate 
    sele_gene_cor(adata, data_impt_reshape, gene_hv, gene = "CD70", 
                        ylabel='FineST Expression', title = "CD70 expression", size=5, 
                        save_path=os.path.join(args.fig_save_path, 'CD70_gene_corr.pdf'))    

    logger.info("Running Gene Correlation task...")
    (pearson_cor_gene, 
    spearman_cor_gene, 
    cosine_sim_gene) = mean_cor(adata, data_impt_reshape, 'reconf2', sample="gene")
    logger.info("Running Gene Correlation task DINE!")

    mean_cor_box(adata, data_impt_reshape)
    mean_cor_box(adata, data_impt_reshape, save_path=os.path.join(args.fig_save_path, 'spot_gene_corr.pdf'))

if __name__ == '__main__':
    main()


# python ./FineST/FineST/demo/NPC_Model_Training.py \
#     --path '/mnt/lingyu/nfs_share2/Python/' \
#     --model_folder 'FineST/FineST_local/Finetune/' \
#     --parameter_file_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
#     --LR_gene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
#     --visium_pos_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
#     --image_embed_path_NPC 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/' \
#     --spatial_pos_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order.csv' \
#     --reduced_mtx_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/harmony_matrix.npy' \
#     --dir_name_model 'FineST/FineST_local/Finetune/20240125140443830148' \
#     --fig_save_path 'FineST/FineST_local/Dataset/NPC/Figures/' 
