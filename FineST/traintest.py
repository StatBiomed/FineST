import torch
import numpy as np
from .utils import *
from .loadData import * 


def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    """Adjusts the learning rate based on the cosine annealing strategy."""
    lr = 0.5 * initial_lr * (1 + np.cos(np.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_model(model, dir_name, params, optimizer, LOSS):
    cur_save_path = os.path.join(dir_name, "epoch_"+str(params["training_epoch"])+".pt")
    torch.save({
                  'epoch': params['training_epoch'],
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': LOSS,
                  'params': params,
                  # 'train_genes':train_genes,
                   }, cur_save_path)
    
    
def train_model(params, model, train_loader, optimizer, cur_epoch, l, tree_type, leaf_size, dataset_class): 
    
    print("train model")    
    
    cur_lr = adjust_learning_rate(optimizer, cur_epoch, params['inital_learning_rate'], params['training_epoch'])
    total_loss, total_num = 0.0, 0.0

    ## load data
    (cur_train_data_mat, 
     cur_train_matrix_mat, 
     cur_train_coors_mat, 
     cur_pos_info) = loadTrainTestData(train_loader, neighbor_k=params['k_nearest_positives'], 
                                       tree_type=tree_type, leaf_size=leaf_size, dataset_class=dataset_class)

    
    for image_profile, gene_profile, positive_index, _ in loadBatchData(cur_train_data_mat, 
                                                                        cur_train_matrix_mat,
                                                                        cur_train_coors_mat, 
                                                                        params['batch_size_pair'],
                                                                        cur_pos_info):
        
        input_gene_exp = torch.tensor(np.asarray(gene_profile)).float().to(device) # torch.Size([64, 128])
        image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [1331, 256, 384] --> [1331*256, 384]
        input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
        
        ## model
        (representation_matrix, 
         reconstruction_matrix, 
         projection_matrix,
         representation_image, 
         reconstruction_iamge, 
         projection_image) = model(input_gene_exp, input_image_exp)     
        
        ## reshape
        _, representation_image_reshape = reshape_latent_image(representation_image, dataset_class)
        _, projection_image_reshape = reshape_latent_image(projection_image, dataset_class)
    
        ## cross decoder
        reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
        _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped, dataset_class)

        ## compute the loss
        loss = l(
            projection_image_reshape,
            projection_matrix,
            # representation_image_reshape,
            # representation_matrix,
            torch.tensor(positive_index).to(device),
            input_image_exp,
            reconstruction_iamge,
            reconstruction_matrix,
            reconstruction_iamge_reshapef2,
            # input_gene_exp,
            # w1=1, w2=1, w3=1, w4=1
            input_gene_exp
        )  

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * gene_profile.shape[0]
        total_num += gene_profile.shape[0]
        pass
    
    LOSS = total_loss/total_num    # shumin end

    return LOSS



def test_model(params, model, test_loader, l, tree_type, leaf_size, dataset_class): 
    
    print("test model")
    
    # load data
    (cur_train_data_mat, 
     cur_train_matrix_mat, 
     cur_train_coors_mat, 
     cur_pos_info) = loadTrainTestData(test_loader, neighbor_k=params['k_nearest_positives'], 
                                       tree_type=tree_type, leaf_size=leaf_size, dataset_class=dataset_class)
        
    total_loss, total_num = 0.0, 0.0
    for image_profile, gene_profile, positive_index, _ in loadBatchData(cur_train_data_mat, 
                                                                        cur_train_matrix_mat,
                                                                        cur_train_coors_mat, 
                                                                        params['batch_size_pair'],
                                                                        cur_pos_info):
        
        input_gene_exp = torch.tensor(np.asarray(gene_profile)).float().to(device) # torch.Size([64, 128])
        image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [1331, 256, 384] --> [1331*256, 384]
        input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
        
        ## model
        (representation_matrix, 
         reconstruction_matrix, 
         projection_matrix,
         representation_image, 
         reconstruction_iamge, 
         projection_image) = model(input_gene_exp, input_image_exp)     
        
        ## reshape
        _, representation_image_reshape = reshape_latent_image(representation_image, dataset_class)
        _, projection_image_reshape = reshape_latent_image(projection_image, dataset_class)
    
        ## cross decoder
        reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
        _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped, dataset_class)

        ## compute the loss
        loss = l(
            projection_image_reshape,
            projection_matrix,
            # representation_image_reshape,
            # representation_matrix,
            torch.tensor(positive_index).to(device),
            input_image_exp,
            reconstruction_iamge,
            reconstruction_matrix,
            reconstruction_iamge_reshapef2,
            # input_gene_exp,
            # w1=1, w2=1, w3=1, w4=1
            input_gene_exp
        )  
    
        total_loss += loss.item() * gene_profile.shape[0]
        total_num += gene_profile.shape[0]
        pass
    
    LOSS = total_loss/total_num 
    
    # count = batch["reduced_expression"].size(0)
    # loss_meter.update(loss.item(), count)
    # tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return LOSS



##################################
# 2025.01.06 Adjust loss function
# Train and Test model adjust
##################################
# def train_model(params, model, train_loader, optimizer, cur_epoch, l, tree_type, leaf_size, dataset_class): 
    
#     print("train model")    
    
#     cur_lr = adjust_learning_rate(optimizer, cur_epoch, params['inital_learning_rate'], params['training_epoch'])
#     total_loss, total_num = 0.0, 0.0

#     ## load data
#     (cur_train_data_mat, 
#      cur_train_matrix_mat, 
#      cur_train_coors_mat, 
#      cur_pos_info) = loadTrainTestData(train_loader, neighbor_k=params['k_nearest_positives'], 
#                                        tree_type=tree_type, leaf_size=leaf_size, dataset_class=dataset_class)

    
#     for image_profile, gene_profile, positive_index, _ in loadBatchData(cur_train_data_mat, 
#                                                                         cur_train_matrix_mat,
#                                                                         cur_train_coors_mat, 
#                                                                         params['batch_size_pair'],
#                                                                         cur_pos_info):
        
#         input_gene_exp = torch.tensor(np.asarray(gene_profile)).float().to(device)   # torch.Size([64, 128])
#         image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [1331, 256, 384] --> [1331*256, 384]
#         input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
        
#         ## model
#         (representation_matrix, 
#          reconstruction_matrix, 
#          projection_matrix,
#          representation_image, 
#          reconstruction_iamge, 
#          projection_image,
#          recon_mat_mean, recon_mat_disp, recon_mat_pi) = model(input_gene_exp, input_image_exp)     
        
#         ## reshape
#         _, representation_image_reshape = reshape_latent_image(representation_image, dataset_class)
#         _, projection_image_reshape = reshape_latent_image(projection_image, dataset_class)
    
#         ## cross decoder
#         reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
#         _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped, dataset_class)

#         ## compute the loss
#         loss = l(
#             projection_image_reshape,
#             projection_matrix,
#             # representation_image_reshape,
#             # representation_matrix,
#             torch.tensor(positive_index).to(device),
#             input_image_exp,
#             reconstruction_iamge,
#             reconstruction_matrix,
#             reconstruction_iamge_reshapef2,
#             # input_gene_exp,
#             # w1=1, w2=1, w3=1, w4=1
#             input_gene_exp,
#             recon_mat_mean, recon_mat_disp, recon_mat_pi
#         )  

#         # optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * gene_profile.shape[0]
#         total_num += gene_profile.shape[0]
#         pass
    
#     LOSS = total_loss/total_num    # shumin end

#     return LOSS



# def test_model(params, model, test_loader, l, tree_type, leaf_size, dataset_class): 
    
#     print("test model")
    
#     # load data
#     (cur_train_data_mat, 
#      cur_train_matrix_mat, 
#      cur_train_coors_mat, 
#      cur_pos_info) = loadTrainTestData(test_loader, neighbor_k=params['k_nearest_positives'], 
#                                        tree_type=tree_type, leaf_size=leaf_size, dataset_class=dataset_class)
        
#     total_loss, total_num = 0.0, 0.0
#     for image_profile, gene_profile, positive_index, _ in loadBatchData(cur_train_data_mat, 
#                                                                         cur_train_matrix_mat,
#                                                                         cur_train_coors_mat, 
#                                                                         params['batch_size_pair'],
#                                                                         cur_pos_info):
        
#         input_gene_exp = torch.tensor(np.asarray(gene_profile)).float().to(device)   # torch.Size([64, 128])
#         image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [1331, 256, 384] --> [1331*256, 384]
#         input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
        
#         ## model
#         (representation_matrix, 
#          reconstruction_matrix, 
#          projection_matrix,
#          representation_image, 
#          reconstruction_iamge, 
#          projection_image,
#          recon_mat_mean, recon_mat_disp, recon_mat_pi) = model(input_gene_exp, input_image_exp)     
        
#         ## reshape
#         _, representation_image_reshape = reshape_latent_image(representation_image, dataset_class)
#         _, projection_image_reshape = reshape_latent_image(projection_image, dataset_class)
    
#         ## cross decoder
#         reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
#         _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped, dataset_class)

#         ## compute the loss
#         loss = l(
#             projection_image_reshape,
#             projection_matrix,
#             # representation_image_reshape,
#             # representation_matrix,
#             torch.tensor(positive_index).to(device),
#             input_image_exp,
#             reconstruction_iamge,
#             reconstruction_matrix,
#             reconstruction_iamge_reshapef2,
#             # input_gene_exp,
#             # w1=1, w2=1, w3=1, w4=1
#             input_gene_exp,
#             recon_mat_mean, recon_mat_disp, recon_mat_pi
#         )  
    
#         total_loss += loss.item() * gene_profile.shape[0]
#         total_num += gene_profile.shape[0]
#         pass
    
#     LOSS = total_loss/total_num 
    
#     # count = batch["reduced_expression"].size(0)
#     # loss_meter.update(loss.item(), count)
#     # tqdm_object.set_postfix(valid_loss=loss_meter.avg)

#     return LOSS


#################################################
# 2025.01.24: add the Train model im main code
#################################################
def train_model_fst(params, model, train_loader, test_loader, 
                    optimizer, l, dir_name, logger, dataset_class='Visium64'):
    
    logger.info('Begin Training ...')
    start_train_time = time.time()

    best_loss = float('inf')
    best_epoch = 0

    ## Initialize lists to store losses
    train_losses = []
    test_losses = []

    for epoch in range(params['training_epoch']):
        logger.info('epoch [{}/{}]'.format(epoch + 1, params['training_epoch']))
        print(f"Epoch: {epoch + 1}")

        #####################
        # Train the model
        #####################
        model.train()
        start_time = time.time()
        train_loss = train_model(params, model, train_loader, optimizer, epoch, l, 
                                     tree_type='KDTree', leaf_size=2, dataset_class=dataset_class)   
        print("--- %s seconds ---" % (time.time() - start_time))
        
        ## Append training loss
        train_losses.append(train_loss)
        
        #####################
        # Evaluate the model
        #####################
        model.eval()
        with torch.no_grad():
            test_loss = test_model(params, model, test_loader, l, 
                                       tree_type='KDTree', leaf_size=2, dataset_class=dataset_class) 
            # Append test loss
            test_losses.append(test_loss)
        
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

    return dir_name, train_losses, test_losses, best_epoch, best_loss