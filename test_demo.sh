## 2026.02.10 when you use this script, please replace your own path with the following:
# your conda env: /ssd2/users/lingyu/conda_envs/FineST/bin/python
# your system path: /ssd2/users/lingyu/Python/FineST/FineST/

########################################################
# Step0: Extract image features
########################################################
## Option A: Using HIPT (recommended for quick start, no token required)
/ssd2/users/lingyu/conda_envs/FineST/bin/python ./demo/Image_feature_extraction.py \
   --dataset NPC \
   --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv \
   --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
   --scale_image False \
   --method HIPT \
   --patch_size 64 \
   --output_img FineST_tutorial_data/ImgEmbeddings/pth_64_16_image \
   --output_pth FineST_tutorial_data/ImgEmbeddings/pth_64_16 \
   --logging FineST_tutorial_data/ImgEmbeddings/Logging/ \
   --scale 0.5  # default is 0.5

########################################################
# Step1: Training FineST on the within spots
########################################################
## HIPT with Visium16 (patch_size=64)
/ssd2/users/lingyu/conda_envs/FineST/bin/python ./demo/Step1_FineST_train_infer.py \
   --system_path '/ssd2/users/lingyu/Python/FineST/FineST/' \
   --parame_path 'parameter/parameters_NPC_HIPT.json' \
   --dataset_class 'Visium16' \
   --image_class 'HIPT' \
   --gene_selected 'CD70' \
   --LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv' \
   --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
   --image_embed_path 'FineST_tutorial_data/ImgEmbeddings/pth_64_16' \
   --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order.csv' \
   --reduced_mtx_path 'FineST_tutorial_data/OrderData/matrix_order.npy' \
   --figure_save_path 'FineST_tutorial_data/Figures/' \
   --save_data_path 'FineST_tutorial_data/SaveData/' \
   --patch_size 64 \
   --weight_w 0.5

########################################################
# Step2: Super-resolution spatial RNA-seq imputation
########################################################

## Setp2.0: Interpolate between spots
/ssd2/users/lingyu/conda_envs/FineST/bin/python ./demo/Spot_interpolation.py \
   --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv


###################################
# Option A: sub-spot resolution
###################################
## Setp A1: Extract image features for between-spots
## Option A: Using HIPT
/ssd2/users/lingyu/conda_envs/FineST/bin/python ./demo/Image_feature_extraction.py \
   --dataset NEW_NPC \
   --position_path FineST_tutorial_data/spatial/tissue_positions_list_add.csv  \
   --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
   --scale_image False \
   --method HIPT \
   --patch_size 64 \
   --output_img FineST_tutorial_data/ImgEmbeddings/NEW_pth_64_16_image \
   --output_pth FineST_tutorial_data/ImgEmbeddings/NEW_pth_64_16 \
   --logging FineST_tutorial_data/ImgEmbeddings/Logging/ \
   --scale 0.5  # Optional, default is 0.5

## Setp A2: Imputation at sub-spot resolution
## [Note: need the 'Figures/weightsxxx' from Step1]

## Option A: Using HIPT 
# /ssd2/users/lingyu/conda_envs/FineST/bin/python ./demo/Step2_High_resolution_imputation.py \
#    --system_path '/ssd2/users/lingyu/Python/FineST/FineST/' \
#    --parame_path 'parameter/parameters_NPC_HIPT.json' \
#    --dataset_class 'Visium16' \
#    --gene_selected 'CD70' \
#    --LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv' \
#    --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
#    --imag_within_path 'FineST_tutorial_data/ImgEmbeddings/pth_64_16' \
#    --imag_betwen_path 'FineST_tutorial_data/ImgEmbeddings/NEW_pth_64_16' \
#    --spatial_pos_path 'FineST_tutorial_data/OrderData/position_order_all.csv' \
#    --weight_save_path 'FineST_tutorial_data/Figures/weights20260209182325169111' \
#    --figure_save_path 'FineST_tutorial_data/Figures/' \
#    --adata_all_supr_path 'FineST_tutorial_data/SaveData/adata_imput_all_subspot.h5ad' \
#    --adata_all_spot_path 'FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad'


###################################
# Option B: single-cell resolution
###################################
## Setp B1: Nuclei segmentation (for single-cell level)
# /ssd2/users/lingyu/conda_envs/FineST/bin/python ./demo/StarDist_nuclei_segmente.py \
#    --tissue NPC_allspot_p075 \
#    --out_dir FineST_tutorial_data/NucleiSegments \
#    --adata_path FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad \
#    --img_path FineST_tutorial_data/20210809-C-AH4199551.tif \
#    --prob_thresh 0.75

## Setp B2: Extract image features for single-nuclei
## Option A: Using HIPT
# /ssd2/users/lingyu/conda_envs/FineST/bin/python ./demo/Image_feature_extraction.py \
#    --dataset sc_NPC \
#    --position_path FineST_tutorial_data/NucleiSegments/NPC_allspot_p075/position_all_tissue_sc.csv  \
#    --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
#    --scale_image False \
#    --method HIPT \
#    --patch_size 16 \
#    --output_img FineST_tutorial_data/ImgEmbeddings/sc_pth_16_16_image \
#    --output_pth FineST_tutorial_data/ImgEmbeddings/sc_pth_16_16 \
#    --logging FineST_tutorial_data/ImgEmbeddings/
#    --scale 0.5  # Optional, default is 0.5

## Setp B3: Imputation at single-cell resolution
# /ssd2/users/lingyu/conda_envs/FineST/bin/python ./demo/Step2_High_resolution_imputation.py \
#    --system_path '/ssd2/users/lingyu/Python/FineST/FineST/' \
#    --parame_path 'parameter/parameters_NPC_HIPT.json' \
#    --dataset_class 'VisiumSC' \
#    --gene_selected 'CD70' \
#    --LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv' \
#    --image_embed_path_sc 'FineST_tutorial_data/ImgEmbeddings/sc_pth_16_16' \
#    --spatial_pos_path_sc 'FineST_tutorial_data/OrderData/position_order_sc.csv' \
#    --weight_save_path 'FineST_tutorial_data/Figures/weights20260209182325169111' \
#    --figure_save_path 'FineST_tutorial_data/Figures/' \
#    --adata_super_path_sc 'FineST_tutorial_data/SaveData/adata_imput_all_sc.h5ad'