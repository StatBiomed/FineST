High-sresolution imputation
===========================

Super-resolved gene expression: sub-spot or single-nuclei level
---------------------------------------------------------------

**Usage illustrations**: 

* For *sub-spot* resolution, using patch-image feature embeddings from Geometric Segmentation.

* For *single-cell* resolution, using patch-image feature embeddings from Nuclei Segmentation.


Step2: Super-resolution spatial RNA-seq imputation (for *sub-spot*)
-------------------------------------------------------------------

Suppose that the trained weights (i.e. **weight_save_path**) have been obtained, just run the following.

.. code-block:: bash

   python ./FineST/FineST/demo/High_resolution_imputation.py \
      --system_path '/mnt/lingyu/nfs_share2/Python/' \
      --weight_path 'FineST/FineST_local/Finetune/' \
      --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
      --dataset_class 'Visium' \
      --gene_selected 'CD70' \
      --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
      --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
      --imag_within_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/' \
      --imag_betwen_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/NEW_AH_Patient1_pth_64_16/' \
      --spatial_pos_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order_all.csv' \
      --weight_save_path 'FineST/FineST_local/Finetune/20240125140443830148' \
      --figure_save_path 'FineST/FineST_local/Dataset/NPC/Figures/' \
      --adata_all_supr_path 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all.h5ad' \
      --adata_all_spot_path 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all_spot.h5ad' 

``High_resolution_imputation.py`` is used to predict super-resolved gene expression 
based on the image segmentation (Geometric ``sub-spot level`` or Nuclei ``single-cell level``).

**Input files:**

* ``parameters_NPC_P10125.json``: The model parameters.
* ``LRgene_CellChatDB_baseline.csv``: The genes involved in Ligand or Receptor from CellChatDB.
* ``tissue_positions_list.csv``: It can be found in the spatial folder of 10x Visium outputs.
* ``AH_Patient1_pth_64_16``: Image feature of within-spots from ``HIPT_image_feature_extract.py``.
* ``NEW_AH_Patient1_pth_64_16``: Image feature of between-spots from ``HIPT_image_feature_extract.py``.
* ``position_order_all.csv``: Ordered tissue positions list, of both within spots and between spots.
* ``20240125140443830148``: The trained weights. Just omit it if you want to newly train a model.

**Output files:**

* ``Finetune``: The logging results ``model.log`` and trained weights ``epoch_50.pt`` (.log and .pt)
* ``Figures``: The visualization plots, used to see whether the model trained well or not (.pdf)
* ``patient1_adata_all.h5ad``: High-resolution gene expression, at sub-spot level (16x3x resolution).
* ``patient1_adata_all_spot.h5ad``: High-resolution gene expression, at spot level (3x resolution).


Step2: Super-resolution spatial RNA-seq imputation (for *single-cell*)
----------------------------------------------------------------------

Using ``sc Patient1 pth 16 16`` 
(saved in `Google Drive <https://drive.google.com/drive/folders/10WvKW2EtQVuH3NWUnrde4JOW_Dd_H6r8>`_), i.e.,   
the image feature of single-nuclei from ``HIPT_image_feature_extract.py``, just run the following.

.. code-block:: bash

   python ./FineST/FineST/demo/High_resolution_imputation.py \
      --system_path '/mnt/lingyu/nfs_share2/Python/' \
      --weight_path 'FineST/FineST_local/Finetune/' \
      --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
      --dataset_class 'VisiumSC' \
      --gene_selected 'CD70' \
      --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
      --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
      --imag_within_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/' \
      --image_embed_path_sc 'NPC/Data/stdata/ZhuoLiang/LLYtest/sc_Patient1_pth_16_16/' \
      --spatial_pos_path_sc 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order_sc.csv' \
      --adata_super_path_sc 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all_sc.h5ad' \
      --weight_save_path 'FineST/FineST_local/Finetune/20240125140443830148' \
      --figure_save_path 'FineST/FineST_local/Dataset/NPC/Figures/'
