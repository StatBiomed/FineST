Model training within spots
===========================

Training-Inferring/Imputation-Evaluation based on Geometric Segmentation
------------------------------------------------------------------------

**Usage illustrations**: 

* For *Training*, using patch-image feature embeddings from Geometric Segmentation.

* For *Inferring*, only input image embeddings from Geometric or Nuclei Segmentation.

* For *Imputation*, based on observed spot-level gene expr with neighbor information.

* For *Evaluation*, using gene correlation (predicted vs observed) across all spots.


Step1: Training FineST on the within spots
------------------------------------------

On *Visium* dataset, if the trained weights (i.e. **weight_save_path**) have been obtained, 
just run the following command.
Otherwise, if you want to newly train a model, 
you can omit **weight_save_path** from the following command.

On *VisiumHD* dataset, remember set ``dataset_class`` as ``'VisiumHD' `` (--dataset_class 'Visium')

.. code-block:: bash

   python ./FineST/FineST/demo/FineST_train_infer.py \
      --system_path '/mnt/lingyu/nfs_share2/Python/' \
      --weight_path 'FineST/FineST_local/Finetune/' \
      --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
      --dataset_class 'Visium' \
      --gene_selected 'CD70' \
      --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
      --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
      --image_embed_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_64_16/' \
      --spatial_pos_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/position_order.csv' \
      --reduced_mtx_path 'FineST/FineST_local/Dataset/NPC/ContrastP1geneLR/harmony_matrix.npy' \
      --weight_save_path 'FineST/FineST_local/Finetune/20240125140443830148' \
      --figure_save_path 'FineST/FineST_local/Dataset/NPC/Figures/' 

``FineST_train_infer.py`` is used to train and evaluate the FineST model using Pearson Correlation, it outputs:

* Average correlation of all spots: 0.8534651812923978
* Average correlation of all genes: 0.8845136777311445

**Input files:**

* ``parameters_NPC_P10125.json``: The model parameters.
* ``LRgene_CellChatDB_baseline.csv``: The genes involved in Ligand or Receptor from CellChatDB.
* ``tissue_positions_list.csv``: It can be found in the spatial folder of 10x Visium outputs.
* ``AH_Patient1_pth_64_16``: Image feature of within-spots from ``HIPT_image_feature_extract.py``.
* ``position_order.csv``: Ordered tissue positions list, according to image patches' coordinates.
* ``harmony_matrix.npy``: Ordered gene expression matrix, according to image patches' coordinates.
* ``20240125140443830148``: The trained weights. Just omit it if you want to newly train a model.

**Output files:**

* ``Finetune``: The logging results ``model.log`` and trained weights ``epoch_50.pt`` (.log and .pt)
* ``Figures``: The visualization plots, used to see whether the model trained well or not (.pdf)
