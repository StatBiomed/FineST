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

Train FineST model on within-spots to learn the mapping from image features to gene expression. 
If pre-trained weights are available, set ``--weight_save_path`` to skip training.

.. code-block:: bash

   python ./demo/Step1_FineST_train_infer.py \
      --system_path '/home/lingyu/ssd/Python/FineST/FineST/' \
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

**Key parameters:**

* ``--dataset_class``: ``'Visium16'`` (HIPT, patch_size=64), ``'Visium64'`` (Virchow2, patch_size=112), or ``'VisiumHD'``
* ``--image_class``: ``'HIPT'`` or ``'Virchow2'`` (must match Step0)
* ``--weight_save_path``: (optional) Path to pre-trained weights to skip training

**Expected output:**

* Average correlation of all spots: ~0.85
* Average correlation of all genes: ~0.88

**Input files:**

* ``parameters_NPC_HIPT.json`` or ``parameters_NPC_virchow2.json``: The model parameters
* ``LRgene_CellChatDB_baseline.csv``: Ligand-receptor genes from CellChatDB
* ``tissue_positions_list.csv``: Visium spot positions (from 10x Visium spatial folder)
* Image embeddings folder (e.g., ``pth_64_16`` for HIPT or ``pth_112_14`` for Virchow2): From ``Image_feature_extraction.py``

**Output files:**

* ``Figures/weights[timestamp]/``: Trained model weights (.pt) and logs (.log)
* ``Figures/Results[timestamp].log``: Complete execution log
* ``Figures/``: Visualization plots (.pdf, .svg)
* ``SaveData/``: Processed AnnData files (adata_count.h5ad, adata_norml.h5ad, adata_infer.h5ad, etc.)
* ``OrderData/position_order.csv``: Ordered tissue positions
* ``OrderData/matrix_order.npy``: Ordered gene expression matrix
