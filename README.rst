==========================================================================================================================
FineST: Contrastive learning integrates histology and spatial transcriptomics for nuclei-resolved ligand-receptor analysis
==========================================================================================================================

This software package implements **FineST** (Fine-grained Spatial Transcriptomics), which  
**identifies super-resolved ligand-receptor interactions with spatial co-expression** 
refining *spot* to *sub-spot* or *single-cell* resolution.

.. image:: https://github.com/StatBiomed/FineST/blob/main/docs/fig/FineST_summary_300.png?raw=true
   :width: 800px
   :align: center

**FineST** comprises three components (*Training*-*Imputation*-*Discovery*) after *HE image feature extraction*: 

* Step0: HE image feature extraction
* Step1: **Training** FineST on the *within spots*
* Step2: Super-resolution spatial RNA-seq **imputation** at *sub-spot* or *single-cell* level
* Step3: Fast fine-grained ligand-receptor pair and cell-cell communication pattern **discovery**

.. It comprises two main steps:

.. 1. global selection `spatialdm_global` to identify significantly interacting LR pairs;
.. 2. local selection `spatialdm_local` to identify local spots for each interaction.

Installation using Conda
========================

.. code-block:: bash

   git clone https://github.com/StatBiomed/FineST.git
   conda create --name FineST python=3.8
   conda activate FineST
   cd FineST
   pip install -r requirements.txt

.. Typically installation is completed within a few minutes. 
.. Then install pytorch, refer to `pytorch installation <https://pytorch.org/get-started/locally/>`_.

.. .. code-block:: bash

..    conda install pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch

Verify the installation using the following command:

.. code-block:: text

   python
   >>> import torch
   >>> print(torch.__version__)
   2.1.2+cu121 (or your installed version)
   >>> print(torch.cuda.is_available())
   True

.. Installation using PyPI
.. =======================

FineST package is available through `PyPI <https://pypi.org/project/FineST/>`_.

.. To install, type the following command line and add ``-U`` for updates:

.. code-block:: bash

   pip install -U FineST

   ## Alternatively, install from GitHub for latest version:
   pip install -U git+https://github.com/StatBiomed/FineST

.. Alternatively, install from this GitHub repository for latest (often
.. development) version (time: < 1 min):

.. .. code-block:: bash

..    pip install -U git+https://github.com/StatBiomed/FineST

The FineST conda environment can be used for the following **Tutorial** by:

.. code-block:: text

   python -m pip install ipykernel
   python -m ipykernel install --user --name=FineST

**Tutorial notebooks:**
* `NPC_Train_Impute_demo.ipynb <https://github.com/StatBiomed/FineST/tree/main/tutorial/NPC_Train_Impute_demo.ipynb>`_ (using Virchow2; requires Hugging Face token, approval may take days)
* `NPC_Train_Impute_demo_HIPT.ipynb <https://github.com/StatBiomed/FineST/blob/main/tutorial/NPC_Train_Impute_demo_HIPT.ipynb>`_ (using HIPT; recommended for quick start)


ROI selection via Napair
========================

To analyze a specific region of interest (ROI), use `napari <https://github.com/napari/napari>`_ to select the region:

.. code-block:: bash

   from PIL import Image
   Image.MAX_IMAGE_PIXELS = None
   import matplotlib.pyplot as plt
   import napari

   image = plt.imread("FineST_tutorial_data/20210809-C-AH4199551.tif")
   viewer = napari.view_image(image, channel_axis=2, ndisplay=2)
   napari.run()

**Quick guide:**
* A *shapes* layer is automatically added when opening napari
* Use the ``Add Polygons`` tool to draw ROI(s) on the HE image
* Optionally rename the ROI layer for clarity

For detailed instructions and ROI extraction using ``fst.crop_img_adata()``, see the 
`tutorial <https://finest-rtd-tutorial.readthedocs.io/en/latest/Crop_ROI_Boundary_image.html>`_ or 
`video guide <https://drive.google.com/file/d/1y3sb_Eemq3OV2gkxwu4gZBhLFp-gpzpH/view?usp=sharing>`_.


Get Started for *Visium* or *Visium HD* data
============================================


Data download
-------------

Download the tutorial data from `Google Drive <https://drive.google.com/drive/folders/10WvKW2EtQVuH3NWUnrde4JOW_Dd_H6r8?usp=sharing>`_ or via command line:

.. code-block:: bash

   python -m pip install gdown
   gdown --folder https://drive.google.com/drive/folders/1rZ235pexAMVvRzbVZt1ONOu7Dcuqz5BD?usp=drive_link

The tutorial includes:
* *Visium*: 10x Visium human nasopharyngeal carcinoma (NPC) data
* *Visium HD*: 10x Visium HD human colorectal cancer (CRC) data (16-um bin)


Step0: HE image feature extraction (for *Visium*)
-------------------------------------------------

For *Visium* data (~5k spots, 55-um spot diameter, 100-um center-to-center distance), 
we first interpolate additional spots between the original measured spots to increase spatial resolution.

**Step 0.1: Interpolate between spots**

Interpolate additional spots in horizontal and vertical directions:

.. code-block:: bash
      
   python ./demo/Spot_interpolation.py \
      --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv

**Input:** ``tissue_positions_list.csv`` (original within-spots)  
**Output:** ``tissue_positions_list_add.csv`` (interpolated between-spots, ~3x original)


**Step 0.2: Extract image features for within-spots**

Extract HE image feature embeddings using either HIPT or Virchow2:

**Option A: Using HIPT** (recommended for quick start, no token required)

.. code-block:: bash

   python ./demo/Image_feature_extraction.py \
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

**Option B: Using Virchow2** (requires Hugging Face token)

.. code-block:: bash

   python ./demo/Image_feature_extraction.py \
      --dataset NPC \
      --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv \
      --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
      --scale_image False \
      --method Virchow2 \
      --patch_size 112 \
      --output_img FineST_tutorial_data/ImgEmbeddings/pth_112_14_image \
      --output_pth FineST_tutorial_data/ImgEmbeddings/pth_112_14 \
      --logging FineST_tutorial_data/ImgEmbeddings/Logging/ \
      --scale 0.5  # default is 0.5


**Step 0.3: Extract image features for between-spots**

Similarly extract features for the interpolated between-spots:

**Option A: Using HIPT**

.. code-block:: bash

   python ./demo/Image_feature_extraction.py \
      --dataset NEW_NPC \
      --position_path FineST_tutorial_data/spatial/tissue_positions_list_add.csv \
      --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
      --scale_image False \
      --method HIPT \
      --patch_size 64 \
      --output_img FineST_tutorial_data/ImgEmbeddings/NEW_pth_64_16_image \
      --output_pth FineST_tutorial_data/ImgEmbeddings/NEW_pth_64_16 \
      --logging FineST_tutorial_data/ImgEmbeddings/Logging/ \
      --scale 0.5  # default is 0.5

**Option B: Using Virchow2**

.. code-block:: bash

   python ./demo/Image_feature_extraction.py \
      --dataset NEW_NPC \
      --position_path FineST_tutorial_data/spatial/tissue_positions_list_add.csv \
      --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
      --scale_image False \
      --method Virchow2 \
      --patch_size 112 \
      --output_img FineST_tutorial_data/ImgEmbeddings/NEW_pth_112_14_image \
      --output_pth FineST_tutorial_data/ImgEmbeddings/NEW_pth_112_14 \
      --logging FineST_tutorial_data/ImgEmbeddings/Logging/ \
      --scale 0.5  # default is 0.5

**Output:** Image feature embeddings (``NEW_pth_64_16`` or ``NEW_pth_112_14``) for between-spots


Step0: HE image feature extraction (for *Visium HD*)
----------------------------------------------------

For *Visium HD* (continuous squares without gaps), extract image features directly:

.. code-block:: bash

   python ./demo/Image_feature_extraction.py \
      --dataset HD_CRC_16um \
      --position_path ./Dataset/CRC/square_016um/tissue_positions.parquet \
      --rawimage_path ./Dataset/CRC/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
      --scale_image True \
      --method Virchow2 \
      --output_img ./Dataset/CRC/HIPT/HD_CRC_16um_pth_28_14_image \
      --output_pth ./Dataset/CRC/HIPT/HD_CRC_16um_pth_28_14 \
      --patch_size 28 \
      --logging ./Logging/HIPT_HD_CRC_16um/ \
      --scale 0.5  # default is 0.5

**Note:** *Visium HD* uses ``.parquet`` for positions and ``.btf`` for images, while *Visium* uses ``.csv`` and ``.tif``.


Step1: Training FineST on the within spots
==========================================

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

**Output files:**
* ``Figures/weights[timestamp]/``: Trained model weights (.pt) and logs (.log)
* ``Figures/Results[timestamp].log``: Complete execution log
* ``Figures/``: Visualization plots (.pdf, .svg)
* ``SaveData/``: Processed AnnData files (adata_count.h5ad, adata_norml.h5ad, adata_infer.h5ad, etc.)
* ``OrderData/position_order.csv``: Ordered tissue positions
* ``OrderData/matrix_order.npy``: Ordered gene expression matrix


Step2: Super-resolution spatial RNA-seq imputation
==================================================

For *sub-spot* resolution
-------------------------

This step supposes that the trained weights (i.e. **weight_save_path**) have been obtained, just run the following.

.. code-block:: bash

   python ./FineST/demo/Step2_High_resolution_imputation.py \
      --system_path '/mnt/lingyu/nfs_share2/Python/' \
      --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
      --dataset_class 'Visium64' \
      --image_class 'Virchow2' \
      --gene_selected 'CD70' \
      --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
      --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
      --imag_within_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_112_14/' \
      --imag_betwen_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/NEW_AH_Patient1_pth_112_14/' \
      --weight_save_path 'FineST/FineST_local/Finetune/20240125140443830148' \
      --patch_size 112 \
      --adata_all_supr_path 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all.h5ad' \
      --adata_all_spot_path 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all_spot.h5ad' 

``Step2_High_resolution_imputation.py`` is used to predict super-resolved gene expression 
based on the image segmentation (Geometric ``sub-spot level`` or Nuclei ``single-cell level``).

**Input files:**

* ``parameters_NPC_P10125.json``: The model parameters.
* ``LRgene_CellChatDB_baseline.csv``: The genes involved in Ligand or Receptor from CellChatDB.
* ``tissue_positions_list.csv``: It can be found in the spatial folder of 10x Visium outputs.
* ``AH_Patient1_pth_112_14``: Image feature of within-spots from ``Image_feature_extraction.py``.
* ``NEW_AH_Patient1_pth_112_14``: Image feature of between-spots from ``Image_feature_extraction.py``.
* ``20240125140443830148``: The trained weights from Step1.

**Output files:**

* ``patient1_adata_all.h5ad``: High-resolution gene expression, at sub-spot level (16x3x resolution).
* ``patient1_adata_all_spot.h5ad``: High-resolution gene expression, at spot level (3x resolution).

For *single-cell* resolution
----------------------------

Using ``sc Patient1 pth 16 16`` 
i.e., the image feature of single-nuclei from ``Image_feature_extraction.py``, just run the following.

.. code-block:: bash

   python ./FineST/demo/Step2_High_resolution_imputation.py \
      --system_path '/mnt/lingyu/nfs_share2/Python/' \
      --parame_path 'FineST/FineST/parameter/parameters_NPC_P10125.json' \
      --dataset_class 'VisiumSC' \
      --image_class 'Virchow2' \
      --gene_selected 'CD70' \
      --LRgene_path 'FineST/FineST/Dataset/LRgene/LRgene_CellChatDB_baseline.csv' \
      --visium_path 'FineST/FineST/Dataset/NPC/patient1/tissue_positions_list.csv' \
      --imag_within_path 'NPC/Data/stdata/ZhuoLiang/LLYtest/AH_Patient1_pth_112_14/' \
      --image_embed_path_sc 'NPC/Data/stdata/ZhuoLiang/LLYtest/sc_Patient1_pth_16_16/' \
      --adata_super_path_sc 'FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_all_sc.h5ad' \
      --weight_save_path 'FineST/FineST_local/Finetune/20240125140443830148' \
      --patch_size 112


Step3: Fine-grained LR pair and CCC pattern discovery
=====================================================

This step is based on `SpatialDM <https://github.com/StatBiomed/SpatialDM>`_ and `SparseAEH <https://github.com/jackywangtj66/SparseAEH>`_ (developed by our Lab). 

 * SpatialDM: for significant fine-grained ligand-receptor pair selection.
 * SparseAEH: for fast cell-cell communication pattern discovery, 1000 times speedup to `SpatialDE <https://github.com/Teichlab/SpatialDE>`_.


Detailed Manual
===============

The full manual is at `FineST tutorial <https://finest-rtd-tutorial.readthedocs.io>`_ for installation, tutorials and examples.

**Spot interpolation** for Visium datasets.

* `Interpolate between-spots among within-spots by FineST (For Visium dataset)`_.

.. _Interpolate between-spots among within-spots by FineST (For Visium dataset): docs/source/Between_spot_demo.ipynb


**Step1 and Step2** Train FineST and impute super-resolved spatial RNA-seq.

* `FineST on Visium HD for super-resolved gene expression prediction (from 16um to 8um)`_.

.. _FineST on Visium HD for super-resolved gene expression prediction (from 16um to 8um): docs/source/CRC16_Train_Impute_count.ipynb

* `FineST on Visium for super-resolved gene expression prediction (sub-spot or single-cell)`_.

.. _FineST on Visium for super-resolved gene expression prediction (sub-spot or single-cell): docs/source/NPC_Train_Impute_count.ipynb


**Step3** Fine-grained LR pair and CCC pattern discovery.

* `Nuclei-resolved ligand-receptor interaction discovery by FineST (For Visium dataset)`_.

.. _Nuclei-resolved ligand-receptor interaction discovery by FineST (For Visium dataset): docs/source/NPC_LRI_CCC_count.ipynb

* `Super-resolved ligand-receptor interaction discovery by FineST (For Visium HD dataset)`_.

.. _Super-resolved ligand-receptor interaction discovery by FineST (For Visium HD dataset): docs/source/CRC_LRI_CCC_count.ipynb


**Downstream analysis** Cell type deconvolution, ROI region cropping, cell-cell colocalization.

* `Nuclei-resolved cell type deconvolution of Visium (use FineST-imputed data)`_.

.. _Nuclei-resolved cell type deconvolution of Visium (use FineST-imputed data): docs/source/transDeconv_NPC_count.ipynb

* `Super-resolved cell type deconvolution of Visium HD (For FineST-imputed data)`_.

.. _Super-resolved cell type deconvolution of Visium HD (For FineST-imputed data): docs/source/transDeconv_CRC_count.ipynb

* `Crop region of interest (ROI) from HE image by FineST (Visium or Visium HD)`_.

.. _Crop region of interest (ROI) from HE image by FineST (Visium or Visium HD): docs/source/Crop_ROI_Boundary_image.ipynb


**Performance evaluation** of FineST vs (TESLA and iSTAR).

* `PCC-SSIM-CelltypeProportion-RunTimes comparison in FineST manuscript`_.

.. _PCC-SSIM-CelltypeProportion-RunTimes comparison in FineST manuscript: docs/source/NPC_Evaluate.ipynb


**Inference comparison** of FineST vs iStar (only LR genes).

* `FineST on demo data`_.

.. _FineST on demo data: docs/source/Demo_Train_Impute_count.ipynb

* `iStar on demo data`_.

.. _iStar on demo data: docs/source/Demo_results_istar_check.ipynb


Citation
========

If you use FineST (Accepted in principle by Nature Comm) in your research, please cite:

.. code-block:: bash

   @misc{FineST,
      author={Li, Lingyu and Wang, Tianjie and Liang, Zhuo and Yu, Huajian and Ma, Stephanie and Yu, Lequan and Huang, Yuanhua},
      title={{FineST: Contrastive learning integrates histology and spatial transcriptomics for nuclei-resolved ligand-receptor analysis}},
      year={2026},
      note = {\url{https://github.com/StatBiomed/FineST}}
   }

Contact Information
===================

Please contact Lingyu Li (`lingyuli@hku.hk <mailto:lingyuli@hku.hk>`_) or Yuanhua Huang (`yuanhua@hku.hk <mailto:yuanhua@hku.hk>`_) if any enquiry.

