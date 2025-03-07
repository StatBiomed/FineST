=================================================================
FineST: Fine-grained Spatial Transcriptomic
=================================================================

A statistical model/toolbox to **identify nuclei-resolved ligand-receptor interactions (LRIs) 
with spatial co-expression** (i.e., spatial association). 
Uniquely, FineST can distinguish co-expressed ligand-receptor pairs (LR pairs) 
from spatially separating pairs at sub-spot level or single-cell level.

.. image:: https://github.com/StatBiomed/FineST/blob/main/docs/fig/FineST_framework_all.png?raw=true
   :width: 800px
   :align: center

It comprises three components (*Training*-*Imputation*-*Discovery*) after HE image feature is extracted: 

* Step0: HE image feature extraction
* Step1: **Training** FineST on the within spots
* Step2: Super-resolution spatial RNA-seq **imputation**
* Step3: Fine-grained LR pair and CCC pattern **discovery**

.. It comprises two main steps:

.. 1. global selection `spatialdm_global` to identify significantly interacting LR pairs;
.. 2. local selection `spatialdm_local` to identify local spots for each interaction.

Installation
============

FineST is available through `PyPI <https://pypi.org/project/FineST/>`_.
To install, type the following command line and add ``-U`` for updates:

.. code-block:: bash

   pip install -U FineST

Alternatively, install from this GitHub repository for latest (often
development) version (time: < 1 min):

.. code-block:: bash

   pip install -U git+https://github.com/StatBiomed/FineST

Installation using Conda
========================

.. code-block:: bash

   $ git clone https://github.com/StatBiomed/FineST.git
   $ conda create --name FineST python=3.8
   $ conda activate FineST
   $ cd FineST
   $ pip install -r requirements.txt

Typically installation is completed within a few minutes. 
Then install pytorch, refer to `pytorch installation <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

   $ conda install pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch

Verify the installation using the following command:

.. code-block:: text

   python
   >>> import torch
   >>> print(torch.__version__)
   >>> print(torch.cuda.is_available())


Get Started for *Visium* or *Visium HD* data
============================================

**Usage illustrations**: 

The source codes for reproducing the FineST analysis in this work are provided (see `demo` directory).
All relevant materials involved in the reproducing codes are available 
from `Google Drive <https://drive.google.com/drive/folders/10WvKW2EtQVuH3NWUnrde4JOW_Dd_H6r8>`_.

* For *Visium*, using a single slice of 10x Visium human nasopharyngeal carcinoma (NPC) data.
* For *Visium HD*, using a single slice of 10x Visium HD human colorectal cancer (CRC) data with 16-um bin.


Step0: HE image feature extraction (for *Visium*)
-------------------------------------------------

*Visium* measures about 5k spots across the entire tissue area. 
The diameter of each individual spot is roughly 55 micrometers (um), 
while the center-to-center distance between two adjacent spots is about 100 um.
In order to capture the gene expression profile across the whole tissue ASAP, 

**Firstly**, interpolate ``between spots`` in horizontal and vertical directions, 
using ``Spot_interpolate.py``.

.. code-block:: bash

   python ./FineST/demo/Spot_interpolate.py \
      --data_path ./Dataset/NPC/ \
      --position_list tissue_positions_list.csv \
      --dataset patient1 

.. ``Spot_interpolate.py`` also output the execution time and spot number ratio:

.. * The spots feature interpolation time is: 2.549 seconds
.. * # of interpolated between-spots are: 2.786 times vs. original within-spots
.. * # 0f final all spots are: 3.786 times vs. original within-spots
   
with **Input:**  ``tissue_positions_list.csv`` - Locations of ``within spots`` (n), 
and **Output:**  ``_position_add_tissue.csv``- Locations of ``between spots`` (m ~= 3n).


.. **Input file:**

.. * ``tissue_positions_list.csv``: Spot locations

.. **Output files:**

.. * ``_position_add_tissue.csv``: Spot locations of the ``between spots`` (m ~= 3n)
.. * ``_position_all_tissue.csv``: Spot locations of all ``between spots`` and ``within spots``

**Then** extracte the ``within spots`` HE image feature embeddings using ``HIPT_image_feature_extract.py``.

.. code-block:: bash

   python ./FineST/demo/HIPT_image_feature_extract.py \
      --dataset AH_Patient1 \
      --position ./Dataset/NPC/patient1/tissue_positions_list.csv \
      --image ./Dataset/NPC/patient1/20210809-C-AH4199551.tif \
      --output_path_img ./Dataset/NPC/HIPT/AH_Patient1_pth_64_16_image \
      --output_path_pth ./Dataset/NPC/HIPT/AH_Patient1_pth_64_16 \
      --patch_size 64 \
      --logging_folder ./Logging/HIPT_AH_Patient1/

.. ``HIPT_image_feature_extract.py`` also output the execution time:

.. * The image segment execution time for the loop is: 3.493 seconds
.. * The image feature extract time for the loop is: 13.374 seconds


.. **Input files:**

.. * ``20210809-C-AH4199551.tif``: Raw histology image
.. * ``tissue_positions_list.csv``: "Within spot" (Original in_tissue spots) locations

.. **Output files:**

.. * ``AH_Patient1_pth_64_16_image``: Segmeted "Within spot" histology image patches (.png)
.. * ``AH_Patient1_pth_64_16``: Extracted "Within spot" image feature embeddiings for each patche (.pth)


**Similarlly**, extracte the ``between spots`` HE image feature embeddings using ``HIPT_image_feature_extract.py``.

.. code-block:: bash

   python ./FineST/demo/HIPT_image_feature_extract.py \
      --dataset AH_Patient1 \
      --position ./Dataset/NPC/patient1/patient1_position_add_tissue.csv \
      --image ./Dataset/NPC/patient1/20210809-C-AH4199551.tif \
      --output_path_img ./Dataset/NPC/HIPT/NEW_AH_Patient1_pth_64_16_image \
      --output_path_pth ./Dataset/NPC/HIPT/NEW_AH_Patient1_pth_64_16 \
      --patch_size 64 \
      --logging_folder ./Logging/HIPT_AH_Patient1/

The image segment execution time: 8.153s, the image feature extract time: 35.499s.

**Input files:**

* ``20210809-C-AH4199551.tif``: Raw histology image 
* ``patient1_position_add_tissue.csv``: "Between spot" (Interpolated spots) locations

**Output files:**

* ``NEW_AH_Patient1_pth_64_16_image``: Segmeted "Between spot" histology image patches (.png)
* ``NEW_AH_Patient1_pth_64_16``: Extracted "Between spot" image feature embeddiings for each patche (.pth)


Step0: HE image feature extraction (for *Visium HD*)
----------------------------------------------------

*Visium HD* captures continuous squares without gaps, it measures the whole tissue area.

.. code-block:: bash

   python ./FineST/demo/HIPT_image_feature_extract.py \
      --dataset HD_CRC_16um \
      --position ./Dataset/CRC/square_016um/tissue_positions.parquet \
      --image ./Dataset/CRC/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
      --output_path_img ./Dataset/CRC/HIPT/HD_CRC_16um_pth_32_16_image \
      --output_path_pth ./Dataset/CRC/HIPT/HD_CRC_16um_pth_32_16 \
      --patch_size 32 \
      --logging_folder ./Logging/HIPT_HD_CRC_16um/

The image segment execution time: 62.491s, the image feature extract time: 1717.818s.

**Input files:**

* ``Visium_HD_Human_Colon_Cancer_tissue_image.btf``: Raw histology image (.btf *Visium HD* or .tif *Visium*)
* ``tissue_positions.parquet``: Spot/bin locations (.parquet *Visium HD* or .csv *Visium*)

**Output files:**

* ``HD_CRC_16um_pth_32_16_image``: Segmeted histology image patches (.png)
* ``HD_CRC_16um_pth_32_16``: Extracted image feature embeddiings for each patche (.pth)


Step1: Training FineST on the within spots
==========================================

On *Visium* dataset, if trained weights (i.e. **weight_save_path**) have been obtained, just run the following command.
Otherwise, if you want to re-train a model, just omit **weight_save_path** line.

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
* ``AH_Patient1_pth_64_16``: Image feature folder from HIPT ``HIPT_image_feature_extract.py``.
* ``position_order.csv``: Ordered tissue positions list, according to image patches' coordinates.
* ``harmony_matrix.npy``: Ordered gene expression matrix, according to image patches' coordinates.
* ``20240125140443830148``: The trained weights. Just omit it if you want to newly train a model.

**Output files:**

* ``Finetune``: The logging results ``model.log`` and trained weights ``epoch_50.pt`` (.log and .pt)
* ``Figures``: The visualization plots, used to see whether the model trained well or not (.pdf)


Step2: Super-resolution spatial RNA-seq imputation
==================================================

For *sub-spot* resolution
-------------------------

This step supposes that the trained weights (i.e. **weight_save_path**) have been obtained, just run the following.

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

For *single-cell* resolution
----------------------------

Using ``sc Patient1 pth 16 16`` 
i.e., the image feature of single-nuclei from ``HIPT_image_feature_extract.py``, just run the following.

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


Step3: Fine-grained LR pair and CCC pattern discovery
=====================================================

This step is based on `SpatialDM <https://github.com/StatBiomed/SpatialDM>`_ and `SparseAEH <https://github.com/jackywangtj66/SparseAEH>`_ (developed by our Lab). 

 * SpatialDM: for significant fine-grained ligand-receptor pair selection.
 * SparseAEH: for fastly cell-cell communication pattern discovery, 1000 times speedup to `SpatialDE <https://github.com/Teichlab/SpatialDE>`_.


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

.. _Super-resolved ligand-receptor interaction discovery by FineST (For Visium HD dataset): docs/source/CRC_LRI_CCC.ipynb


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


Contact Information
===================

Please contact Lingyu Li (`lingyuli@hku.hk <mailto:lingyuli@hku.hk>`_) or Yuanhua Huang (`yuanhua@hku.hk <mailto:yuanhua@hku.hk>`_) if any enquiry.

