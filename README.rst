========================================================================================================================
FineST: Contrastive learning integrates histology and spatial transcriptomics for nuclei-resolved ligand-receptor analysis
========================================================================================================================

|Paper| |Python| |PyTorch| |License| |Docs| |PyPI|

.. |Paper| image:: https://img.shields.io/badge/Paper-Nature_Communications-b31b1b.svg
   :target: https://www.nature.com/articles/s41467-026-70528-7
   :alt: Paper

.. |Python| image:: https://img.shields.io/badge/Python-3.8+-3776ab.svg
   :target: https://www.python.org/
   :alt: Python

.. |PyTorch| image:: https://img.shields.io/badge/PyTorch-1.7%2B-ee4c2c.svg
   :target: https://pytorch.org/
   :alt: PyTorch

.. |License| image:: https://img.shields.io/badge/License-Apache--2.0-green.svg
   :target: https://github.com/StatBiomed/FineST/blob/main/LICENSE
   :alt: License

.. |Docs| image:: https://img.shields.io/badge/Docs-Readthedocs-blue.svg
   :target: https://finest-rtd-tutorial.readthedocs.io
   :alt: Documentation

.. |PyPI| image:: https://img.shields.io/badge/PyPI-finest-orange.svg
   :target: https://pypi.org/project/finest/
   :alt: PyPI

🔬 **FineST** (Fine-grained Spatial Transcriptomics) is a contrastive learning framework that 
integrates **HE histology images** with **spatial transcriptomics data** to 
uncover fine-grained molecular and cellular interactions in tissue.

📋 It facilitates precise nuclei segmentation, 
high-resolution RNA expression imputation, 
and fine-grained ligand-receptor (LR) and cell-cell communication (CCC) discovery 
on **whole-slide images (WSIs)** or **regions of interest (ROIs)**.

.. image:: https://github.com/StatBiomed/FineST/blob/main/docs/fig/FineST_framework.png?raw=true
   :width: 800px
   :align: center


.. contents:: **Quick Navigation**
   :local:
   :depth: 2

|

What is FineST?
===============

Overview
--------

📊 **Core applications**

* 📈 **Imputation** — recover weak or missing gene signals using HE image context
* 🔬 **Resolution** — refine Visium spots to sub-spot / single-cell, Visium HD 16-µm bin to 8-µm bin
* 🔗 **Discovery** — identify fine-grained LR pairs and CCC patterns at super resolution (7 or 8-µm)

🎯 **Key capabilities**

* 💰 **Cost-efficient** — leverage existing HE images; no extra sequencing required for imputation
* 🖼️ **Morphology-aware** — contrastive learning links HE cell morphology to gene expression
* ⚡ **Multi-resolution** — enhance spot/bin resolution to sub-spot, single-cell, or 8-µm bins
* 🌍 **Broad applicability** — supports Visium, Visium HD datasets for WSI- or ROI-based analysis

🧠 **How it works**

FineST follows a four-step pipeline:

1. 🖼️ **Step 0** — HE image feature extraction (HIPT / Virchow2)
2. 🔄 **Step 1** — **Training** FineST model on within-spot or 16-µm bin expression
3. 📐 **Step 2** — Super-resolution **Imputation** at sub-spot or single-cell level
4. 💬 **Step 3** — Fast **Discovery** of LR pairs and CCC patterns (SpatialDM + SparseAEH)

Supported ST platforms and image encoders
-----------------------------------------

+------------------------+----------------------------------------------------------------------------------+---------------------------------------------------------------------+
| Capability             | Visium (sparse>80%)                                                              | Visium HD (sparse>90%)                                              |
+========================+==================================================================================+=====================================================================+
| Signal imputation      | Impute spot-level gene expression                                                | Impute 16-µm bin gene expression                                    |
+------------------------+----------------------------------------------------------------------------------+---------------------------------------------------------------------+
| Resolution enhancement | 55-µm → 7/8-µm: sub-spot or single-cell, also support between-spot interpolation | 16-µm → 7/8-µm: sub-bin or single-cell                              |
+------------------------+----------------------------------------------------------------------------------+---------------------------------------------------------------------+
| Fine-grained discovery | 1. Cell-type deconvolution, 2. LR interaction, 3. CCC pattern                    | 4. Pathway enrichment, 5. Cell colocalization, 6. L-R-TF-TG program |
+------------------------+----------------------------------------------------------------------------------+---------------------------------------------------------------------+

+-----------------------------------------+------+---------------------------------------------------------+--------------------------+---------------------------------+
| Histology foundation model              | Dim  | Visium                                                  | Visium HD                | FineST-enhanced                 |
+=========================================+======+=========================================================+==========================+=================================+
| HIPT (Publicly available; Quick start)  | 384  | patch 64-pix (55-µm spot), need rescale to 0.5um/pixel  | patch 32-pix (16-µm bin) | → tile 16-pixel (8-µm sub-spot) |
+-----------------------------------------+------+---------------------------------------------------------+--------------------------+---------------------------------+
| Virchow2 (Require Token, Paper setting) | 1280 | patch 112-pix (55-µm spot), need rescale to 0.5um/pixel | patch 28-pix (16-µm bin) | → tile 14-pixel (7-µm sub-spot) |
+-----------------------------------------+------+---------------------------------------------------------+--------------------------+---------------------------------+


Installation
============

🔧 **Environment setup (Prerequisites)**

* 🖥️ **OS**: Linux (Ubuntu recommended)
* 🐍 **Python**: 3.8+
* 🎮 **GPU**: NVIDIA GPU with CUDA strongly recommended (A100 used for FineST paper)
* 🔥 **PyTorch**: 1.7+ with CUDA (install separately; see `PyTorch <https://pytorch.org/get-started/locally/>`_)

Conda (recommended)
-------------------

.. code-block:: bash

   git clone https://github.com/StatBiomed/FineST.git
   conda create --name FineST python=3.8
   conda activate FineST
   cd FineST
   pip install -r requirements.txt

Verify:

.. code-block:: bash

   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

PyPI
----

.. code-block:: bash

   pip install -U FineST

   ## Alternatively, install from GitHub for latest version:
   pip install -U git+https://github.com/StatBiomed/FineST

**Note:** To run the Jupyter notebook tutorials, register this environment as a kernel:

.. code-block:: bash

   python -m pip install ipykernel
   python -m ipykernel install --user --name=FineST

Quick start
===========

🗂️ **Project Structure (Repository layout)**

.. code-block:: text

   FineST/
   ├── FineST/              # Python package (model, inference, plottings, ...)
   ├── demo/                # Command-line scripts (Step 0–2)
   ├── docs/source/         # Jupyter notebooks (Visium, Visium HD, LR, CCC, ...)
   ├── parameter/           # Model hyperparameter JSON files
   ├── docs/fig/            # Framework figures
   └── test_demo.sh         # Quick demo script


Download demo datasets
----------------------

📥 **Visium tutorial data** (*FineST_tutorial_data*) is available on `Google Drive <https://drive.google.com/drive/folders/10WvKW2EtQVuH3NWUnrde4JOW_Dd_H6r8?usp=sharing>`_.

.. code-block:: bash

   pip install gdown
   gdown --folder https://drive.google.com/drive/folders/1rZ235pexAMVvRzbVZt1ONOu7Dcuqz5BD?usp=drive_link

📥 **Visium HD demo data** (*Dataset/CRC16um*) is available on `10x Genomics - Sample P2 CRC <https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression/dataset-human-crc>`_.

.. code-block:: bash

   wget https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer_P2/Visium_HD_Human_Colon_Cancer_P2_tissue_image.btf
   wget https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer_P2/Visium_HD_Human_Colon_Cancer_P2_spatial.tar.gz
   wget https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer_P2/Visium_HD_Human_Colon_Cancer_P2_binned_outputs.tar.gz


Experienced bioinformatics users
--------------------------------

🚀 **Command-line demo** (~10 min, Visium + HIPT)

.. code-block:: bash

   bash test_demo.sh

This script runs Step 0–2 (HIPT feature extraction → training → between-spot interpolation).

Bioinformatics beginners
------------------------

⚡ **Jupyter Notebook tutorials** (recommended first run)

🧬 **Visium end-to-end (~10 min)**

* HIPT: `NPC_Train_Impute_count_HIPT.ipynb <docs/source/NPC_Train_Impute_count_HIPT.ipynb>`_
* Virchow2: `NPC_Train_Impute_count_virchow2.ipynb <docs/source/NPC_Train_Impute_count_virchow2.ipynb>`_

🗺️ **Visium HD end-to-end (~1–3 hours, large data)**

* HIPT: `CRC16_Train_Impute_count_HIPT.ipynb <docs/source/CRC16_Train_Impute_count_HIPT.ipynb>`_
* Virchow2: `CRC16_Train_Impute_count_virchow2.ipynb <docs/source/CRC16_Train_Impute_count_virchow2.ipynb>`_

💬 **LR / CCC discovery (after imputation)**

* Visium: `NPC_LRI_CCC_count.ipynb <docs/source/NPC_LRI_CCC_count.ipynb>`_
* Visium HD: `CRC_LRI_CCC_count.ipynb <docs/source/CRC_LRI_CCC_count.ipynb>`_

✂️ **ROI-based analysis (~1 min)** 

* ROI selection and cropping: `Crop_ROI_Boundary_image.ipynb <docs/source/Crop_ROI_Boundary_image.ipynb>`_

Step-by-step tutorials
======================

📚 **Tutorials and scripts organized by task.** For the complete online manual, see `FineST tutorial <https://finest-rtd-tutorial.readthedocs.io>`_.

Visium (NPC demo)
-----------------

* **Imputation + 8µm enhancement (HIPT):** `NPC_Train_Impute_count_HIPT.ipynb <docs/source/NPC_Train_Impute_count_HIPT.ipynb>`_
* **Imputation + 7µm enhancement (Virchow2):** `NPC_Train_Impute_count_virchow2.ipynb <docs/source/NPC_Train_Impute_count_virchow2.ipynb>`_
* **Between-spot interpolation:** `Between_spot_demo.ipynb <docs/source/Between_spot_demo.ipynb>`_
* **LR pair & CCC discovery:** `NPC_LRI_CCC_count.ipynb <docs/source/NPC_LRI_CCC_count.ipynb>`_
* **Cell-type deconvolution:** `transDeconv_NPC_count.ipynb <docs/source/transDeconv_NPC_count.ipynb>`_
* **Performance evaluation:** `NPC_Evaluate.ipynb <docs/source/NPC_Evaluate.ipynb>`_

Visium HD (CRC 16µm demo)
-------------------------

* **Imputation + 8µm enhancement (HIPT):** `CRC16_Train_Impute_count_HIPT.ipynb <docs/source/CRC16_Train_Impute_count_HIPT.ipynb>`_
* **Imputation + 7µm enhancement (Virchow2):** `CRC16_Train_Impute_count_virchow2.ipynb <docs/source/CRC16_Train_Impute_count_virchow2.ipynb>`_
* **LR pair & CCC discovery:** `CRC_LRI_CCC_count.ipynb <docs/source/CRC_LRI_CCC_count.ipynb>`_
* **Cell-type deconvolution:** `transDeconv_CRC_count.ipynb <docs/source/transDeconv_CRC_count.ipynb>`_

Command-line workflow
=====================

🔄 **End-to-end workflow:**

.. code-block:: text

   Step 0  🖼️  HE image feature extraction     demo/Image_feature_extraction.py
               (Additional: Spot_interpolation.py / StarDist_nuclei_segmente.py)
   Step 1  🧠  Train on within-spot / 16µm     demo/Step1_FineST_train_infer.py
   Step 2  📐  Super-resolution imputation     demo/Step2_High_resolution_imputation.py
   Step 3  💬  LR pair & CCC discovery         docs/source/*_LRI_CCC_count.ipynb


⚙️ Step 0: HE image feature extraction
-------------------------------------- 

🖼️ **Visium — within-spots**

.. code-block:: bash

   ## HIPT (recommended)
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
      --scale 0.789  

   ## Virchow2 (requires Hugging Face token)
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
      --scale 0.789  


🗺️ **Visium HD — 16-µm bins**

.. code-block:: bash

   ## HIPT (recommended)
   python ./demo/Image_feature_extraction.py \
      --dataset HD_CRC_16um \
      --position_path Dataset/CRC16um/square_016um/tissue_positions.parquet \
      --rawimage_path Dataset/CRC16um/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
      --scale_image True \
      --method HIPT \
      --patch_size 32 \
      --output_img Dataset/CRC16um/HIPT/HD_CRC_16um_pth_32_16_image \
      --output_pth Dataset/CRC16um/HIPT/HD_CRC_16um_pth_32_16 \
      --logging Dataset/CRC16um/HIPT/Logging/ \
      --scale 0.548  

   ## Virchow2 (requires Hugging Face token)
   python ./demo/Image_feature_extraction.py \
      --dataset HD_CRC_16um \
      --position_path Dataset/CRC16um/square_016um/tissue_positions.parquet \
      --rawimage_path Dataset/CRC16um/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
      --scale_image True \
      --method Virchow2 \
      --patch_size 28 \
      --output_img Dataset/CRC16um/Virchow2/HD_CRC_16um_pth_28_14_image \
      --output_pth Dataset/CRC16um/Virchow2/HD_CRC_16um_pth_28_14 \
      --logging Dataset/CRC16um/Virchow2/Logging/ \
      --scale 0.548  

FineST standardizes image resolution to **0.5 µm/pixel** using ``--scale`` before patch extraction.
Only takes effect when ``scale_image=True``.

**Formula:** ``--scale = microns_per_pixel / 0.5``

* **Visium (NPC demo)**

  - Get ``microns_per_pixel`` from ``scalefactors_json.json``:
    ``microns_per_pixel = 55 / spot_diameter_fullres``
  - Example (``Dataset/NPC/patient1/``): ``spot_diameter_fullres = 139.45``
    → ``55 / 139.45 ≈ 0.394`` µm/px → ``--scale ≈ 0.789``
  - NPC commands use ``scale_image=False``, so ``--scale`` is **not applied**
    (value kept in commands for reference when you enable rescaling).

* **Visium HD (CRC 16 µm)**

  - Read ``microns_per_pixel`` directly from ``scalefactors_json.json``
  - Example (``Dataset/CRC16um/square_016um/``): ``0.274`` µm/px
    → ``--scale = 0.274 / 0.5 ≈ 0.548``
  - Set ``scale_image=True`` so ``--scale`` is applied.


🧠 Step 1: Training FineST model
--------------------------------

🖼️ **Visium — within-spots**

.. code-block:: bash

   ## HIPT with Visium16 (patch_size=64)
   python ./demo/Step1_FineST_train_infer.py \
      --system_path './FineST/' \
      --parame_path 'parameter/parameters_NPC_HIPT.json' \
      --dataset_class 'Visium16' \
      --image_class 'HIPT' \
      --gene_selected 'CD70' \
      --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
      --image_embed_path 'FineST_tutorial_data/ImgEmbeddings/pth_64_16' \
      --do_scale False \
      --weight_w 0.5

   ## Virchow2 with Visium64 (patch_size=112)
   python ./demo/Step1_FineST_train_infer.py \
      --system_path './FineST/' \
      --parame_path 'parameter/parameters_NPC_virchow2.json' \
      --dataset_class 'Visium64' \
      --image_class 'Virchow2' \
      --gene_selected 'CD70' \
      --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
      --image_embed_path 'FineST_tutorial_data/ImgEmbeddings/pth_112_14' \
      --do_scale False \
      --weight_w 0.5


🗺️ **Visium HD — 16-µm bins**

.. code-block:: bash

   ## HIPT with VisiumHD (patch_size=32)
   python ./demo/Step1_FineST_train_infer.py \
      --system_path './' \
      --parame_path 'parameter/parameters_CRC16_HIPT.json' \
      --dataset_class 'VisiumHD' \
      --image_class 'HIPT' \
      --gene_selected 'SPP1' \
      --visium_path 'Dataset/CRC16um/square_016um/tissue_positions.parquet' \
      --image_embed_path 'Dataset/CRC16um/HIPT/HD_CRC_16um_pth_32_16' \
      --do_scale False \
      --weight_w 0.5

   ## Virchow2 with VisiumHD (patch_size=28)
   python ./demo/Step1_FineST_train_infer.py \
      --system_path './' \
      --parame_path 'parameter/parameters_CRC16_virchow2.json' \
      --dataset_class 'VisiumHD' \
      --image_class 'Virchow2' \
      --gene_selected 'SPP1' \
      --visium_path 'Dataset/CRC16um/square_016um/tissue_positions.parquet' \
      --image_embed_path 'Dataset/CRC16um/Virchow2/HD_CRC_16um_pth_28_14' \
      --do_scale False \
      --weight_w 0.5

**Key parameters**

* **Must match Step 0**

  - ``--dataset_class`` — sub-spot tiling: ``Visium16`` (HIPT, 16 tiles),
    ``Visium64`` (Virchow2, 64 tiles), ``VisiumHD`` (Visium HD)
  - ``--image_class`` — image encoder: ``HIPT`` (384-dim) or ``Virchow2`` (1280-dim);
    must be the same method used in Step 0

* **Imputation blending** (shown in commands above; adjust as needed)

  - ``--do_scale`` (default ``False``) — z-score expression before combining
    image-inferred (``adata_infer``) and neighbor-smoothed (``adata_smooth``) signals
  - ``--weight_w`` (default ``0.5``) — blend weight:
    ``adata_imput = weight_w × adata_infer + (1 - weight_w) × adata_smooth``

* **Auto-inferred** (usually omit from command line)

  - Output directories ``OrderData/``, ``Figures/``, ``SaveData/`` are derived from
    ``--image_embed_path``.
  - LR genes default to the bundled human list: ``--LRgene_path 'LR_genes'``
  - Users can specify the LR gene file explicitly, e.g.:
    ``--LRgene_path 'FineST/datasets/LR_gene/LRgene_CellChatDB_baseline_human.csv'``


📐 Step 2: Super-resolution imputation (Visium)
------------------------------------------------

**Enhance spatial resolution**

* Option A — sub-spot level (geometric segmentation within each spot)
* Option B — single-cell level (nuclei segmentation with ``StarDist``)

For **Visium** (~5k spots; 55-µm spot diameter; 100-µm center-to-center distance), interpolate additional spots between original spots first to increase spatial coverage (~3× spots), then extract between-spot image features and impute.

Interpolate between spots
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ## Interpolate spots in horizontal and vertical directions
   python ./demo/Spot_interpolation.py \
      --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv

Option A: Sub-spot resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step A1: Extract image features for between-spots

.. code-block:: bash

   ## HIPT (recommended)
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
      --scale 0.789

   ## Virchow2 (requires Hugging Face token)
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
      --scale 0.789

Step A2: Impute at sub-spot resolution

Requires the Step 1 weights folder (``--weight_save_path``). Replace ``weights[timestamp]`` with your actual folder name.

.. code-block:: bash

   ## HIPT with Visium16
   python ./demo/Step2_High_resolution_imputation.py \
      --system_path './FineST/' \
      --parame_path 'parameter/parameters_NPC_HIPT.json' \
      --dataset_class 'Visium16' \
      --gene_selected 'CD70' \
      --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
      --imag_within_path 'FineST_tutorial_data/ImgEmbeddings/pth_64_16' \
      --imag_betwen_path 'FineST_tutorial_data/ImgEmbeddings/NEW_pth_64_16' \
      --weight_save_path 'FineST_tutorial_data/Figures/weights[timestamp]'

   ## Virchow2 with Visium64
   python ./demo/Step2_High_resolution_imputation.py \
      --system_path './FineST/' \
      --parame_path 'parameter/parameters_NPC_virchow2.json' \
      --dataset_class 'Visium64' \
      --gene_selected 'CD70' \
      --visium_path 'FineST_tutorial_data/spatial/tissue_positions_list.csv' \
      --imag_within_path 'FineST_tutorial_data/ImgEmbeddings/pth_112_14' \
      --imag_betwen_path 'FineST_tutorial_data/ImgEmbeddings/NEW_pth_112_14' \
      --weight_save_path 'FineST_tutorial_data/Figures/weights[timestamp]'

**Key inputs (Option A)**

* ``pth_64_16/`` or ``pth_112_14/`` — within-spot image features (Step 0)
* ``NEW_pth_64_16/`` or ``NEW_pth_112_14/`` — between-spot image features (Step A1)
* ``weights[timestamp]/`` — trained model from Step 1 (e.g., ``weights20260204191708183236``)

**Key outputs (Option A)**

* ``SaveData/adata_imput_all_subspot.h5ad`` — sub-spot level expression (~16× per spot for HIPT; ~64× for Virchow2)
* ``SaveData/adata_imput_all_spot.h5ad`` — spot-level aggregated expression (~3× spatial density after interpolation)

Option B: Single-cell (nuclei) resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run **after Option A** (needs ``adata_imput_all_spot.h5ad`` from Step A2).

**Step B1 — Nuclei segmentation**

.. code-block:: bash

   python ./demo/StarDist_nuclei_segmente.py \
      --tissue NPC_allspot_p075 \
      --out_dir FineST_tutorial_data/NucleiSegments \
      --adata_path FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad \
      --img_path FineST_tutorial_data/20210809-C-AH4199551.tif \
      --prob_thresh 0.75

Adjust ``--prob_thresh`` if segmentation is too sparse or dense (NPC demo: ``0.75``). Nuclei segmentation results are saved in ``FineST_tutorial_data/NucleiSegments/NPC_allspot_p075/``.

**Step B2 — Extract image features for single-nuclei**

.. code-block:: bash

   ## HIPT
   python ./demo/Image_feature_extraction.py \
      --dataset sc_NPC \
      --position_path FineST_tutorial_data/NucleiSegments/NPC_allspot_p075/position_all_tissue_sc.csv \
      --rawimage_path FineST_tutorial_data/20210809-C-AH4199551.tif \
      --scale_image False \
      --method HIPT \
      --patch_size 16 \
      --output_img FineST_tutorial_data/ImgEmbeddings/sc_pth_16_16_image \
      --output_pth FineST_tutorial_data/ImgEmbeddings/sc_pth_16_16 \
      --logging FineST_tutorial_data/ImgEmbeddings/Logging/ \
      --scale 0.789

**Step B3 — Impute at single-cell resolution**

.. code-block:: bash

   python ./demo/Step2_High_resolution_imputation.py \
      --system_path './FineST/' \
      --parame_path 'parameter/parameters_NPC_HIPT.json' \
      --dataset_class 'VisiumSC' \
      --gene_selected 'CD70' \
      --image_embed_path_sc 'FineST_tutorial_data/ImgEmbeddings/sc_pth_16_16' \
      --weight_save_path 'FineST_tutorial_data/Figures/weights[timestamp]'

**Key outputs (Option B)**

* ``SaveData/adata_imput_all_sc.h5ad`` — single-nuclei resolution expression.

**Key parameters (Step 2)**

* Match Step 0/1: ``--dataset_class``, ``--parame_path``, ``--weight_save_path`` (Step 1 weights folder)
* Image features: ``--imag_within_path`` + ``--imag_betwen_path`` (Option A); ``--image_embed_path_sc`` (Option B)
* Auto-inferred: ``Figures/``, ``OrderData/``, ``SaveData/`` output paths
* LR genes: default ``--LRgene_path 'LR_genes'`` (bundled human list; same as Step 1)

**Note:** Full command chains for Options A and B are also in ``test_demo.sh``.

Visium HD (16 µm → 8 µm)
~~~~~~~~~~~~~~~~~~~~~~~~

Visium HD uses continuous bin squares and does not require spot interpolation. See the end-to-end notebook:
`CRC16_Train_Impute_count_HIPT.ipynb <docs/source/CRC16_Train_Impute_count_HIPT.ipynb>`_ or `CRC16_Train_Impute_count_virchow2.ipynb <docs/source/CRC16_Train_Impute_count_virchow2.ipynb>`_.

💬 Step 3: Fine-grained ligand-receptor interaction
---------------------------------------------------

Identify ligand-receptor interactions and communication patterns based on `SpatialDM <https://github.com/StatBiomed/SpatialDM>`_ and
`SparseAEH <https://github.com/jackywangtj66/SparseAEH>`_.

* Visium: `NPC_LRI_CCC_count.ipynb <docs/source/NPC_LRI_CCC_count.ipynb>`_
* Visium HD: `CRC_LRI_CCC_count.ipynb <docs/source/CRC_LRI_CCC_count.ipynb>`_

Perform cell-type deconvolution on super-resolved gene expression data with ``expDeconv()`` from `TransImpute <https://transpa.readthedocs.io/en/latest/transDeconv.html>`_.

* Visium: `transDeconv_NPC_count.ipynb <docs/source/transDeconv_NPC_count.ipynb>`_
* Visium HD: `transDeconv_CRC_count.ipynb <docs/source/transDeconv_CRC_count.ipynb>`_

ROI selection with Napari
=========================

To analyze a specific region of interest (ROI) on the HE image, use `napari <https://github.com/napari/napari>`_:

.. code-block:: python

   from PIL import Image
   Image.MAX_IMAGE_PIXELS = None
   import matplotlib.pyplot as plt
   import napari

   image = plt.imread("FineST_tutorial_data/20210809-C-AH4199551.tif")
   viewer = napari.view_image(image, channel_axis=2, ndisplay=2)
   napari.run()

For detailed instructions and ROI extraction, please see
| `online tutorial <https://finest-rtd-tutorial.readthedocs.io/en/latest/Crop_ROI_Boundary_image.html>`_, or 
| `video guide <https://drive.google.com/file/d/1y3sb_Eemq3OV2gkxwu4gZBhLFp-gpzpH/view?usp=sharing>`_.

**Quick guide:**

* A **shapes** layer is automatically added when opening napari
* Use the ``Add Polygons`` tool to draw ROI(s) on the HE image
* Optionally rename the ROI layer for clarity


FineST also supports extracting cropped image and AnnData with ``fst.crop_img_adata()``
(see `Crop_ROI_Boundary_image.ipynb <docs/source/Crop_ROI_Boundary_image.ipynb>`_).

Citation and Contact
====================

If you use FineST in your research, please cite:

Li, L., Wang, T., Liang, Z., Yu, H., Ma, S., Yu, L., & Huang, Y. (2026).
FineST: contrastive learning integrates histology and spatial transcriptomics for
nuclei-resolved ligand-receptor analysis.
`Nature Communications <https://www.nature.com/articles/s41467-026-70528-7>`_, 17(1), 4645.

.. code-block:: text

   @article{li2026finest,
     title={FineST: contrastive learning integrates histology and spatial transcriptomics for nuclei-resolved ligand-receptor analysis},
     author={Li, Lingyu and Wang, Tianjie and Liang, Zhuo and Yu, Huajian and Ma, Stephanie and Yu, Lequan and Huang, Yuanhua},
     journal={Nature Communications},
     volume={17},
     number={1},
     pages={4645},
     year={2026},
     publisher={Nature Publishing Group UK London}
   }

For any enquiries, please contact Dr. Lingyu Li (`lingyuli@hku.hk <mailto:lingyuli@hku.hk>`_) or Dr. Yuanhua Huang (`yuanhua@hku.hk <mailto:yuanhua@hku.hk>`_).
