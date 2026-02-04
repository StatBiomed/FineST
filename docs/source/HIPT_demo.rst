Geometric segmentation
======================

Image embedding extraction
--------------------------

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
      --scale 0.5

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
      --scale 0.5

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
      --scale 0.5

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
      --scale 0.5

**Output:** Image feature embeddings (``NEW_pth_64_16`` or ``NEW_pth_112_14``) for between-spots


Step0: For *Visium*: single-cell resolution
---------------------------------------------

For single-cell resolution analysis:

1. Get ``_adata_imput_all_spot.h5ad`` from ``_Train_Impute.ipynb``
2. Get ``sp._adata_ns.h5ad`` and ``_position_all_tissue_sc.csv`` from ``StarDist_nuclei_segmentate.py``
3. Extract image features using ``Image_feature_extraction.py`` with ``Virchow2``

.. code-block:: bash

   python ./demo/Image_feature_extraction.py \
      --dataset AH_Patient1 \
      --position_path ./FineST_local/Dataset/NPC/StarDist/DataOutput/NPC1_allspot_p075_test/_position_all_tissue_sc.csv \
      --rawimage_path ./FineST_local/Dataset/NPC/patient1/20210809-C-AH4199551.tif \
      --scale_image False \
      --method Virchow2 \
      --output_img ./FineST_local/Dataset/NPC/HIPT/sc_Patient1_pth_14_14_image \
      --output_pth ./FineST_local/Dataset/NPC/HIPT/sc_Patient1_pth_14_14 \
      --patch_size 14 \
      --logging ./FineST_local/Logging/HIPT_AH_Patient1/ \
      --scale 0.5


Step0: For *Visium HD*
----------------------------------------------------

*Visium HD* captures continuous squares without gaps, it measures the whole tissue area.
For CRC dataset, the ``spot_diameter_fullres`` is 58.417 or 29.208 pixels, corresponding to 16-um and 8-um data. 
Here we use  ``scale_image`` with ``scale=0.5`` to re-scale image,
then split each 28-pixels patch_image to 14-pixels tile_image. 

.. code-block:: bash

   python ./FineST/demo/Image_feature_extraction.py \
      --dataset HD_CRC_16um \
      --position ./Dataset/CRC/square_016um/tissue_positions.parquet \
      --imagefile ./Dataset/CRC/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
      --scale_image True \
      --method Virchow2 \
      --output_path_img ./Dataset/CRC/HIPT/HD_CRC_16um_pth_28_14_image \
      --output_path_pth ./Dataset/CRC/HIPT/HD_CRC_16um_pth_28_14 \
      --patch_size 28 \
      --logging_folder ./Logging/HIPT_HD_CRC_16um/

``Image_feature_extraction.py`` also output the execution time:

* The image segment execution time for the loop is: 125.442 seconds
* The image feature extract time for the loop is: 2486.118 seconds

**Input files:**

* ``Visium_HD_Human_Colon_Cancer_tissue_image.btf``: Raw histology image (.btf *Visium HD* or .tif *Visium*)
* ``tissue_positions.parquet``: Spot/bin locations (.parquet *Visium HD* or .csv *Visium*)

**Output files:**

* ``HD_CRC_16um_pth_28_14_image``: Segmeted histology image patches (.png)
* ``HD_CRC_16um_pth_28_14``: Extracted image feature embeddiings for each patche (.pth)
