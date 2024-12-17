Geometric segmentation
======================

Image feature/embedding extraction: for *Visium* or *Visium HD* data
--------------------------------------------------------------------

**Usage illustrations**: 

* For *Visium*, using a single slice of 10x Visium human nasopharyngeal carcinoma (NPC) data.

* For *Visium HD*, using a single slice of 10x Visium HD human colorectal cancer (CRC) data with 16-um bin.


Step0: HE image feature extraction (for *Visium*)
-------------------------------------------------

*Visium (v2)* measures about 5k spots across the entire tissue area. 
The diameter of each individual spot is roughly 55 micrometers (um), 
while the center-to-center distance between two adjacent spots is about 100 um.
In order to capture the gene expression profile across the whole tissue ASSP, 

**Firstly**, interpolate ``between spots`` in horizontal and vertical directions, 
using ``Spot_interpolate.py``.

.. code-block:: bash

   python ./FineST/demo/Spot_interpolate.py \
      --data_path ./Dataset/NPC/ \
      --position_list tissue_positions_list.csv \
      --dataset patient1 

``Spot_interpolate.py`` also output the execution time and spot number ratio:

* The spots feature interpolation time is: 2.549 seconds
* # of interpolated between-spots are: 2.786 times vs. original within-spots
* # 0f final all spots are: 3.786 times vs. original within-spots


**Input file:**

* ``tissue_positions_list.csv``: Spot locations

**Output files:**

* ``_position_add_tissue.csv``: Spot locations of the ``between spots`` (m ~= 3n)
* ``_position_all_tissue.csv``: Spot locations of all ``between spots`` and ``within spots``


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

``HIPT_image_feature_extract.py`` also output the execution time:

* The image segment execution time for the loop is: 3.493 seconds
* The image feature extract time for the loop is: 13.374 seconds


**Input files:**

* ``20210809-C-AH4199551.tif``: Raw histology image
* ``tissue_positions_list.csv``: "Within spot" (Original in_tissue spots) locations

**Output files:**

* ``AH_Patient1_pth_64_16_image``: Segmeted "Within spot" histology image patches (.png)
* ``AH_Patient1_pth_64_16``: Extracted "Within spot" image feature embeddiings for each patche (.pth)


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

``HIPT_image_feature_extract.py`` also output the execution time:

* The image segment execution time for the loop is:  8.153 seconds
* The image feature extract time for the loop is: 35.499 seconds


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

   python .FineST/demo/HIPT_image_feature_extract.py \
      --dataset HD_CRC_16um \
      --position ./Dataset/CRC/square_016um/tissue_positions.parquet \
      --image ./Dataset/CRC/square_016um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
      --output_path_img ./Dataset/CRC/HIPT/HD_CRC_16um_pth_32_16_image \
      --output_path_pth ./Dataset/CRC/HIPT/HD_CRC_16um_pth_32_16 \
      --patch_size 32 \
      --logging_folder ./Logging/HIPT_HD_CRC_16um/

``HIPT_image_feature_extract.py`` also output the execution time:

* The image segment execution time for the loop is: 62.491 seconds
* The image feature extract time for the loop is: 1717.818 seconds

**Input files:**

* ``Visium_HD_Human_Colon_Cancer_tissue_image.btf``: Raw histology image (.btf *Visium HD* or .tif *Visium*)
* ``tissue_positions.parquet``: Spot/bin locations (.parquet *Visium HD* or .csv *Visium*)

**Output files:**

* ``HD_CRC_16um_pth_32_16_image``: Segmeted histology image patches (.png)
* ``HD_CRC_16um_pth_32_16``: Extracted image feature embeddiings for each patche (.pth)
