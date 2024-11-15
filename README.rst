===========================================
FineST: Fine-grained Spatial Transcriptomic
===========================================


A tatistical model and toolbox to identify the super-resolved ligand-receptor interaction 
with spatial co-expression (i.e., spatial association). 
Uniquely, FineST can distinguish co-expressed ligand-receptor pairs (LR pairs) 
from spatially separating pairs at sub-spot level or single-cell level, 
and identify the super-resolved ligand-receptor interaction (LRI).

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/FineST_framework_all.png?raw=true
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

   pip install -U git+https://github.com/LingyuLi-math/FineST

Installation using Conda
========================

.. code-block:: bash

   $ git clone https://github.com/LingyuLi-math/FineST.git
   $ conda create --name FineST python=3.8
   $ conda activate FineST
   $ cd FineST
   $ pip install -r requirements.txt

Typically installation is completed within a few minutes. 
Then install pytorch, refer to `pytorch installation <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

   $ conda install pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch

Verify the installation using the following command:

.. code-block:: bash

   python
   >>> import torch
   >>> print(torch.__version__)
   >>> print(torch.cuda.is_available())


Get Started for *Visium* or *Visium HD* data
============================================

**Usage illustrations**: 

* For *Visium*, using a single slice of 10x Visium human nasopharyngeal carcinoma (NPC) data.

* For *Visium HD*, using a single slice of 10x Visium HD human colorectal cancer (CRC) data with 16-um bin.


Step0: HE image feature extraction (for *Visium*)
-------------------------------------------------

*Visium (v2)* measures about 5k spots across the entire tissue area. 
The diameter of each individual spot is roughly 55 micrometers (um), 
while the center-to-center distance between two adjacent spots is about 100 um.

In order to capture the gene expression profile across the whole tissue ASSP, 
firstly, we will interpolate ``between spots`` in horizontal and vertical directions, 
using ``Spot_interpolate.py``.

.. code-block:: bash

   python ./FineST/Spot_interpolate.py \
      --data_path ./Dataset/NPC/ \
      --position_list tissue_positions_list.csv \
      --dataset patient1 

.. ``Spot_interpolate.py`` also output the execution time and spot number ratio:

.. * The spots feature interpolation time is: 2.549 seconds
.. * # of interpolated between-spots are: 2.786 times vs. original within-spots
.. * # 0f final all spots are: 3.786 times vs. original within-spots


**Input file:**

* ``tissue_positions_list.csv``: Spot locations

**Output files:**

* ``_position_add_tissue.csv``: Spot locations of the ``between spots`` (m ~= 3n)
* ``_position_all_tissue.csv``: Spot locations of all ``between spots`` and ``within spots``

Then extracte the ``within spots`` HE image feature embeddings using ``HIPT_image_feature_extract.py``.

.. code-block:: bash

   python ./FineST/HIPT_image_feature_extract.py \
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


Similarlly, extracte the ``between spots`` HE image feature embeddings using ``HIPT_image_feature_extract.py``.

.. code-block:: bash

   python ./FineST/HIPT_image_feature_extract.py \
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

   python .FineST/HIPT_image_feature_extract.py \
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


Step1: Training FineST on the within spots
------------------------------------------


Step2: Super-resolution spatial RNA-seq imputation
--------------------------------------------------


Step3: Fine-grained LR pair and CCC pattern discovery
-----------------------------------------------------



.. Quick example
.. =============

.. Using the build-in NPC dataset as an example, the following Python script
.. will predict super-resolution ST gene expression and compute the p-value indicating whether a certain Ligand-Receptor is
.. spatially co-expressed. 

Detailed Manual
===============

The full manual is at `finest-rtd-tutorial <https://finest-rtd-tutorial.readthedocs.io>`_ for installation, tutorials and examples. 

* `Interpolate between-spots among within-spots by FineST (For Visium dataset)`_.

* `Sub-spot level prediction by FineST (For Visium dataset)`_.

* `Super-resolved ligand-receptor interavtion discovery by FineST`_.

.. _Interpolate between-spots among within-spots by FineST (For Visium dataset): docs/source/Between_spot_demo.ipynb

.. _Sub-spot level prediction by FineST (For Visium dataset): docs/source/NPC_Train_Impute.ipynb

.. _Super-resolved ligand-receptor interavtion discovery by FineST: docs/source/NPC_LRI_CCC.ipynb

