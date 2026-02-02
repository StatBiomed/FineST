Nuclei segmentation
===================

ROI image cropping and nuclei segmentation
------------------------------------------

**Usage illustrations**: 

* For *Visium*, use one slice - 10x Visium human nasopharyngeal carcinoma (NPC) data.

* For *Visium HD*, use one slice - 10x Visium HD human colorectal cancer (CRC) data with 16um bin.


Step0: For *Visium*
------------------------------------------------------------

.. code-block:: bash

    conda activate FineST
    time python ./FineST/FineST/demo/StarDist_nuclei_segmente.py \
        --tissue NPC1_allspot_p075_test \
        --out_dir ./FineST/FineST_local/Dataset/NPC/StarDist/DataOutput \
        --adata_path ./FineST/FineST_local/Dataset/ImputData/patient1/patient1_adata_imput_all_spot.h5ad \
        --img_path ./FineST/FineST_local/Dataset/NPC/patient1/20210809-C-AH4199551.tif \
        --prob_thresh 0.75

``StarDist_nuclei_segmente.py`` will cost `4m26.463s` in this dataset.

**Input file:**

* ``NPC1_allspot_p075_test``: The name of setted output file folder
* ``out_dir``: The pathway that output, the above level of setted output file folder
* ``adata_path``: The pathway of ``.h5ad`` adata file
* ``img_path``: The pathway of ``.tif`` or ``.btf`` HE image file with high-resolution

**Output files (saved in tissue NPC1_allspot_p075_test):**

* ``nuclei_segmentation.png``: figure includes ``HE`` image, ``Nuclei Segmentation`` image and ``Cell count``
* ``sp_adata_ns.h5ad``: The segmentated ``.h5ad`` adata file, which contains the coordinates of each nuclei
* ``logs.log``: the logging file of running ``StarDist_nuclei_segmente.py`` every time


Step0: For *Visium HD*
---------------------------------------------------------------

Here, the HE image of Visium HD (CRC) ``> 10 GB``, nuclei-segmentation is limited by storage, 
and the measured region for CRC Visium HD dataset is much less than the given HE mage (~1/6). 

**First**, Crop the Region of interest (ROI) image with corresponding adata for nuclei-segmentation. 

.. code-block:: bash

    conda activate FineST
    python ./FineST/FineST/FineST/demo/StarDist_nuclei_segmente.py \
        --tissue CRC16um_ROI_test \
        --out_dir ./FineST/FineST_local/Dataset/CRC16um/StarDist/DataOutput \
        --roi_path ./VisiumHD/Dataset/Colon_Cancer/ResultsROIs/ROI4.csv \
        --adata_path ./VisiumHD/Dataset/Colon_Cancer_square_016um.h5ad \
        --img_path ./VisiumHD/Dataset/Colon_Cancer/Visium_HD_Human_Colon_Cancer_tissue_image.btf

``StarDist_nuclei_segmente.py`` will cost `1m29.716s` in this task.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/nuclei_segmentation_ROI4.png?raw=true
   :width: 600px
   :align: center


**Additionally**, the following script provides the achievement of cropping the measured/whole image from one big HE image,
where ``SelectedShapes.csv`` is the selected adata-measured region.

.. code-block:: bash

   conda activate FineST
   python ./FineST/FineST/FineST/demo/StarDist_nuclei_segmente.py \
       --tissue CRC_human_ROI \
       --out_dir ./FineST/FineST_local/Dataset/CRC16um/StarDist/DataOutput \
       --roi_path ./VisiumHD/Dataset/Colon_Cancer/ResultsROIs/SelectedShapes.csv \
       --adata_path ./VisiumHD/Dataset/Colon_Cancer_square_016um.h5ad \
       --img_path ./VisiumHD/Dataset/Colon_Cancer/Visium_HD_Human_Colon_Cancer_tissue_image.btf


The Visium HD dataset (CRC 16um bin) can be downloaded from CRC16um in `Goole Drive <https://drive.google.com/drive/folders/1XQiRCyZv_xFrjjHMc3TrQ-R_srSwnGLE?dmr=1&ec=wgc-drive-globalnav-goto>`_ .

* where ``ROI4.csv`` and ``SelectedShapes.csv`` are two coordinate files used in this illustration.
* `ROI1.csv`, `ROI2.csv` and `ROI3.csv` are other three ROIs in paper,  using `napari` package.
* `Rec1.csv`, `Rec2.csv` and `Rec3.csv` are rectangular regions in paper,  using `napari` package. 
* `Colon_Cancer_square_016um.h5ad` can be found at `figshare <https://figshare.com/articles/dataset/FineST_supplementary_data/26763241>`_ .