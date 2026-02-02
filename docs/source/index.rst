|PyPI| |Docs| |Build Status|

.. |PyPI| image:: https://badge.fury.io/py/FineST.svg
    :target: https://badge.fury.io/py/FineST
.. |Docs| image:: https://readthedocs.org/projects/finest-rtd-tutorial/badge/?version=latest
   :target: https://finest-rtd-tutorial.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |Build Status| image:: https://travis-ci.org/LingyuLi-math/FineST.svg?branch=main
   :target: https://travis-ci.org/LingyuLi-math/FineST
====
Home
====


About FineST
============

**FineST** (**Fine** -grained **S** patial **T** ranscriptomic) is a computational method 
to identify super-resolved spatial co-expression (i.e., spatial association) 
between a pair of ligand and receptor.

.. It pulls data from the `Open Food Facts database <https://world.openfoodfacts.org/>`_
.. and offers a *simple* and *intuitive* API.

Uniquely, **FineST** can distinguish co-expressed ligand-receptor pairs (LR pairs) 
from spatially separating pairs at **sub-spot level** or **single-cell level**, 
and identify the super-resolved LR interaction.

.. The effectiveness of **FineST** has been substantiated through its significant enhancement 
.. of accuracy and fidelity, utilizing 10x Visium HD data as a reliable benchmark. 
.. FineST has identified intricate tumor-immune interactions within nasopharyngeal carcinoma (NPC) Visium data.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/FineST_workflow.png?raw=true
   :width: 600px
   :align: center


.. note::

   This project is under active development.


Tutorial
========

Please refer to our tutorials for details:


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


References
==========
FineST manuscript has been submitted (not available on bioRxiv_ now). 
If people are interested in reading FineST, please contact Yuanhua Huang (`yuanhua@hku.hk <mailto:yuanhua@hku.hk>`_).

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1/



.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   install
   StarDist_demo
   HIPT_demo
   FineST_demo
   Impute_demo
   release

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   Between_spot_demo
   CRC16_Train_Impute_count
   NPC_Train_Impute_count
   NPC_LRI_CCC_count
   CRC_LRI_CCC
   transDeconv_NPC_count
   transDeconv_CRC_count
   Crop_ROI_Boundary_image
   NPC_Evaluate
   Demo_Train_Impute_count
   Demo_results_istar_check

