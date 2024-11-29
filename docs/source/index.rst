|PyPI| |Docs|

.. |PyPI| image:: https://badge.fury.io/py/FineST@2x.png
    :target: https://badge.fury.io/py/FineST
.. |Docs| image:: https://readthedocs.org/projects/finest-rtd-tutorial/badge/?version=latest
   :target: https://finest-rtd-tutorial.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
   
====
Home
====


About FineST
============

**FineST** (**Fine** -grained **S** patial **T** ranscriptomic) is a computational method and toolbox 
to identify the super-resolved spatial co-expression (i.e., spatial association) 
between a pair of ligand and receptor.

.. It pulls data from the `Open Food Facts database <https://world.openfoodfacts.org/>`_
.. and offers a *simple* and *intuitive* API.

Uniquely, **FineST** can distinguish co-expressed ligand-receptor pairs (LR pairs) 
from spatially separating pairs at sub-spot level or single-cell level, 
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

* `Interpolate between-spots among within-spots by FineST (For Visium dataset)`_.

* `Crop region of interest (ROI) from HE image by FineST (Visium or Visium HD)`_.

* `Sub-spot level (16x resolution) prediction by FineST (For Visium dataset)`_.

* `Sub-bin level (from 16um to 8um) prediction by FineST (For Visium HD dataset)`_.

* `Super-resolved ligand-receptor interavtion discovery by FineST`_.

.. _Interpolate between-spots among within-spots by FineST (For Visium dataset): Between_spot_demo.ipynb

.. _Crop region of interest (ROI) from HE image by FineST (Visium or Visium HD): Crop_ROI_image.ipynb

.. _Sub-spot level (16x resolution) prediction by FineST (For Visium dataset): NPC_Train_Impute.ipynb

.. _Sub-bin level (from 16um to 8um) prediction by FineST (For Visium HD dataset): CRC16_Train_Impute.ipynb

.. _Super-resolved ligand-receptor interavtion discovery by FineST: NPC_LRI_CCC.ipynb


.. References
.. ==========
.. FineST manuscript with more details is available on bioRxiv_ now and is currently under review.

.. .. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1/



.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   install
   StarDist_demo
   HIPT_demo
   release

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   Between_spot_demo
   Crop_ROI_image
   NPC_Train_Impute
   CRC16_Train_Impute
   NPC_LRI_CCC

