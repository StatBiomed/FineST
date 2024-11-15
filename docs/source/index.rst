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

**FineST** (Fine-grained Spatial Transcriptomic) is a statistical model and toolbox to identify the super-resolved spatial co-expression (i.e., spatial association) between a pair of ligand and receptor.
It pulls data from the `Open Food Facts database <https://world.openfoodfacts.org/>`_
and offers a *simple* and *intuitive* API.

Uniquely, **FineST** can distinguish co-expressed ligand-receptor pairs (LR pairs) from spatially separating pairs at sub-spot level or single-cell level, and identify the super-resolved interaction.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/FineST_workflow.png?raw=true
   :width: 600px
   :align: center

.. note::

   This project is under active development.

Please refer to our tutorials for details:

* `Interpolate between-spots among within-spots by FineST (For Visium dataset)`_.

* `Sub-spot level prediction by FineST (For Visium dataset)`_.

* `Super-resolved ligand-receptor interavtion discovery by FineST`_.

.. _Interpolate between-spots among within-spots by FineST (For Visium dataset): Between_spot_demo.ipynb

.. _Sub-spot level prediction by FineST (For Visium dataset): NPC_Train_Impute.ipynb

.. _Super-resolved ligand-receptor interavtion discovery by FineST: NPC_LRI_CCC.ipynb


References
==========
FineST manuscript with more details is available on bioRxiv_ now and is currently under review.

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1/



.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   install
   release

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   Between_spot_demo
   NPC_Train_Impute
   NPC_LRI_CCC

