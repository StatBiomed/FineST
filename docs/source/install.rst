Installation
============

Installation using Conda
------------------------

.. code-block:: bash

   git clone https://github.com/StatBiomed/FineST.git
   conda create --name FineST python=3.8
   conda activate FineST
   cd FineST
   pip install -r requirements.txt

Verify the installation using the following command:

.. code-block:: bash

   python
   >>> import torch
   >>> print(torch.__version__)
   2.1.2+cu121 (or your installed version)
   >>> print(torch.cuda.is_available())
   True

Installation using PyPI
-----------------------

FineST package is available through `PyPI <https://pypi.org/project/FineST/>`_.

.. code-block:: bash

   pip install -U FineST

Alternatively, install from GitHub for latest version:

.. code-block:: bash

   pip install -U git+https://github.com/StatBiomed/FineST

Setup Jupyter Notebook Kernel
------------------------------

The FineST conda environment can be used for Jupyter notebooks:

.. code-block:: bash

   python -m pip install ipykernel
   python -m ipykernel install --user --name=FineST

**Tutorial notebooks:**
* `NPC_Train_Impute_demo.ipynb <https://github.com/StatBiomed/FineST/tree/main/tutorial/NPC_Train_Impute_demo.ipynb>`_ (using Virchow2; requires Hugging Face token, approval may take days)
* `NPC_Train_Impute_demo_HIPT.ipynb <https://github.com/StatBiomed/FineST/blob/main/tutorial/NPC_Train_Impute_demo_HIPT.ipynb>`_ (using HIPT; recommended for quick start)
