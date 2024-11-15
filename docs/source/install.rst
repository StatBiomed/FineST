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
