# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime

from pathlib import Path
from sphinx.application import Sphinx
from sphinx.ext import autosummary

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
# Only add _ext to path if it exists
_ext_path = HERE / "_ext"
if _ext_path.exists():
    sys.path.insert(0, str(_ext_path))
sys.path.insert(0, os.path.abspath('.'))

# -- Retrieve notebooks ------------------------------------------------
# Note: Notebooks are already in the source directory, so downloading is disabled
# Uncomment below if you need to download notebooks from GitHub

# from urllib.request import urlretrieve
# 
# notebooks_url = "https://raw.githubusercontent.com/StatBiomed/FineST/main/docs/source/"
# notebooks = [
#     "Between_spot_demo.ipynb",
#     "CRC16_Train_Impute_count.ipynb",
#     "NPC_Train_Impute_count.ipynb",
#     "NPC_LRI_CCC_count.ipynb",
#     "CRC_LRI_CCC_count.ipynb",
#     "transDeconv_NPC_count.ipynb",
#     "transDeconv_CRC_count.ipynb",
#     "Crop_ROI_Boundary_image.ipynb",
#     "NPC_Evaluate.ipynb",
#     "Demo_Train_Impute_count.ipynb",
#     "Demo_results_istar_check.ipynb",
# ]
# for nb in notebooks:
#     try:
#         urlretrieve(notebooks_url + nb, nb)
#     except:
#         raise ValueError(f'{nb} cannot be retrieved.')

# -- Project information -----------------------------------------------------

project = 'FineST'
author = 'Lingyu Li'
title = "Fine-grained Spatial Transcriptomic"
copyright = f"{datetime.now():%Y}, {author}"

# Disable pdf and epub generation
enable_pdf_build = False
enable_epub_build = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# The full version, including alpha/beta/rc tags
release = '0.1.3'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

needs_sphinx = "1.7"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    # "edit_on_github",  # Disabled: package not available on PyPI or GitHub
]

autosummary_generate = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = ['.rst']
# Note: nbsphinx>=0.8.7 automatically handles .ipynb files, no need to add explicitly

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The root document.
root_doc = 'index'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_options = dict(navigation_depth=1, titles_only=True)

## 2026.02.03 LLY adjust the 'edit_on_github'
# Add " Edit on GitHub" link on each page using html_context
# This is the standard way to enable GitHub edit links in sphinx_rtd_theme
html_context = {
    'display_github': True,  # Enable GitHub link display
    'github_user': 'StatBiomed',  # GitHub username/organization
    'github_repo': 'FineST',  # Repository name
    'github_version': 'main',  # Branch name
    'conf_py_path': '/docs/source/',  # Path in the repository to the docs root (where conf.py is located)
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'FineST'