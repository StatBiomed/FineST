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
sys.path.insert(0, f"{HERE.parent.parent}")
sys.path.insert(0, os.path.abspath("_ext"))
sys.path.insert(0, os.path.abspath('.'))

# -- Retrieve notebooks ------------------------------------------------

# from urllib.request import urlretrieve

# notebooks_url = "https://github.com/LingyuLi-math/FineST/tree/main/tutorial/"
# notebooks = [
#     # "AEContraNPC1_16_LRgene_clear_0618pvalue.ipynb",
#     # "scAEContraNPC1_16_LRgene_clear_0604.ipynb"
#     "NPC_Train_Impute.ipynb"
# ]
# for nb in notebooks:
#     try:
#         urlretrieve(notebooks_url + nb, nb)
#     except:
#         raise ValueError(f'{nb} cannot be retrieved.')

# -- Project information -----------------------------------------------------

project = 'FineST'
copyright = '2024, Lingyu Li'
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
release = '0.0.1'

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
    # "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    # "sphinx_autodoc_typehints",
    "nbsphinx",
    # "edit_on_github",
]

autosummary_generate = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = ['.rst']
## don't add '.ipynb' for nbsphinx>=0.8.7

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
github_repo = 'FineST'
github_nb_repo = 'FineST'
html_theme_options = dict(navigation_depth=1, titles_only=True)

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'FineST'