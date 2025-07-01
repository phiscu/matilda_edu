# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = 'Matilda ONLINE'
copyright = '2025, Alexander Georgi, Phillip Schuster, Mia Janzen'
author = 'Alexander Georgi, Phillip Schuster, Mia Janzen'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'myst_parser',        
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.mathjax'
]

myst_enable_extensions = ["dollarmath"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

def suppress_core_module_docstring(app):
    import matilda.core
    import matilda.mspot_glacier
    matilda.core.__doc__ = None
    matilda.mspot_glacier.__doc__ = None

def setup(app):
    app.connect("builder-inited", suppress_core_module_docstring)
