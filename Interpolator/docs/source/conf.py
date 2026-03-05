# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '5D_Interpolator'
copyright = '2025, Wenye Song'
author = 'Wenye Song'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

language = '0.1'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": False,
}
html_static_path = ['_static']




import os
import sys

# sphinx only for python backend
sys.path.insert(0, os.path.abspath('../../backend'))
