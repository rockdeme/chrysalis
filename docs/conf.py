import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'chrysalis'
copyright = '2023, Demeter Túrós'
author = 'Demeter Túrós'
release = '2023'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'nbsphinx',
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = "docs_logo.svg"
html_favicon = "docs_logo.svg"

html_theme_options = {"logo_only": True}

html_css_files = ['css/custom.css',
                  ]

autodoc_exclude_members = {
    'chrysalis.core': ['detect_svgs', 'pca', 'aa'],
}

def setup(app):
    app.add_css_file("css/custom.css")