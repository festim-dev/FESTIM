# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# Generate the map

# Add the directory containing your Python script to the Python path
sys.path.insert(0, os.path.abspath("."))

import map

m = map.generate_map()
current_dir = os.path.dirname(__file__)
html_path = os.path.join(current_dir, "_static", "map.html")

# create _static directory if it doesn't exist
os.makedirs(os.path.dirname(html_path), exist_ok=True)

m.save(html_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FESTIM"
copyright = "2022-2023, FESTIM contributors"
author = "FESTIM-dev"
release = "1.0.0"
version = "1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.images",
]

suppress_warnings = ["autosectionlabel.*"]

napoleon_use_ivar = True  # needed to correctly format class attributes

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# shorten module names in readme
add_module_names = False

# bibliography file
bibtex_bibfiles = ["bibliography/references.bib"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]

html_context = {
    "github_user": "festim-dev",
    "github_repo": "FESTIM",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_theme_options = {
    "use_edit_page_button": True,
    "logo": {
        "image_light": "images/festim logo.png",
        "image_dark": "images/festim logo dark.png",
    },
    "icon_links": [
        {
            "name": "Support Forum",
            "url": "https://festim.discourse.group/",
            "icon": "fa-brands fa-discourse",
        },
        {
            "name": "Slack",
            "url": "https://join.slack.com/t/festim-dev/shared_invite/zt-246hw8d6o-htWASLsbdosUo_2nRKCf9g",
            "icon": "fa-brands fa-slack",
        },
    ],
}

html_sidebars = {
    "**": [
        "search-field",
        "sidebar-nav-bs",
    ],
}

html_title = "FESTIM Documentation"
