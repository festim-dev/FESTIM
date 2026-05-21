# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("_ext"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FESTIM"
copyright = "2022-2026, FESTIM contributors"
author = "FESTIM-dev"


# -- General configuration ---------------------------------------------------

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
    "publications",
]

suppress_warnings = ["autosectionlabel.*"]

napoleon_use_ivar = True

templates_path = ["_templates"]

exclude_patterns = []

source_suffix = ".rst"

master_doc = "index"

pygments_style = None

add_module_names = False

bibtex_bibfiles = ["bibliography/references.bib"]

# -- Options for HTML output -------------------------------------------------

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
    "external_links": [
        {
            "name": "Tutorials",
            "url": "https://festim-workshop.readthedocs.io/",
        },
        {
            "name": "V&V",
            "url": "https://festim-vv-report.readthedocs.io/en/latest/",
        },
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/festim-dev/FESTIM",
            "icon": "fa-brands fa-github fa-fw",
            "type": "fontawesome",
        },
        {
            "name": "Support Forum",
            "url": "https://festim.discourse.group/",
            "icon": "fa-brands fa-discourse fa-fw",
        },
        {
            "name": "Slack",
            "url": "https://join.slack.com/t/festim-dev/shared_invite/zt-246hw8d6o-htWASLsbdosUo_2nRKCf9g",
            "icon": "fa-brands fa-slack fa-fw",
        },
    ],
    "header_links_before_dropdown": 7,
    "show_prev_next": False,
    "analytics": dict(google_analytics_id="G-SCL2TVV7BK"),
}

if datetime.date.today() < datetime.date(2026, 5, 15):
    html_theme_options["announcement"] = (
        '📢 Register now for the <a href="https://mit.zoom.us/meeting/register/PElrepT-Tpivqzu9Jepr_g#/registration" target="_blank">FESTIM 2.0 webinar on 15th May, 9:00 am EST'
    )

html_favicon = "_static/favicon.png"

html_sidebars = {
    "**": [
        "sidebar-nav-bs",
    ],
}

html_title = "FESTIM Documentation"
