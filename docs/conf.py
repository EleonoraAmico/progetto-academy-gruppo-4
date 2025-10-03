import os
import sys
from datetime import datetime

# Add project root to sys.path for autodoc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

project = 'Project 4'
author = 'Adriano Neroni, Alessandro Venanzi, Eleonora Amico, Fabrizio Corda'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
     'members': True,
     'undoc-members': False,
     'inherited-members': False,
     'show-inheritance': False,
}

autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_inherit_docstrings = False
autoclass_content = 'class'
autosummary_generate = True

# Avoid importing heavy or unavailable dependencies during autodoc
autodoc_mock_imports = [
    'rag_qdrant_hybrid', 'qdrant_client', 'ragas', 'ddgs',
    'crewai', 'crewai.flow', 'crewai_tools', 'opik',
    'langchain', 'langchain_core', 'langchain_community', 'langchain_openai',
    'openai', 'dotenv', 'yaml', 'pydantic', 'bs4'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']


