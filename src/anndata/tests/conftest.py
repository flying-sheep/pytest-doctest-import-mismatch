from __future__ import annotations

import warnings

import anndata

# TODO: Should be done in pyproject.toml, see anndata/conftest.py
warnings.filterwarnings("ignore", category=anndata.OldFormatWarning)

# TODO: remove once we extricated test utils and tests
collect_ignore = ["helpers.py"]
