from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from functools import singledispatch

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
from scipy import sparse

from anndata import AnnData, Raw
from anndata._core.aligned_mapping import AlignedMapping
from anndata._core.views import ArrayView
from anndata.utils import asarray


# TODO: Use hypothesis for this?
def gen_adata(shape: tuple[int, int]) -> AnnData:
    return AnnData(X=np.random.binomial(100, 0.005, shape))
