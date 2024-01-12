from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import wraps
from inspect import Parameter, signature
from pathlib import Path
from typing import Union
from warnings import warn

import numpy as np
import pandas as pd
from packaging.version import Version
from scipy.sparse import spmatrix


class Empty:
    pass


Index1D = Union[slice, int, str, np.int64, np.ndarray]
Index = Union[Index1D, tuple[Index1D, Index1D], spmatrix]


#############################
# stdlib
#############################


if sys.version_info >= (3, 11):
    from contextlib import chdir
else:

    @dataclass
    class chdir(AbstractContextManager):
        path: Path
        _old_cwd: list[Path] = field(default_factory=list)

        def __enter__(self) -> None:
            self._old_cwd.append(Path())
            os.chdir(self.path)

        def __exit__(self, *_exc_info) -> None:
            os.chdir(self._old_cwd.pop())


if sys.version_info >= (3, 10):
    from itertools import pairwise
else:

    def pairwise(iterable):
        from itertools import tee

        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


#############################
# Dealing with uns
#############################


def _clean_uns(adata: AnnData):  # noqa: F821
    """
    Compat function for when categorical keys were stored in uns.
    This used to be buggy because when storing categorical columns in obs and var with
    the same column name, only one `<colname>_categories` is retained.
    """
    k_to_delete = set()
    for cats_name, cats in adata.uns.items():
        if not cats_name.endswith("_categories"):
            continue
        name = cats_name.replace("_categories", "")
        # fix categories with a single category
        if isinstance(cats, (str, int)):
            cats = [cats]
        for ann in [adata.obs, adata.var]:
            if name not in ann:
                continue
            codes: np.ndarray = ann[name].values
            # hack to maybe find the axis the categories were for
            if not np.all(codes < len(cats)):
                continue
            ann[name] = pd.Categorical.from_codes(codes, cats)
            k_to_delete.add(cats_name)
    for cats_name in k_to_delete:
        del adata.uns[cats_name]


def _move_adj_mtx(d):
    """
    Read-time fix for moving adjacency matrices from uns to obsp
    """
    n = d.get("uns", {}).get("neighbors", {})
    obsp = d.setdefault("obsp", {})

    for k in ("distances", "connectivities"):
        if (
            (k in n)
            and isinstance(n[k], (spmatrix, np.ndarray))
            and len(n[k].shape) == 2
        ):
            warn(
                f"Moving element from .uns['neighbors']['{k}'] to .obsp['{k}'].\n\n"
                "This is where adjacency matrices should go now.",
                FutureWarning,
            )
            obsp[k] = n.pop(k)


def _find_sparse_matrices(d: Mapping, n: int, keys: tuple, paths: list):
    """Find paths to sparse matrices with shape (n, n)."""
    for k, v in d.items():
        if isinstance(v, Mapping):
            _find_sparse_matrices(v, n, (*keys, k), paths)
        elif isinstance(v, spmatrix) and v.shape == (n, n):
            paths.append((*keys, k))
    return paths


# This function was adapted from scikit-learn
# github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py
def _deprecate_positional_args(func=None, *, version: str = "1.0 (renaming of 0.25)"):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    func
        Function to check arguments on.
    version
        The version when positional arguments will result in error.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = [
                f"{name}={arg}"
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg = ", ".join(args_msg)
            warn(
                f"Pass {args_msg} as keyword args. From version {version} passing "
                "these as positional arguments will result in an error",
                FutureWarning,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args


def _safe_transpose(x):
    """Safely transpose x

    This is a workaround for: https://github.com/scipy/scipy/issues/19161
    """

    return x.T


def _map_cat_to_str(cat: pd.Categorical) -> pd.Categorical:
    if Version(pd.__version__) >= Version("2.1"):
        # Argument added in pandas 2.1
        return cat.map(str, na_action="ignore")
    else:
        return cat.map(str)
