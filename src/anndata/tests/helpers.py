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


def _assert_equal(a, b):
    """Allows reporting elem name for simple assertion."""
    assert a == b


@singledispatch
def assert_equal(a, b, exact=False, elem_name=None):
    _assert_equal(a, b, _elem_name=elem_name)


@assert_equal.register(np.ndarray)
def assert_equal_ndarray(a, b, exact=False, elem_name=None):
    b = asarray(b)
    if not exact and is_numeric_dtype(a) and is_numeric_dtype(b):
        assert a.shape == b.shape
        assert np.allclose(a, b, equal_nan=True)
    elif (  # Structured dtype
        not exact
        and hasattr(a, "dtype")
        and hasattr(b, "dtype")
        and len(a.dtype) > 1
        and len(b.dtype) > 0
    ):
        assert_equal(pd.DataFrame(a), pd.DataFrame(b), exact, elem_name)
    else:
        assert np.all(a == b)


@assert_equal.register(ArrayView)
def assert_equal_arrayview(a, b, exact=False, elem_name=None):
    assert_equal(asarray(a), asarray(b), exact=exact, elem_name=elem_name)


@assert_equal.register(pd.DataFrame)
def are_equal_dataframe(a, b, exact=False, elem_name=None):
    if not isinstance(b, pd.DataFrame):
        assert_equal(b, a, exact, elem_name)  # , a.values maybe?


@assert_equal.register(Mapping)
def assert_equal_mapping(a, b, exact=False, elem_name=None):
    assert set(a.keys()) == set(b.keys())
    for k in a.keys():
        if elem_name is None:
            elem_name = ""
        assert_equal(a[k], b[k], exact, f"{elem_name}/{k}")


@assert_equal.register(AlignedMapping)
def assert_equal_aligned_mapping(a, b, exact=False, elem_name=None):
    a_indices = (a.parent.obs_names, a.parent.var_names)
    b_indices = (b.parent.obs_names, b.parent.var_names)
    for axis_idx in a.axes:
        assert_equal(
            a_indices[axis_idx], b_indices[axis_idx], exact=exact, elem_name=axis_idx
        )
    assert a.attrname == b.attrname
    assert_equal_mapping(a, b, exact=exact, elem_name=elem_name)


@assert_equal.register(pd.Index)
def assert_equal_index(a, b, exact=False, elem_name=None):
    if not exact:
        pd.testing.assert_index_equal(
            a, b, check_names=False, check_categorical=False, _elem_name=elem_name
        )
    else:
        pd.testing.assert_index_equal(a, b, _elem_name=elem_name)


@assert_equal.register(pd.api.extensions.ExtensionArray)
def assert_equal_extension_array(a, b, exact=False, elem_name=None):
    pd.testing.assert_extension_array_equal(
        a,
        b,
        check_dtype=exact,
        check_exact=exact,
        _elem_name=elem_name,
    )


@assert_equal.register(Raw)
def assert_equal_raw(a, b, exact=False, elem_name=None):
    def assert_is_not_none(x):  # can't put an assert in a lambda
        assert x is not None

    assert_is_not_none(b, _elem_name=elem_name)
    for attr in ["X", "var", "varm", "obs_names"]:
        assert_equal(
            getattr(a, attr),
            getattr(b, attr),
            exact=exact,
            elem_name=f"{elem_name}/{attr}",
        )


@assert_equal.register(AnnData)
def assert_adata_equal(
    a: AnnData, b: AnnData, exact: bool = False, elem_name: str | None = None
):
    """\
    Check whether two AnnData objects are equivalent,
    raising an AssertionError if they aren’t.

    Params
    ------
    a
    b
    exact
        Whether comparisons should be exact or not. This has a somewhat flexible
        meaning and should probably get refined in the future.
    """

    def fmt_name(x):
        if elem_name is None:
            return x
        else:
            return f"{elem_name}/{x}"

    # There may be issues comparing views, since np.allclose
    # can modify ArrayViews if they contain `nan`s
    assert_equal(a.obs_names, b.obs_names, exact, elem_name=fmt_name("obs_names"))
    assert_equal(a.var_names, b.var_names, exact, elem_name=fmt_name("var_names"))
    if not exact:
        # Reorder all elements if necessary
        idx = [slice(None), slice(None)]
        # Since it’s a pain to compare a list of pandas objects
        change_flag = False
        if not np.all(a.obs_names == b.obs_names):
            idx[0] = a.obs_names
            change_flag = True
        if not np.all(a.var_names == b.var_names):
            idx[1] = a.var_names
            change_flag = True
        if change_flag:
            b = b[tuple(idx)].copy()
    for attr in [
        "X",
        "obs",
        "var",
        "obsm",
        "varm",
        "layers",
        "uns",
        "obsp",
        "varp",
        "raw",
    ]:
        assert_equal(
            getattr(a, attr),
            getattr(b, attr),
            exact,
            elem_name=fmt_name(attr),
        )


def _half_chunk_size(a: tuple[int, ...]) -> tuple[int, ...]:
    def half_rounded_up(x):
        div, mod = divmod(x, 2)
        return div + (mod > 0)

    return tuple(half_rounded_up(x) for x in a)


@contextmanager
def pytest_8_raises(exc_cls, *, match: str | re.Pattern = None):
    """Error handling using pytest 8's support for __notes__.

    See: https://github.com/pytest-dev/pytest/pull/11227

    Remove once pytest 8 is out!
    """

    with pytest.raises(exc_cls) as exc_info:
        yield exc_info

    check_error_or_notes_match(exc_info, match)


def check_error_or_notes_match(e: pytest.ExceptionInfo, pattern: str | re.Pattern):
    """
    Checks whether the printed error message or the notes contains the given pattern.

    DOES NOT WORK IN IPYTHON - because of the way IPython handles exceptions
    """
    import traceback

    message = "".join(traceback.format_exception_only(e.type, e.value))
    assert re.search(
        pattern, message
    ), f"Could not find pattern: '{pattern}' in error:\n\n{message}\n"


@singledispatch
def shares_memory(x, y) -> bool:
    return np.shares_memory(x, y)


@shares_memory.register(sparse.spmatrix)
def shares_memory_sparse(x, y):
    return (
        np.shares_memory(x.data, y.data)
        and np.shares_memory(x.indices, y.indices)
        and np.shares_memory(x.indptr, y.indptr)
    )


BASE_MATRIX_PARAMS = [
    pytest.param(asarray, id="np_array"),
    pytest.param(sparse.csr_matrix, id="scipy_csr"),
    pytest.param(sparse.csc_matrix, id="scipy_csc"),
]
