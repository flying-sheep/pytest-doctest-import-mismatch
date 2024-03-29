from __future__ import annotations

import warnings
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce, singledispatch, wraps
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from scipy import sparse

from .access import ElementRef

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, KeysView, Sequence

    from anndata import AnnData


@contextmanager
def view_update(adata_view: AnnData, attr_name: str, keys: tuple[str, ...]):
    """Context manager for updating a view of an AnnData object.

    Contains logic for "actualizing" a view. Yields the object to be modified in-place.

    Parameters
    ----------
    adata_view
        A view of an AnnData
    attr_name
        Name of the attribute being updated
    keys
        Keys to the attribute being updated

    Yields
    ------

    `adata.attr[key1][key2][keyn]...`
    """
    new = adata_view.copy()
    attr = getattr(new, attr_name)
    container = reduce(lambda d, k: d[k], keys, attr)
    yield container
    adata_view._init_as_actual(new)


class _SetItemMixin:
    """\
    Class which (when values are being set) lets their parent AnnData view know,
    so it can make a copy of itself.
    This implements copy-on-modify semantics for views of AnnData objects.
    """

    _view_args: ElementRef | None

    def __setitem__(self, idx: Any, value: Any):
        if self._view_args is None:
            super().__setitem__(idx, value)
        else:
            with view_update(*self._view_args) as container:
                container[idx] = value


class _ViewMixin(_SetItemMixin):
    def __init__(
        self,
        *args,
        view_args: tuple[AnnData, str, tuple[str, ...]] = None,
        **kwargs,
    ):
        if view_args is not None:
            view_args = ElementRef(*view_args)
        self._view_args = view_args
        super().__init__(*args, **kwargs)

    # TODO: This makes `deepcopy(obj)` return `obj._view_args.parent._adata_ref`, fix it
    def __deepcopy__(self, memo):
        parent, attrname, keys = self._view_args
        return deepcopy(getattr(parent._adata_ref, attrname))


_UFuncMethod = Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"]


class ArrayView(_SetItemMixin, np.ndarray):
    def __new__(
        cls,
        input_array: Sequence[Any],
        view_args: tuple[AnnData, str, tuple[str, ...]] = None,
    ):
        arr = np.asanyarray(input_array).view(cls)

        if view_args is not None:
            view_args = ElementRef(*view_args)
        arr._view_args = view_args
        return arr

    def __array_finalize__(self, obj: np.ndarray | None):
        if obj is not None:
            self._view_args = getattr(obj, "_view_args", None)

    def __array_ufunc__(
        self: ArrayView,
        ufunc: Callable[..., Any],
        method: _UFuncMethod,
        *inputs,
        out: tuple[np.ndarray, ...] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Makes numpy ufuncs convert all instances of views to plain arrays.

        See https://numpy.org/devdocs/user/basics.subclassing.html#array-ufunc-for-ufuncs
        """

        def convert_all(arrs: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
            return (
                arr.view(np.ndarray) if isinstance(arr, ArrayView) else arr
                for arr in arrs
            )

        if out is None:
            outputs = (None,) * ufunc.nout
        else:
            out = outputs = tuple(convert_all(out))

        results = super().__array_ufunc__(
            ufunc, method, *convert_all(inputs), out=out, **kwargs
        )
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)
        results = tuple(
            (np.asarray(result) if output is None else output)
            for result, output in zip(results, outputs)
        )
        return results[0] if len(results) == 1 else results

    def keys(self) -> KeysView[str]:
        # it’s a structured array
        return self.dtype.names

    def copy(self, order: str = "C") -> np.ndarray:
        # we want a conventional array
        return np.array(self)

    def toarray(self) -> np.ndarray:
        return self.copy()


# Unlike array views, SparseCSRView and SparseCSCView
# do not propagate through subsetting
class SparseCSRView(_ViewMixin, sparse.csr_matrix):
    # https://github.com/scverse/anndata/issues/656
    def copy(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(self).copy()


class SparseCSCView(_ViewMixin, sparse.csc_matrix):
    # https://github.com/scverse/anndata/issues/656
    def copy(self) -> sparse.csc_matrix:
        return sparse.csc_matrix(self).copy()


class DictView(_ViewMixin, dict):
    pass


class DataFrameView(_ViewMixin, pd.DataFrame):
    _metadata = ["_view_args"]

    @wraps(pd.DataFrame.drop)
    def drop(self, *args, inplace: bool = False, **kw):
        if not inplace:
            return self.copy().drop(*args, **kw)
        with view_update(*self._view_args) as df:
            df.drop(*args, inplace=True, **kw)


@singledispatch
def as_view(obj, view_args):
    raise NotImplementedError(f"No view type has been registered for {type(obj)}")


@as_view.register(np.ndarray)
def as_view_array(array, view_args):
    return ArrayView(array, view_args=view_args)


@as_view.register(pd.DataFrame)
def as_view_df(df, view_args):
    return DataFrameView(df, view_args=view_args)


@as_view.register(sparse.csr_matrix)
def as_view_csr(mtx, view_args):
    return SparseCSRView(mtx, view_args=view_args)


@as_view.register(sparse.csc_matrix)
def as_view_csc(mtx, view_args):
    return SparseCSCView(mtx, view_args=view_args)


@as_view.register(dict)
def as_view_dict(d, view_args):
    return DictView(d, view_args=view_args)


def _resolve_idxs(old, new, adata):
    t = tuple(_resolve_idx(old[i], new[i], adata.shape[i]) for i in (0, 1))
    return t


@singledispatch
def _resolve_idx(old, new, l):
    return old[new]


@_resolve_idx.register(np.ndarray)
def _resolve_idx_ndarray(old, new, l):
    if is_bool_dtype(old):
        old = np.where(old)[0]
    return old[new]


@_resolve_idx.register(np.integer)
@_resolve_idx.register(int)
def _resolve_idx_scalar(old, new, l):
    return np.array([old])[new]


@_resolve_idx.register(slice)
def _resolve_idx_slice(old, new, l):
    if isinstance(new, slice):
        return _resolve_idx_slice_slice(old, new, l)
    else:
        return np.arange(*old.indices(l))[new]


def _resolve_idx_slice_slice(old, new, l):
    r = range(*old.indices(l))[new]
    # Convert back to slice
    start, stop, step = r.start, r.stop, r.step
    if len(r) == 0:
        stop = start
    elif stop < 0:
        stop = None
    return slice(start, stop, step)
