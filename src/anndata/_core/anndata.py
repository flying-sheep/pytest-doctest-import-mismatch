"""\
Main class and helper functions.
"""
from __future__ import annotations

import collections.abc as cabc
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import copy, deepcopy
from enum import Enum
from functools import singledispatch
from textwrap import dedent
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy import ma
from pandas.api.types import infer_dtype, is_string_dtype
from scipy import sparse
from scipy.sparse import issparse

from anndata._warnings import ImplicitModificationWarning

from .. import utils
from ..compat import _move_adj_mtx
from ..logging import anndata_logger as logger
from ..utils import convert_to_dict, dim_len, ensure_df_homogeneous
from .access import ElementRef
from .aligned_mapping import (
    AxisArrays,
    AxisArraysView,
    Layers,
    LayersView,
    PairwiseArrays,
    PairwiseArraysView,
)
from .index import Index, Index1D, _normalize_indices, _subset, get_vector
from .raw import Raw
from .views import ArrayView, DataFrameView, DictView, _resolve_idxs, as_view


class StorageType(Enum):
    Array = np.ndarray
    Masked = ma.MaskedArray
    Sparse = sparse.spmatrix

    @classmethod
    def classes(cls):
        return tuple(c.value for c in cls.__members__.values())


# for backwards compat
def _find_corresponding_multicol_key(key, keys_multicol):
    """Find the corresponding multicolumn key."""
    for mk in keys_multicol:
        if key.startswith(mk) and "of" in key:
            return mk
    return None


# for backwards compat
def _gen_keys_from_multicol_key(key_multicol, n_keys):
    """Generates single-column keys from multicolumn key."""
    keys = [f"{key_multicol}{i + 1:03}of{n_keys:03}" for i in range(n_keys)]
    return keys


def _check_2d_shape(X):
    """\
    Check shape of array or sparse matrix.

    Assure that X is always 2D: Unlike numpy we always deal with 2D arrays.
    """
    if X.dtype.names is None and len(X.shape) != 2:
        raise ValueError(
            f"X needs to be 2-dimensional, not {len(X.shape)}-dimensional."
        )


def _mk_df_error(
    source: Literal["X", "shape"],
    attr: Literal["obs", "var"],
    expected: int,
    actual: int,
):
    if source == "X":
        what = "row" if attr == "obs" else "column"
        msg = (
            f"Observations annot. `{attr}` must have as many rows as `X` has {what}s "
            f"({expected}), but has {actual} rows."
        )
    else:
        msg = (
            f"`shape` is inconsistent with `{attr}` "
            "({actual} {what}s instead of {expected})"
        )
    return ValueError(msg)


@singledispatch
def _gen_dataframe(
    anno: Mapping[str, Any],
    index_names: Iterable[str],
    *,
    source: Literal["X", "shape"],
    attr: Literal["obs", "var"],
    length: int | None = None,
) -> pd.DataFrame:
    if anno is None or len(anno) == 0:
        anno = {}

    def mk_index(l: int) -> pd.Index:
        return pd.RangeIndex(0, l, name=None).astype(str)

    for index_name in index_names:
        if index_name not in anno:
            continue
        df = pd.DataFrame(
            anno,
            index=anno[index_name],
            columns=[k for k in anno.keys() if k != index_name],
        )
        break
    else:
        df = pd.DataFrame(
            anno,
            index=None if length is None else mk_index(length),
            columns=None if len(anno) else [],
        )

    if length is None:
        df.index = mk_index(len(df))
    elif length != len(df):
        raise _mk_df_error(source, attr, length, len(df))
    return df


@_gen_dataframe.register(pd.DataFrame)
def _gen_dataframe_df(
    anno: pd.DataFrame,
    index_names: Iterable[str],
    *,
    source: Literal["X", "shape"],
    attr: Literal["obs", "var"],
    length: int | None = None,
):
    if length is not None and length != len(anno):
        raise _mk_df_error(source, attr, length, len(anno))
    anno = anno.copy(deep=False)
    if not is_string_dtype(anno.index):
        warnings.warn("Transforming to str index.", ImplicitModificationWarning)
        anno.index = anno.index.astype(str)
    if not len(anno.columns):
        anno.columns = anno.columns.astype(str)
    return anno


@_gen_dataframe.register(pd.Series)
@_gen_dataframe.register(pd.Index)
def _gen_dataframe_1d(
    anno: pd.Series | pd.Index,
    index_names: Iterable[str],
    *,
    source: Literal["X", "shape"],
    attr: Literal["obs", "var"],
    length: int | None = None,
):
    raise ValueError(f"Cannot convert {type(anno)} to {attr} DataFrame")


class AnnData:
    _BACKED_ATTRS = ["X", "raw.X"]

    # backwards compat
    _H5_ALIASES = dict(
        X={"X", "_X", "data", "_data"},
        obs={"obs", "_obs", "smp", "_smp"},
        var={"var", "_var"},
        uns={"uns"},
        obsm={"obsm", "_obsm", "smpm", "_smpm"},
        varm={"varm", "_varm"},
        layers={"layers", "_layers"},
    )

    _H5_ALIASES_NAMES = dict(
        obs={"obs_names", "smp_names", "row_names", "index"},
        var={"var_names", "col_names", "index"},
    )

    def __init__(
        self,
        X: np.ndarray | sparse.spmatrix | pd.DataFrame | None = None,
        obs: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        var: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        uns: Mapping[str, Any] | None = None,
        obsm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        varm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        layers: Mapping[str, np.ndarray | sparse.spmatrix] | None = None,
        raw: Mapping[str, Any] | None = None,
        dtype: np.dtype | type | str | None = None,
        shape: tuple[int, int] | None = None,
        asview: bool = False,
        *,
        obsp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        varp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        oidx: Index1D = None,
        vidx: Index1D = None,
    ):
        if asview:
            if not isinstance(X, AnnData):
                raise ValueError("`X` has to be an AnnData object.")
            self._init_as_view(X, oidx, vidx)
        else:
            self._init_as_actual(
                X=X,
                obs=obs,
                var=var,
                uns=uns,
                obsm=obsm,
                varm=varm,
                raw=raw,
                layers=layers,
                dtype=dtype,
                shape=shape,
                obsp=obsp,
                varp=varp,
            )

    def _init_as_view(self, adata_ref: AnnData, oidx: Index, vidx: Index):
        self._is_view = True
        if isinstance(oidx, (int, np.integer)):
            if not (-adata_ref.n_obs <= oidx < adata_ref.n_obs):
                raise IndexError(f"Observation index `{oidx}` is out of range.")
            oidx += adata_ref.n_obs * (oidx < 0)
            oidx = slice(oidx, oidx + 1, 1)
        if isinstance(vidx, (int, np.integer)):
            if not (-adata_ref.n_vars <= vidx < adata_ref.n_vars):
                raise IndexError(f"Variable index `{vidx}` is out of range.")
            vidx += adata_ref.n_vars * (vidx < 0)
            vidx = slice(vidx, vidx + 1, 1)
        if adata_ref.is_view:
            prev_oidx, prev_vidx = adata_ref._oidx, adata_ref._vidx
            adata_ref = adata_ref._adata_ref
            oidx, vidx = _resolve_idxs((prev_oidx, prev_vidx), (oidx, vidx), adata_ref)
        # self._adata_ref is never a view
        self._adata_ref = adata_ref
        self._oidx = oidx
        self._vidx = vidx
        # views on attributes of adata_ref
        obs_sub = adata_ref.obs.iloc[oidx]
        var_sub = adata_ref.var.iloc[vidx]
        self._obsm = adata_ref.obsm._view(self, (oidx,))
        self._varm = adata_ref.varm._view(self, (vidx,))
        self._layers = adata_ref.layers._view(self, (oidx, vidx))
        self._obsp = adata_ref.obsp._view(self, oidx)
        self._varp = adata_ref.varp._view(self, vidx)
        # fix categories
        uns = copy(adata_ref._uns)
        self._remove_unused_categories(adata_ref.obs, obs_sub, uns)
        self._remove_unused_categories(adata_ref.var, var_sub, uns)
        # set attributes
        self._obs = DataFrameView(obs_sub, view_args=(self, "obs"))
        self._var = DataFrameView(var_sub, view_args=(self, "var"))
        self._uns = uns

        # set raw, easy, as it’s immutable anyways...
        if adata_ref._raw is not None:
            # slicing along variables axis is ignored
            self._raw = adata_ref.raw[oidx]
            self._raw._adata = self
        else:
            self._raw = None

    def _init_as_actual(
        self,
        X=None,
        obs=None,
        var=None,
        uns=None,
        obsm=None,
        varm=None,
        varp=None,
        obsp=None,
        raw=None,
        layers=None,
        dtype=None,
        shape=None,
    ):
        # view attributes
        self._is_view = False
        self._adata_ref = None
        self._oidx = None
        self._vidx = None

        # ----------------------------------------------------------------------
        # various ways of initializing the data
        # ----------------------------------------------------------------------

        # If X is a data frame, we store its indices for verification
        x_indices = []

        # init from AnnData
        if isinstance(X, AnnData):
            if any((obs, var, uns, obsm, varm, obsp, varp)):
                raise ValueError(
                    "If `X` is a dict no further arguments must be provided."
                )
            X, obs, var, uns, obsm, varm, obsp, varp, layers, raw = (
                X._X,
                X.obs,
                X.var,
                X.uns,
                X.obsm,
                X.varm,
                X.obsp,
                X.varp,
                X.layers,
                X.raw,
            )

        # init from DataFrame
        elif isinstance(X, pd.DataFrame):
            # to verify index matching, we wait until obs and var are DataFrames
            if obs is None:
                obs = pd.DataFrame(index=X.index)
            elif not isinstance(X.index, pd.RangeIndex):
                x_indices.append(("obs", "index", X.index.astype(str)))
            if var is None:
                var = pd.DataFrame(index=X.columns)
            elif not isinstance(X.columns, pd.RangeIndex):
                x_indices.append(("var", "columns", X.columns.astype(str)))
            X = ensure_df_homogeneous(X, "X")

        # ----------------------------------------------------------------------
        # actually process the data
        # ----------------------------------------------------------------------

        # check data type of X
        if X is not None:
            for s_type in StorageType:
                if isinstance(X, s_type.value):
                    break
            else:
                class_names = ", ".join(c.__name__ for c in StorageType.classes())
                raise ValueError(
                    f"`X` needs to be of one of {class_names}, not {type(X)}."
                )
            if shape is not None:
                raise ValueError("`shape` needs to be `None` if `X` is not `None`.")
            _check_2d_shape(X)
            # data matrix and shape
            self._X = X
            n_obs, n_vars = X.shape
            source = "X"
        else:
            self._X = None
            n_obs, n_vars = (None, None) if shape is None else shape
            source = "shape"

        # annotations
        self._obs = _gen_dataframe(
            obs, ["obs_names", "row_names"], source=source, attr="obs", length=n_obs
        )
        self._var = _gen_dataframe(
            var, ["var_names", "col_names"], source=source, attr="var", length=n_vars
        )

        # now we can verify if indices match!
        for attr_name, x_name, idx in x_indices:
            attr = getattr(self, attr_name)
            if isinstance(attr.index, pd.RangeIndex):
                attr.index = idx
            elif not idx.equals(attr.index):
                raise ValueError(f"Index of {attr_name} must match {x_name} of X.")

        # unstructured annotations
        self.uns = uns or OrderedDict()

        # TODO: Think about consequences of making obsm a group in hdf
        self._obsm = AxisArrays(self, 0, vals=convert_to_dict(obsm))
        self._varm = AxisArrays(self, 1, vals=convert_to_dict(varm))

        self._obsp = PairwiseArrays(self, 0, vals=convert_to_dict(obsp))
        self._varp = PairwiseArrays(self, 1, vals=convert_to_dict(varp))

        # Backwards compat for connectivities matrices in uns["neighbors"]
        _move_adj_mtx({"uns": self._uns, "obsp": self._obsp})

        self._check_dimensions()
        self._check_uniqueness()

        if not raw:
            self._raw = None
        elif isinstance(raw, cabc.Mapping):
            self._raw = Raw(self, **raw)
        else:  # is a Raw from another AnnData
            self._raw = Raw(self, raw._X, raw.var, raw.varm)

        # clean up old formats
        self._clean_up_old_format(uns)

        # layers
        self._layers = Layers(self, layers)

    def _gen_repr(self, n_obs, n_vars) -> str:
        descr = f"AnnData object with n_obs × n_vars = {n_obs} × {n_vars}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars)
        else:
            return self._gen_repr(self.n_obs, self.n_vars)

    def __eq__(self, other):
        """Equality testing"""
        raise NotImplementedError(
            "Equality comparisons are not supported for AnnData objects, "
            "instead compare the desired attributes."
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of data matrix (:attr:`n_obs`, :attr:`n_vars`)."""
        return self.n_obs, self.n_vars

    @property
    def X(self) -> np.ndarray | sparse.spmatrix | ArrayView | None:
        """Data matrix of shape :attr:`n_obs` × :attr:`n_vars`."""
        if self.is_view and self._adata_ref.X is None:
            X = None
        elif self.is_view:
            X = as_view(
                _subset(self._adata_ref.X, (self._oidx, self._vidx)),
                ElementRef(self, "X"),
            )
        else:
            X = self._X
        return X
        # if self.n_obs == 1 and self.n_vars == 1:
        #     return X[0, 0]
        # elif self.n_obs == 1 or self.n_vars == 1:
        #     if issparse(X): X = X.toarray()
        #     return X.flatten()
        # else:
        #     return X

    @X.setter
    def X(self, value: np.ndarray | sparse.spmatrix | None):
        if value is None:
            if self.is_view:
                self._init_as_actual(self.copy())
            self._X = None
            return
        if not isinstance(value, StorageType.classes()) and not np.isscalar(value):
            if hasattr(value, "to_numpy") and hasattr(value, "dtypes"):
                value = ensure_df_homogeneous(value, "X")
            else:  # TODO: asarray? asanyarray?
                value = np.array(value)

        # If indices are both arrays, we need to modify them
        # so we don’t set values like coordinates
        # This can occur if there are successive views
        if (
            self.is_view
            and isinstance(self._oidx, np.ndarray)
            and isinstance(self._vidx, np.ndarray)
        ):
            oidx, vidx = np.ix_(self._oidx, self._vidx)
        else:
            oidx, vidx = self._oidx, self._vidx
        if (
            np.isscalar(value)
            or (hasattr(value, "shape") and (self.shape == value.shape))
            or (self.n_vars == 1 and self.n_obs == len(value))
            or (self.n_obs == 1 and self.n_vars == len(value))
        ):
            if not np.isscalar(value) and self.shape != value.shape:
                # For assigning vector of values to 2d array or matrix
                # Not necessary for row of 2d array
                value = value.reshape(self.shape)
            if self.is_view:
                if sparse.issparse(self._adata_ref._X) and isinstance(
                    value, np.ndarray
                ):
                    value = sparse.coo_matrix(value)
                self._adata_ref._X[oidx, vidx] = value
            else:
                self._X = value
        else:
            raise ValueError(
                f"Data matrix has wrong shape {value.shape}, "
                f"need to be {self.shape}."
            )

    @X.deleter
    def X(self):
        self.X = None

    @property
    def layers(self) -> Layers | LayersView:
        """\
        Dictionary-like object with values of the same dimensions as :attr:`X`.

        Layers in AnnData are inspired by loompy’s :ref:`loomlayers`.

        Return the layer named `"unspliced"`::

            adata.layers["unspliced"]

        Create or replace the `"spliced"` layer::

            adata.layers["spliced"] = ...

        Assign the 10th column of layer `"spliced"` to the variable a::

            a = adata.layers["spliced"][:, 10]

        Delete the `"spliced"` layer::

            del adata.layers["spliced"]

        Return layers’ names::

            adata.layers.keys()
        """
        return self._layers

    @layers.setter
    def layers(self, value):
        layers = Layers(self, vals=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._layers = layers

    @layers.deleter
    def layers(self):
        self.layers = dict()

    @property
    def raw(self) -> Raw:
        """\
        Store raw version of :attr:`X` and :attr:`var` as `.raw.X` and `.raw.var`.

        The :attr:`raw` attribute is initialized with the current content
        of an object by setting::

            adata.raw = adata

        Its content can be deleted::

            adata.raw = None
            # or
            del adata.raw

        Upon slicing an AnnData object along the obs (row) axis, :attr:`raw`
        is also sliced. Slicing an AnnData object along the vars (columns) axis
        leaves :attr:`raw` unaffected. Note that you can call::

             adata.raw[:, 'orig_variable_name'].X

        to retrieve the data associated with a variable that might have been
        filtered out or "compressed away" in :attr:`X`.
        """
        return self._raw

    @raw.setter
    def raw(self, value: AnnData):
        if value is None:
            del self.raw
        elif not isinstance(value, AnnData):
            raise ValueError("Can only init raw attribute with an AnnData object.")
        else:
            if self.is_view:
                self._init_as_actual(self.copy())
            self._raw = Raw(self, X=value.X, var=value.var, varm=value.varm)

    @raw.deleter
    def raw(self):
        if self.is_view:
            self._init_as_actual(self.copy())
        self._raw = None

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return len(self.obs_names)

    @property
    def n_vars(self) -> int:
        """Number of variables/features."""
        return len(self.var_names)

    def _set_dim_df(self, value: pd.DataFrame, attr: str):
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"Can only assign pd.DataFrame to {attr}.")
        value_idx = self._prep_dim_index(value.index, attr)
        if self.is_view:
            self._init_as_actual(self.copy())
        setattr(self, f"_{attr}", value)
        self._set_dim_index(value_idx, attr)
        if not len(value.columns):
            value.columns = value.columns.astype(str)

    def _prep_dim_index(self, value, attr: str) -> pd.Index:
        """Prepares index to be uses as obs_names or var_names for AnnData object.AssertionError

        If a pd.Index is passed, this will use a reference, otherwise a new index object is created.
        """
        if self.shape[attr == "var"] != len(value):
            raise ValueError(
                f"Length of passed value for {attr}_names is {len(value)}, but this AnnData has shape: {self.shape}"
            )
        if isinstance(value, pd.Index) and not isinstance(
            value.name, (str, type(None))
        ):
            raise ValueError(
                f"AnnData expects .{attr}.index.name to be a string or None, "
                f"but you passed a name of type {type(value.name).__name__!r}"
            )
        else:
            value = pd.Index(value)
            if not isinstance(value.name, (str, type(None))):
                value.name = None
        if (
            len(value) > 0
            and not isinstance(value, pd.RangeIndex)
            and infer_dtype(value) not in ("string", "bytes")
        ):
            sample = list(value[: min(len(value), 5)])
            msg = dedent(
                f"""
                AnnData expects .{attr}.index to contain strings, but got values like:
                    {sample}

                    Inferred to be: {infer_dtype(value)}
                """
            )
            warnings.warn(msg, stacklevel=2)
        return value

    def _set_dim_index(self, value: pd.Index, attr: str):
        # Assumes _prep_dim_index has been run
        if self.is_view:
            self._init_as_actual(self.copy())
        getattr(self, attr).index = value
        for v in getattr(self, f"{attr}m").values():
            if isinstance(v, pd.DataFrame):
                v.index = value

    @property
    def obs(self) -> pd.DataFrame:
        """One-dimensional annotation of observations (`pd.DataFrame`)."""
        return self._obs

    @obs.setter
    def obs(self, value: pd.DataFrame):
        self._set_dim_df(value, "obs")

    @obs.deleter
    def obs(self):
        self.obs = pd.DataFrame({}, index=self.obs_names)

    @property
    def obs_names(self) -> pd.Index:
        """Names of observations (alias for `.obs.index`)."""
        return self.obs.index

    @obs_names.setter
    def obs_names(self, names: Sequence[str]):
        names = self._prep_dim_index(names, "obs")
        self._set_dim_index(names, "obs")

    @property
    def var(self) -> pd.DataFrame:
        """One-dimensional annotation of variables/ features (`pd.DataFrame`)."""
        return self._var

    @var.setter
    def var(self, value: pd.DataFrame):
        self._set_dim_df(value, "var")

    @var.deleter
    def var(self):
        self.var = pd.DataFrame({}, index=self.var_names)

    @property
    def var_names(self) -> pd.Index:
        """Names of variables (alias for `.var.index`)."""
        return self.var.index

    @var_names.setter
    def var_names(self, names: Sequence[str]):
        names = self._prep_dim_index(names, "var")
        self._set_dim_index(names, "var")

    @property
    def uns(self) -> MutableMapping:
        """Unstructured annotation (ordered dictionary)."""
        uns = self._uns
        if self.is_view:
            uns = DictView(uns, view_args=(self, "_uns"))
        return uns

    @uns.setter
    def uns(self, value: MutableMapping):
        if not isinstance(value, MutableMapping):
            raise ValueError(
                "Only mutable mapping types (e.g. dict) are allowed for `.uns`."
            )
        if isinstance(value, DictView):
            value = value.copy()
        if self.is_view:
            self._init_as_actual(self.copy())
        self._uns = value

    @uns.deleter
    def uns(self):
        self.uns = OrderedDict()

    @property
    def obsm(self) -> AxisArrays | AxisArraysView:
        """\
        Multi-dimensional annotation of observations
        (mutable structured :class:`~numpy.ndarray`).

        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        of length `n_obs`.
        Is sliced with `data` and `obs` but behaves otherwise like a :term:`mapping`.
        """
        return self._obsm

    @obsm.setter
    def obsm(self, value):
        obsm = AxisArrays(self, 0, vals=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obsm = obsm

    @obsm.deleter
    def obsm(self):
        self.obsm = dict()

    @property
    def varm(self) -> AxisArrays | AxisArraysView:
        """\
        Multi-dimensional annotation of variables/features
        (mutable structured :class:`~numpy.ndarray`).

        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        of length `n_vars`.
        Is sliced with `data` and `var` but behaves otherwise like a :term:`mapping`.
        """
        return self._varm

    @varm.setter
    def varm(self, value):
        varm = AxisArrays(self, 1, vals=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._varm = varm

    @varm.deleter
    def varm(self):
        self.varm = dict()

    @property
    def obsp(self) -> PairwiseArrays | PairwiseArraysView:
        """\
        Pairwise annotation of observations,
        a mutable mapping with array-like values.

        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        whose first two dimensions are of length `n_obs`.
        Is sliced with `data` and `obs` but behaves otherwise like a :term:`mapping`.
        """
        return self._obsp

    @obsp.setter
    def obsp(self, value):
        obsp = PairwiseArrays(self, 0, vals=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._obsp = obsp

    @obsp.deleter
    def obsp(self):
        self.obsp = dict()

    @property
    def varp(self) -> PairwiseArrays | PairwiseArraysView:
        """\
        Pairwise annotation of variables/features,
        a mutable mapping with array-like values.

        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        whose first two dimensions are of length `n_var`.
        Is sliced with `data` and `var` but behaves otherwise like a :term:`mapping`.
        """
        return self._varp

    @varp.setter
    def varp(self, value):
        varp = PairwiseArrays(self, 1, vals=convert_to_dict(value))
        if self.is_view:
            self._init_as_actual(self.copy())
        self._varp = varp

    @varp.deleter
    def varp(self):
        self.varp = dict()

    def obs_keys(self) -> list[str]:
        """List keys of observation annotation :attr:`obs`."""
        return self._obs.keys().tolist()

    def var_keys(self) -> list[str]:
        """List keys of variable annotation :attr:`var`."""
        return self._var.keys().tolist()

    def obsm_keys(self) -> list[str]:
        """List keys of observation annotation :attr:`obsm`."""
        return list(self._obsm.keys())

    def varm_keys(self) -> list[str]:
        """List keys of variable annotation :attr:`varm`."""
        return list(self._varm.keys())

    def uns_keys(self) -> list[str]:
        """List keys of unstructured annotation."""
        return sorted(list(self._uns.keys()))

    @property
    def is_view(self) -> bool:
        """`True` if object is view of another AnnData object, `False` otherwise."""
        return self._is_view

    def _normalize_indices(self, index: Index | None) -> tuple[slice, slice]:
        return _normalize_indices(index, self.obs_names, self.var_names)

    # TODO: this is not quite complete...
    def __delitem__(self, index: Index):
        obs, var = self._normalize_indices(index)
        del self._X[obs, var]
        if var == slice(None):
            del self._obs.iloc[obs, :]
        if obs == slice(None):
            del self._var.iloc[var, :]

    def __getitem__(self, index: Index) -> AnnData:
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return AnnData(self, oidx=oidx, vidx=vidx, asview=True)

    def _remove_unused_categories(
        self, df_full: pd.DataFrame, df_sub: pd.DataFrame, uns: dict[str, Any]
    ):
        for k in df_full:
            if not isinstance(df_full[k].dtype, pd.CategoricalDtype):
                continue
            all_categories = df_full[k].cat.categories
            with pd.option_context("mode.chained_assignment", None):
                df_sub[k] = df_sub[k].cat.remove_unused_categories()
            # also correct the colors...
            color_key = f"{k}_colors"
            if color_key not in uns:
                continue
            color_vec = uns[color_key]
            if np.array(color_vec).ndim == 0:
                # Make 0D arrays into 1D ones
                uns[color_key] = np.array(color_vec)[(None,)]
            elif len(color_vec) != len(all_categories):
                # Reset colors
                del uns[color_key]
            else:
                idx = np.where(np.in1d(all_categories, df_sub[k].cat.categories))[0]
                uns[color_key] = np.array(color_vec)[(idx,)]

    def rename_categories(self, key: str, categories: Sequence[Any]):
        """\
        Rename categories of annotation `key` in :attr:`obs`, :attr:`var`,
        and :attr:`uns`.

        Only supports passing a list/array-like `categories` argument.

        Besides calling `self.obs[key].cat.categories = categories` –
        similar for :attr:`var` - this also renames categories in unstructured
        annotation that uses the categorical annotation `key`.

        Parameters
        ----------
        key
             Key for observations or variables annotation.
        categories
             New categories, the same number as the old categories.
        """
        if isinstance(categories, Mapping):
            raise ValueError("Only list-like `categories` is supported.")
        if key in self.obs:
            old_categories = self.obs[key].cat.categories.tolist()
            self.obs[key] = self.obs[key].cat.rename_categories(categories)
        elif key in self.var:
            old_categories = self.var[key].cat.categories.tolist()
            self.var[key] = self.var[key].cat.rename_categories(categories)
        else:
            raise ValueError(f"{key} is neither in `.obs` nor in `.var`.")
        # this is not a good solution
        # but depends on the scanpy conventions for storing the categorical key
        # as `groupby` in the `params` slot
        for k1, v1 in self.uns.items():
            if not (
                isinstance(v1, Mapping)
                and "params" in v1
                and "groupby" in v1["params"]
                and v1["params"]["groupby"] == key
            ):
                continue
            for k2, v2 in v1.items():
                # picks out the recarrays that are named according to the old
                # categories
                if isinstance(v2, np.ndarray) and v2.dtype.names is not None:
                    if list(v2.dtype.names) == old_categories:
                        self.uns[k1][k2].dtype.names = categories
                    else:
                        logger.warning(
                            f"Omitting {k1}/{k2} as old categories do not match."
                        )

    def strings_to_categoricals(self, df: pd.DataFrame | None = None):
        """\
        Transform string annotations to categoricals.

        Only affects string annotations that lead to less categories than the
        total number of observations.

        Params
        ------
        df
            If `df` is `None`, modifies both :attr:`obs` and :attr:`var`,
            otherwise modifies `df` inplace.

        Notes
        -----
        Turns the view of an :class:`~anndata.AnnData` into an actual
        :class:`~anndata.AnnData`.
        """
        dont_modify = False  # only necessary for backed views
        if df is None:
            dfs = [self.obs, self.var]
        else:
            dfs = [df]
        for df in dfs:
            string_cols = [
                key for key in df.columns if infer_dtype(df[key]) == "string"
            ]
            for key in string_cols:
                c = pd.Categorical(df[key])
                # TODO: We should only check if non-null values are unique, but
                # this would break cases where string columns with nulls could
                # be written as categorical, but not as string.
                # Possible solution: https://github.com/scverse/anndata/issues/504
                if len(c.categories) >= len(c):
                    continue
                # Ideally this could be done inplace
                sorted_categories = sorted(c.categories)
                if not np.array_equal(c.categories, sorted_categories):
                    c = c.reorder_categories(sorted_categories)
                if dont_modify:
                    raise RuntimeError(
                        "Please call `.strings_to_categoricals()` on full "
                        "AnnData, not on this view. You might encounter this"
                        "error message while copying or writing to disk."
                    )
                df[key] = c
                logger.info(f"... storing {key!r} as categorical")

    _sanitize = strings_to_categoricals  # backwards compat

    def _inplace_subset_var(self, index: Index1D):
        """\
        Inplace subsetting along variables dimension.

        Same as `adata = adata[:, index]`, but inplace.
        """
        adata_subset = self[:, index].copy()

        self._init_as_actual(adata_subset)

    def _inplace_subset_obs(self, index: Index1D):
        """\
        Inplace subsetting along variables dimension.

        Same as `adata = adata[index, :]`, but inplace.
        """
        adata_subset = self[index].copy()

        self._init_as_actual(adata_subset)

    # TODO: Update, possibly remove
    def __setitem__(
        self, index: Index, val: int | float | np.ndarray | sparse.spmatrix
    ):
        if self.is_view:
            raise ValueError("Object is view and cannot be accessed with `[]`.")
        obs, var = self._normalize_indices(index)
        self._X[obs, var] = val

    def __len__(self) -> int:
        return self.shape[0]

    def transpose(self) -> AnnData:
        """\
        Transpose whole object.

        Data matrix is transposed, observations and variables are interchanged.
        Ignores `.raw`.
        """
        from anndata.compat import _safe_transpose

        X = self.X
        if self.is_view:
            raise ValueError(
                "You’re trying to transpose a view of an `AnnData`, "
                "which is currently not implemented. Call `.copy()` before transposing."
            )

        return AnnData(
            X=_safe_transpose(X) if X is not None else None,
            layers={k: _safe_transpose(v) for k, v in self.layers.items()},
            obs=self.var,
            var=self.obs,
            uns=self._uns,
            obsm=self._varm,
            varm=self._obsm,
            obsp=self._varp,
            varp=self._obsp,
        )

    T = property(transpose)

    def to_df(self, layer=None) -> pd.DataFrame:
        """\
        Generate shallow :class:`~pandas.DataFrame`.

        The data matrix :attr:`X` is returned as
        :class:`~pandas.DataFrame`, where :attr:`obs_names` initializes the
        index, and :attr:`var_names` the columns.

        * No annotations are maintained in the returned object.
        * The data matrix is densified in case it is sparse.

        Params
        ------
        layer : str
            Key for `.layers`.
        """
        if layer is not None:
            X = self.layers[layer]
        elif not self._has_X():
            raise ValueError("X is None, cannot convert to dataframe.")
        else:
            X = self.X
        if issparse(X):
            X = X.toarray()
        return pd.DataFrame(X, index=self.obs_names, columns=self.var_names)

    def _get_X(self, use_raw=False, layer=None):
        """\
        Convenience method for getting expression values
        with common arguments and error handling.
        """
        is_layer = layer is not None
        if use_raw and is_layer:
            raise ValueError(
                "Cannot use expression from both layer and raw. You provided:"
                f"`use_raw={use_raw}` and `layer={layer}`"
            )
        if is_layer:
            return self.layers[layer]
        elif use_raw:
            if self.raw is None:
                raise ValueError("This AnnData doesn’t have a value in `.raw`.")
            return self.raw.X
        else:
            return self.X

    def obs_vector(self, k: str, *, layer: str | None = None) -> np.ndarray:
        """\
        Convenience function for returning a 1 dimensional ndarray of values
        from :attr:`X`, :attr:`layers`\\ `[k]`, or :attr:`obs`.

        Made for convenience, not performance.
        Intentionally permissive about arguments, for easy iterative use.

        Params
        ------
        k
            Key to use. Should be in :attr:`var_names` or :attr:`obs`\\ `.columns`.
        layer
            What layer values should be returned from. If `None`, :attr:`X` is used.

        Returns
        -------
        A one dimensional ndarray, with values for each obs in the same order
        as :attr:`obs_names`.
        """
        if layer == "X":
            if "X" in self.layers:
                pass
            else:
                warnings.warn(
                    "In a future version of AnnData, access to `.X` by passing"
                    " `layer='X'` will be removed. Instead pass `layer=None`.",
                    FutureWarning,
                )
                layer = None
        return get_vector(self, k, "obs", "var", layer=layer)

    def var_vector(self, k, *, layer: str | None = None) -> np.ndarray:
        """\
        Convenience function for returning a 1 dimensional ndarray of values
        from :attr:`X`, :attr:`layers`\\ `[k]`, or :attr:`obs`.

        Made for convenience, not performance. Intentionally permissive about
        arguments, for easy iterative use.

        Params
        ------
        k
            Key to use. Should be in :attr:`obs_names` or :attr:`var`\\ `.columns`.
        layer
            What layer values should be returned from. If `None`, :attr:`X` is used.

        Returns
        -------
        A one dimensional ndarray, with values for each var in the same order
        as :attr:`var_names`.
        """
        if layer == "X":
            if "X" in self.layers:
                pass
            else:
                warnings.warn(
                    "In a future version of AnnData, access to `.X` by passing "
                    "`layer='X'` will be removed. Instead pass `layer=None`.",
                    FutureWarning,
                )
                layer = None
        return get_vector(self, k, "var", "obs", layer=layer)

    def _mutated_copy(self, **kwargs):
        """Creating AnnData with attributes optionally specified via kwargs."""
        new = {}

        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "layers"]:
            if key in kwargs:
                new[key] = kwargs[key]
            else:
                new[key] = getattr(self, key).copy()
        if "X" in kwargs:
            new["X"] = kwargs["X"]
        elif self._has_X():
            new["X"] = self.X.copy()
        if "uns" in kwargs:
            new["uns"] = kwargs["uns"]
        else:
            new["uns"] = deepcopy(self._uns)
        if "raw" in kwargs:
            new["raw"] = kwargs["raw"]
        elif self.raw is not None:
            new["raw"] = self.raw.copy()
        return AnnData(**new)

    def copy(self) -> AnnData:
        """Full copy, optionally on disk."""
        if self.is_view and self._has_X():
            return self._mutated_copy(
                X=_subset(self._adata_ref.X, (self._oidx, self._vidx)).copy()
            )
        else:
            return self._mutated_copy()

    def var_names_make_unique(self, join: str = "-"):
        # Important to go through the setter so obsm dataframes are updated too
        self.var_names = utils.make_index_unique(self.var.index, join)

    var_names_make_unique.__doc__ = utils.make_index_unique.__doc__

    def obs_names_make_unique(self, join: str = "-"):
        # Important to go through the setter so obsm dataframes are updated too
        self.obs_names = utils.make_index_unique(self.obs.index, join)

    obs_names_make_unique.__doc__ = utils.make_index_unique.__doc__

    def _check_uniqueness(self):
        if not self.obs.index.is_unique:
            utils.warn_names_duplicates("obs")
        if not self.var.index.is_unique:
            utils.warn_names_duplicates("var")

    def __contains__(self, key: Any):
        raise AttributeError(
            "AnnData has no attribute __contains__, don’t check `in adata`."
        )

    def _check_dimensions(self, key=None):
        if key is None:
            key = {"obsm", "varm"}
        else:
            key = {key}
        if "obsm" in key:
            obsm = self._obsm
            if (
                not all([dim_len(o, 0) == self.n_obs for o in obsm.values()])
                and len(obsm.dim_names) != self.n_obs
            ):
                raise ValueError(
                    "Observations annot. `obsm` must have number of rows of `X`"
                    f" ({self.n_obs}), but has {len(obsm)} rows."
                )
        if "varm" in key:
            varm = self._varm
            if (
                not all([dim_len(v, 0) == self.n_vars for v in varm.values()])
                and len(varm.dim_names) != self.n_vars
            ):
                raise ValueError(
                    "Variables annot. `varm` must have number of columns of `X`"
                    f" ({self.n_vars}), but has {len(varm)} rows."
                )

    def chunked_X(self, chunk_size: int | None = None):
        """\
        Return an iterator over the rows of the data matrix :attr:`X`.

        Parameters
        ----------
        chunk_size
            Row size of a single chunk.
        """
        if chunk_size is None:
            # Should be some adaptive code
            chunk_size = 6000
        start = 0
        n = self.n_obs
        for _ in range(int(n // chunk_size)):
            end = start + chunk_size
            yield (self.X[start:end], start, end)
            start = end
        if start < n:
            yield (self.X[start:n], start, n)

    def chunk_X(
        self,
        select: int | Sequence[int] | np.ndarray = 1000,
        replace: bool = True,
    ):
        """\
        Return a chunk of the data matrix :attr:`X` with random or specified indices.

        Parameters
        ----------
        select
            Depending on the type:

            :class:`int`
                A random chunk with `select` rows will be returned.
            :term:`sequence` (e.g. a list, tuple or numpy array) of :class:`int`
                A chunk with these indices will be returned.

        replace
            If `select` is an integer then `True` means random sampling of
            indices with replacement, `False` without replacement.
        """
        if isinstance(select, int):
            select = select if select < self.n_obs else self.n_obs
            choice = np.random.choice(self.n_obs, select, replace)
        elif isinstance(select, (np.ndarray, cabc.Sequence)):
            choice = np.asarray(select)
        else:
            raise ValueError("select should be int or array")

        reverse = None
        selection = self.X[choice]

        selection = selection.toarray() if issparse(selection) else selection
        return selection if reverse is None else selection[reverse]

    def _has_X(self) -> bool:
        """
        Check if X is None.

        This is more efficient than trying `adata.X is None` for views, since creating
        views (at least anndata's kind) can be expensive.
        """
        if not self.is_view:
            return self.X is not None
        else:
            return self._adata_ref.X is not None

    # --------------------------------------------------------------------------
    # all of the following is for backwards compat
    # --------------------------------------------------------------------------

    def _clean_up_old_format(self, uns):
        # multicolumn keys
        # all of the rest is only for backwards compat
        for bases in [["obs", "smp"], ["var"]]:
            axis = bases[0]
            for k in [f"{p}{base}_keys_multicol" for p in ["", "_"] for base in bases]:
                if uns and k in uns:
                    keys = list(uns[k])
                    del uns[k]
                    break
            else:
                keys = []
            # now, for compat, fill the old multicolumn entries into obsm and varm
            # and remove them from obs and var
            m_attr = getattr(self, f"_{axis}m")
            for key in keys:
                m_attr[key] = self._get_and_delete_multicol_field(axis, key)

    def _get_and_delete_multicol_field(self, a, key_multicol):
        keys = []
        for k in getattr(self, a).columns:
            if k.startswith(key_multicol):
                keys.append(k)
        values = getattr(self, a)[keys].values
        getattr(self, a).drop(keys, axis=1, inplace=True)
        return values
