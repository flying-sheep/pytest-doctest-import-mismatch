from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy import sparse

import anndata as ad
from anndata.tests.helpers import (
    GEN_ADATA_ARGS,
    assert_equal,
    gen_adata,
    subset_func,
)
from anndata.utils import asarray

subset_func2 = subset_func
# -------------------------------------------------------------------------------
# Some test data
# -------------------------------------------------------------------------------


@pytest.fixture
def adata():
    X_list = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]  # data matrix of shape n_obs x n_vars
    X = np.array(X_list)
    obs_dict = dict(  # annotation of observations / rows
        row_names=["name1", "name2", "name3"],  # row annotation
        oanno1=["cat1", "cat2", "cat2"],  # categorical annotation
        oanno2=["o1", "o2", "o3"],  # string annotation
        oanno3=[2.1, 2.2, 2.3],  # float annotation
    )
    var_dict = dict(vanno1=[3.1, 3.2, 3.3])  # annotation of variables / columns
    uns_dict = dict(  # unstructured annotation
        oanno1_colors=["#000000", "#FFFFFF"], uns2=["some annotation"]
    )
    return ad.AnnData(
        X,
        obs=obs_dict,
        var=var_dict,
        uns=uns_dict,
        obsm=dict(o1=np.zeros((X.shape[0], 10))),
        varm=dict(v1=np.ones((X.shape[1], 20))),
        layers=dict(float=X.astype(float), sparse=sparse.csr_matrix(X)),
    )


@pytest.fixture(
    params=[sparse.csr_matrix, sparse.csc_matrix, np.array],
    ids=["scipy-csr", "scipy-csc", "np-array"],
)
def mtx_format(request):
    return request.param


@pytest.fixture(params=[sparse.csr_matrix, sparse.csc_matrix])
def sparse_format(request):
    return request.param


@pytest.fixture(params=["r+", "r", False])
def backed_mode(request):
    return request.param


@pytest.fixture(params=(("X",), ()))
def as_dense(request):
    return request.param


# -------------------------------------------------------------------------------
# The test functions
# -------------------------------------------------------------------------------


# TODO: Also test updating the backing file inplace
def test_backed_raw(tmp_path):
    backed_pth = tmp_path / "backed.h5ad"
    final_pth = tmp_path / "final.h5ad"
    mem_adata = gen_adata((10, 10), **GEN_ADATA_ARGS)
    mem_adata.raw = mem_adata
    mem_adata.write(backed_pth)

    backed_adata = ad.read_h5ad(backed_pth, backed="r")
    assert_equal(backed_adata, mem_adata)
    backed_adata.write_h5ad(final_pth)

    final_adata = ad.read_h5ad(final_pth)
    assert_equal(final_adata, mem_adata)


@pytest.mark.parametrize(
    "array_type",
    [
        pytest.param(asarray, id="dense_array"),
        pytest.param(sparse.csr_matrix, id="csr_matrix"),
    ],
)
def test_to_memory_full(tmp_path, array_type):
    backed_pth = tmp_path / "backed.h5ad"
    mem_adata = gen_adata((15, 10), X_type=array_type, **GEN_ADATA_ARGS)
    mem_adata.raw = gen_adata((15, 12), X_type=array_type, **GEN_ADATA_ARGS)
    mem_adata.write_h5ad(backed_pth, compression="lzf")

    backed_adata = ad.read_h5ad(backed_pth, backed="r")
    assert_equal(mem_adata, backed_adata.to_memory())

    # Test that raw can be removed
    del backed_adata.raw
    del mem_adata.raw
    assert_equal(mem_adata, backed_adata.to_memory())
