from __future__ import annotations


from anndata import AnnData
from anndata.tests.helpers import assert_equal, gen_adata


def test_attr_deletion():
    full = gen_adata((30, 30))
    # Empty has just X, obs_names, var_names
    empty = AnnData(None, obs=full.obs[[]], var=full.var[[]])
    for attr in ["X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns"]:
        delattr(full, attr)
        assert_equal(getattr(full, attr), getattr(empty, attr))
    assert_equal(full, empty, exact=True)
