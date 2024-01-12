pytest reproducer
=================

When using the following together, pytest 7.4.4 fails to import from the correct module.
The bug triggers in pytest’s collection phase.

Necessary for the bug to trigger:

- ``--import-mode=importlib`` (I’m sure it’s also possible with ``append`` in another way)
- ``--doctest-modules``
- ``src/`` layout
- no editable install

Run with e.g.

.. code:: bash

   hatch run pytest [--import-mode=importlib]
   # or if you like typing
   python -m virtualenv .venv; source .venv/bin/activate; pip install .; pytest [--import-mode=importlib]

errors
------

When run with ``--import-mode=importlib``, I see:

.. code-block:: pytb

   ____ ERROR collecting _core/aligned_mapping.py ____
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/_core/aligned_mapping.py:15: in <module>
      from .index import _subset
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/_core/index.py:12: in <module>
      from ..compat import Index, Index1D
   E   ModuleNotFoundError: No module named 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata.compat'; 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata' is not a package
   ____ ERROR collecting _core/anndata.py ____
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/_core/anndata.py:24: in <module>
      from ..compat import _move_adj_mtx
   E   ModuleNotFoundError: No module named 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata.compat'; 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata' is not a package
   ____ ERROR collecting _core/index.py ____
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/_core/index.py:12: in <module>
      from ..compat import Index, Index1D
   E   ModuleNotFoundError: No module named 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata.compat'; 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata' is not a package
   ____ ERROR collecting _core/merge.py ____
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/_core/merge.py:28: in <module>
      from ..compat import _map_cat_to_str
   E   ModuleNotFoundError: No module named 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata.compat'; 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata' is not a package
   ____ ERROR collecting _core/raw.py ____
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/_core/raw.py:9: in <module>
      from .aligned_mapping import AxisArrays
   E   ImportError: cannot import name 'AxisArrays' from 'home.phil..local.share.hatch.env.virtual.anndata.kX3YdB0h.anndata.lib.python3.11.site-packages.anndata._core.aligned_mapping' (/home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/_core/aligned_mapping.py)

Otherwise, I see:

.. code-block:: pytb
   
   ____ ERROR collecting src/anndata/tests/test_base.py ____
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/_pytest/runner.py:341: in from_call
      result: Optional[TResult] = func()
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/_pytest/runner.py:372: in <lambda>
      call = CallInfo.from_call(lambda: list(collector.collect()), "collect")
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/_pytest/doctest.py:567: in collect
      module = import_path(
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/_pytest/pathlib.py:590: in import_path
      raise ImportPathMismatchError(module_name, module_file, path)
   E   _pytest.pathlib.ImportPathMismatchError: ('test_base', '/home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/tests/test_base.py', PosixPath('/home/phil/Dev/Python/_reproducers/pytest-doctest-import-mismatch/src/anndata/tests/test_base.py'))
   ____ ERROR collecting src/anndata/tests/test_base.py ____
   import file mismatch:
   imported module 'test_base' has this __file__ attribute:
   /home/phil/.local/share/hatch/env/virtual/anndata/kX3YdB0h/anndata/lib/python3.11/site-packages/anndata/tests/test_base.py
   which is not the same as the test file we want to collect:
   /home/phil/Dev/Python/_reproducers/pytest-doctest-import-mismatch/src/anndata/tests/test_base.py
   HINT: remove __pycache__ / .pyc files and/or use a unique basename for your test file modules