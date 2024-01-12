pytest reproducer
=================

When using the following together, pytest 7.4.4 tries to import from different 
The bug triggers in pytest’s collection phase.

Necessary for the bug to trigger:

- ``--import-mode=importlib`` (I’m sure it’s also possible with ``append`` in another way)
- ``--doctest-modules``
- ``src/`` layout
- no editable install

Run with e.g.

.. code:: bash

   hatch run pytest
   # or if you like typing
   python -m virtualenv .venv; source .venv/bin/activate; pip install .; pytest

No matter if you opt to install in dev mode like this, specify ``PYTHONPATH``,
or install regularly, the bug is reproducible in any case.
