from __future__ import annotations


def assert_str_contents_equal(A, B):
    lA = [
        [str(el) if not isinstance(el, bytes) else el.decode("utf-8") for el in a]
        for a in A
    ]
    lB = [
        [str(el) if not isinstance(el, bytes) else el.decode("utf-8") for el in b]
        for b in B
    ]
    assert lA == lB
