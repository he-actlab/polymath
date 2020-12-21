import polymath as pm
import numpy as np
from itertools import product
import pytest
from polymath.srdfg.templates.template_utils import _get_elem_indices


def test_index_op():
    with pm.Node(name="indexop") as graph:
        m = pm.parameter(name="m")
        n = pm.parameter(name="n")
        i = pm.index(0, m-1, name="i")
        j = pm.index(0, n-1, name="j")
        i_ = (i + 1).set_name("i_")

        k = (i + j).set_name("k")
    m_ = 5
    n_ = 3
    input_info = {"m": m_, "n": n_}
    res = graph("k", input_info)
    op1 = np.arange(0, m_)
    op2 = np.arange(0, n_)
    value = np.array(list(product(*(op1, op2))))
    value = np.array(list(map(lambda x: x[0]+x[1], value)))
    np.testing.assert_allclose(res, value)

@pytest.mark.parametrize('a_shape, b_shape, c_shape',[
    ((8, 1), (1, 16), (8, 16)),
    ((8, 2), (2, 16), (8, 2, 16))
])
def test_broadcast(a_shape, b_shape, c_shape):

    from einops import repeat
    with pm.Node(name="broadcast") as graph:
        a = pm.input("a", shape=a_shape)
        b = pm.input("b", shape=b_shape)
        c = pm.output("c", shape=c_shape)
        a_idx, b_idx, c_idx = _get_elem_indices(a, b, c)

        c[c_idx] = a[a_idx] + b[b_idx]

    a_np = np.random.randint(0, 32, np.prod(a_shape)).reshape(a_shape)
    b_np = np.random.randint(0, 32, np.prod(b_shape)).reshape(b_shape)
    if len(c_shape) > 2:
        c_np_out = np.zeros(c_shape)
    else:
        c_np_out = np.zeros((c_shape[0], 1, c_shape[1]))

    a_np_t = repeat(a_np, 'i k -> i k j', j=b_shape[1])
    b_np_t = repeat(b_np, 'i k -> j i k', j=a_shape[0])
    actual_res = (a_np_t + b_np_t).squeeze()
    graph_res = graph("c", {"a": a_np, "b": b_np})

    np.testing.assert_allclose(graph_res, actual_res)
