import polymath as pm
import numpy as np
from itertools import product

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
    assert np.allclose(res, value)