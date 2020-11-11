import polymath as pm
import numpy as np

def test_unsqueeze():
    with pm.Node(name="indexop") as graph:
        m = pm.parameter(name="m")
        n = pm.parameter(name="n")
        x = pm.state("x", shape=(m, n))
        x_us = pm.unsqueeze(x, axis=1, name="res")
    m_ = 5
    n_ = 3
    x_ = np.random.randint(0, 10, (m_, n_))

    input_info = {"m": m_, "n": n_, "x": x_}
    res = graph("res", input_info)

    np.testing.assert_allclose(res, np.expand_dims(x_, axis=1))

def test_squeeze():
    with pm.Node(name="indexop") as graph:
        m = pm.parameter(name="m")
        n = pm.parameter(name="n")
        x = pm.state("x", shape=(m, n))
        x_us = pm.squeeze(x, axis=None, name="res")
    m_ = 5
    n_ = 1
    x_ = np.random.randint(0, 10, (m_, n_))
    input_info = {"m": m_, "n": n_, "x": x_}
    res = graph("res", input_info)

    np.testing.assert_allclose(res, np.squeeze(x_, axis=1))

def test_flatten():
    shape = (2, 3, 4, 5)
    a = np.random.random_sample(shape).astype(np.float32)
    for i in range(len(shape)):
        with pm.Node(name="flatten_op") as graph:
            x = pm.state("x", shape=shape)
            x_us = pm.flatten(x, axis=i, name="res")

        new_shape = (1, -1) if i == 0 else (np.prod(shape[0:i]).astype(int), -1)
        b = np.reshape(a, new_shape)
        pm_b = graph("res", {"x": a})
        np.testing.assert_allclose(pm_b, b)

def quick_flatten(a, new_shape):
    from itertools import product
    b = np.empty(new_shape)
    in_idx = tuple([list(range(i)) for i in a.shape])
    out_idx = tuple([list(range(i)) for i in new_shape])
    perm_in = list(product(*in_idx))
    perm_out = list(product(*out_idx))
    for a_idx, b_idx in zip(tuple(perm_in), tuple(perm_out)):
        b[b_idx] = a[a_idx]
    return b

def test_int():
    shape = (2, 3, 4, 5)
    a = np.random.random_sample(shape).astype(np.float32)

    for i in range(-len(shape), 0):
        new_shape = (np.prod(shape[0:i]).astype(int), -1)
        b = a.reshape(new_shape)
        tb = quick_flatten(a, b.shape)
        np.testing.assert_allclose(tb, b)