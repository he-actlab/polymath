import polymath as pm
import numpy as np
import pytest

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


def test_gather0():
    axis=0
    x = np.random.randn(5, 4, 3, 2).astype(np.float32)
    idx = np.array([0, 1, 3])

    with pm.Node(name="gather_op") as graph:
        data = pm.input(name="input", shape=x.shape)
        indices = pm.input(name="indices", shape=idx.shape)
        out = pm.gather(data, indices, axis=axis, name="res")

    pm_y = graph("res", {"input": x, "indices": idx})
    np_y = np.take(x, idx, axis=axis)
    np.testing.assert_allclose(np_y, pm_y)

def test_gather2d():
    axis = 1
    x = np.random.randn(3, 3).astype(np.float32)
    idx = np.array([[0, 2]])

    with pm.Node(name="gather_op") as graph:
        data = pm.input(name="input", shape=x.shape)
        indices = pm.input(name="indices", shape=idx.shape)
        out = pm.gather(data, indices, axis=axis, name="res")

    pm_y = graph("res", {"input": x, "indices": idx})
    np_y = np.take(x, idx, axis=axis)
    np.testing.assert_allclose(np_y, pm_y)

def test_gather1():
    axis = 1
    x = np.random.randn(5, 4, 3, 2).astype(np.float32)
    idx = np.array([0, 1, 3])

    with pm.Node(name="gather_op") as graph:
        data = pm.input(name="input", shape=x.shape)
        indices = pm.input(name="indices", shape=idx.shape)
        out = pm.gather(data, indices, axis=axis, name="res")

    pm_y = graph("res", {"input": x, "indices": idx})
    np_y = np.take(x, idx, axis=axis)
    np.testing.assert_allclose(np_y, pm_y)

@pytest.mark.parametrize('in_shape, out_shape',[
    ((5, 100,), (1, 500,)),
    ((5, 100,), (5, 25, 4)),
])
def test_reshape(in_shape, out_shape):
    x = np.zeros(in_shape).astype(np.float32)

    with pm.Node(name="reshape_op") as graph:
        data = pm.input(name="input", shape=x.shape)
        out = pm.reshape(data, out_shape, name="res")

    pm_y = graph("res", {"input": x})
    np_y = np.reshape(x, out_shape)
    np.testing.assert_allclose(np_y, pm_y)
    assert np_y.shape == pm_y.shape


@pytest.mark.parametrize('in_shape, axis',[
    ((5, 100,), (1, 0)),
    ((3, 4, 5, 6), (3, 2, 1, 0)),
    ((3, 4, 5, 6), (1, 0, 2, 3)),
])
def test_transpose(in_shape, axis):
    x = np.random.randn(*in_shape).astype(np.float32)

    with pm.Node(name="transpose_op") as graph:
        data = pm.input(name="input", shape=x.shape)
        out = pm.transpose(data, axis, name="res")

    np_y = np.transpose(x, axis)
    pm_y = graph("res", {"input": x})
    np.testing.assert_allclose(np_y, pm_y)
    assert np_y.shape == pm_y.shape

@pytest.mark.parametrize('in_shape, axis',[
    ((5, 100,), (0,)),
    ((5, 100,), (0,1)),
    ((3, 4, 5, 6), (0, 1, 2)),
    ((3, 4, 5, 6), (1,)),
])
def test_flip(in_shape, axis):
    x = np.random.randn(*in_shape).astype(np.float32)

    with pm.Node(name="flip_op") as graph:
        data = pm.input(name="input", shape=x.shape)
        out = pm.flip(data, axis, name="res")

    np_y = np.flip(x, axis)
    pm_y = graph("res", {"input": x})
    np.testing.assert_allclose(np_y, pm_y)


@pytest.mark.parametrize('in_shape, pad_start, pad_end',[
    ((5, 100,), (0, 2), None),
    ((5, 100,), (0, 2), (0, 0)),
    ((3, 4, 5, 6), (1, 1, 1, 1), None),
    ((3, 4, 5, 6), (1, 1, 1, 1), (1, 0, 0, 1)),
])
def test_pad(in_shape, pad_start, pad_end):
    x = np.random.randn(*in_shape).astype(np.float32)

    with pm.Node(name="pad_op") as graph:
        data = pm.input(name="input", shape=x.shape)
        out = pm.pad(data, pad_start, pad_end=pad_end, name="res")

    if pad_end is None:
        padding_val = tuple((pad_start[i], pad_start[i]) for i in range(len(pad_start)))
    else:
        padding_val = tuple((pad_start[i], pad_end[i]) for i in range(len(pad_start)))
    np_y = np.pad(x, padding_val)
    pm_y = graph("res", {"input": x})
    assert np_y.shape == pm_y.shape
    np.testing.assert_allclose(np_y, pm_y)


