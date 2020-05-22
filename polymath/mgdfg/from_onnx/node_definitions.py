import polymath as pm
from polymath.mgdfg.util import squeeze_shape
import numpy as np
import functools

class dense(pm.Template):
    def define_graph(self, x, w, y, **kwargs):
        i = pm.index(0, (w.shape[1] - 1), name="i")
        j = pm.index(0, (w.shape[0] - 1), name="j")
        y.set_shape((w.shape[0]))
        y[j] = pm.sum([i], w[j, i] * x[i], name="h")

class dense_sigmoid(pm.Template):
    def define_graph(self, x, w, y, **kwargs):
        i = pm.index(0, (w.shape[1] - 1), name="i")
        j = pm.index(0, (w.shape[0] - 1), name="j")
        y[j] = pm.sigmoid(pm.sum([i], w[j, i] * x[i], name="h"))

class dense_bp(pm.Template):
    def define_graph(self, x, w, m, n, y, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        j = pm.index(0, (n - 1).set_name("n-1"), name="j")
        y[j] = pm.sigmoid(pm.sum([i], w[j, i] * x[i], name="h"))

class bench_bp(pm.Template):
    def define_graph(self, x, w1, w2, l1, l2, l3, y, **kwargs):

        i1 = pm.index(0, (w1.shape[1] - 1), name="i1")
        i2 = pm.index(0, (w1.shape[0] - 1), name="i2")
        i3 = pm.index(0, (w2.shape[0] - 1), name="i3")
        a1 = pm.temp("a1", shape=w1.shape[0])
        a1[i2] = pm.sigmoid(pm.sum([i1], w1[i2, i1] * x[i1], name="h1"))
        a2 = pm.temp("a2", shape=w2.shape[0])
        a2[i3] = pm.sigmoid(pm.sum([i2], w2[i3, i2] * a1[i2], name="h2"))


class svm_classifier_train(pm.Template):
    def define_graph(self, x, w, y, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")
        c = (y*h).set_name("c")
        ny = (0 - y).set_name("ny")
        p = ((c > 1)*ny).set_name("p")
        g = (p * x[i]).set_name("g")
        w[i] = w[i] - mu * g[i]

class logistic_regressor_train(pm.Template):

    def define_graph(self, x, w, y, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h"))
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]

class relu(pm.Template):
    def define_graph(self, inp, out, **kwargs):
        i = pm.index(0, inp.shape[0] - 1, name="i")
        j = pm.index(0, inp.shape[1] - 1, name="j")
        k = pm.index(0, inp.shape[2] - 1, name="k")
        l = pm.index(0, inp.shape[3] - 1, name="l")
        out.set_shape(inp.shape)
        out.write((0 < inp[i, j, k, l]) * inp[i, j, k, l])

class relu1d(pm.Template):
    def define_graph(self, inp, out, **kwargs):
        i = pm.index(0, inp.shape[0] - 1, name="i")
        out.set_shape(inp.shape)
        out.write((0 < inp[i]) * inp[i])

class conv(pm.Template):
    def define_graph(self, data, w, bias, out, stride, pad):
        oh = ((data.shape[2] + 2 * pad - w.shape[2]) // stride + 1).set_name("oh")
        ow = ((data.shape[3] + 2 * pad - w.shape[3]) // stride + 1).set_name("ow")
        out.set_shape((data.shape[0], w.shape[0], oh, ow))
        # im2col
        b = pm.index(0, data.shape[0]-1, name="b")
        c = pm.index(0, w.shape[0]-1, name="c")
        y = pm.index(0, oh-1, name="y")
        x = pm.index(0, ow-1, name="x")
        dy = pm.index(0, w.shape[2]-1, name="dy")
        dx = pm.index(0, w.shape[3]-1, name="dx")
        iy = pm.index(0, data.shape[2]-1, name="iy")
        ix = pm.index(0, data.shape[3]-1, name="ix")
        k = pm.index(0, data.shape[1]-1, name="k")

        ihp = (data.shape[2] + pad*2)
        iwp = data.shape[3] + pad*2
        ihp_ = pm.index(0, ihp-1, name="ihp")
        iwp_ = pm.index(0, iwp-1, name="iwp")
        padded = pm.temp(name="padded", shape=(data.shape[0], data.shape[1], ihp, iwp))
        padded[b, k, ihp_, iwp_] = 0
        padded[b, k, iy + pad, ix + pad] = data[b, k, iy, ix]

        out[b, c, y, x] = (pm.sum([dy, dx, k], (padded[b, k, dy + stride*y, dx + stride*x] * w[c, k, dy, dx]).set_name("p*w"), name="conv_sum") + bias[c]).set_name("c+bias")

class avg_pool2d(pm.Template):
    def define_graph(self, inp, out, kh, kw, stride, pad, **kwargs):

        oh = ((inp.shape[2] + 2 * pad - kh) // stride + 1)
        ow = ((inp.shape[3] + 2 * pad - kw) // stride + 1)
        out.set_shape((inp.shape[0], inp.shape[1], oh, ow))

        b = pm.index(0, inp.shape[0]-1, name="b")
        c = pm.index(0, inp.shape[1]-1, name="c")
        y = pm.index(0, oh-1, name="y")
        x = pm.index(0, ow-1, name="x")
        m = pm.index(0, kh-1, name="m")
        n = pm.index(0, kw-1, name="n_")
        ihp = (inp.shape[2] + pad*2)
        iwp = inp.shape[3] + pad*2
        ihp_ = pm.index(0, ihp-1, name="ihp")
        iwp_ = pm.index(0, iwp-1, name="iwp")
        iy = pm.index(0, inp.shape[2]-1, name="iy")
        ix = pm.index(0, inp.shape[3]-1, name="ix")
        padded = pm.temp(name="padded", shape=(inp.shape[0], inp.shape[1], ihp, iwp))
        padded[b, c, ihp_, iwp_] = 0
        padded[b, c, iy + pad, ix + pad] = inp[b, c, iy, ix]
        out[b, c, y, x] = ((1/(kh*kw)) * pm.sum([m, n], padded[b, c, stride*y + m, stride*x + n], name="apool_sum")).set_name("final")

class max_pool2d(pm.Node):
    pass

class linear_classifier(pm.Node):
    pass

class normalizer(pm.Node):
    pass


class zipmap(pm.Template):
    pass


class linear_regressor(pm.Template):

    def define_graph(self, x, w, y_pred, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        y_pred.write(pm.sum([i], (x[i] * w[i]), name="h"))


class logistic_regressor(pm.Template):

    def define_graph(self, x, w, y_pred, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        y_pred.write(pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h")))


class mc_logistic_regressor_train(pm.Template):

    def define_graph(self, x, w, y, y_pred, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.temp(name="h", shape=(m))
        h = pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h"))
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]

class mc_logistic_regressor(pm.Template):

    def define_graph(self, x, w, y_pred, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h"))

class linear_regressor_train(pm.Template):

    def define_graph(self, x, w, y, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]

class dropout(pm.Template):
    def define_graph(self, *args, **kwargs):
        pass


class log_softmax(pm.Template):
    def define_graph(self, *args, **kwargs):
        pass

class softmax(pm.Template):
    def define_graph(self, data, out, **kwargs):
        out.set_shape(data.shape)
        i = pm.index(0, data.shape[0]-1)
        j = pm.index(0, data.shape[0]-1)
        mval = pm.max([i], data[i], name="max_test")
        e_x = pm.exp((data[i] - mval))
        out[i] = e_x[i] / pm.sum([j], e_x[j], name="num")

class batch_flatten(pm.Template):
    def define_graph(self, data, out, **kwargs):
        out.set_shape((data.shape[0]*data.shape[1]*data.shape[2]*data.shape[3],))
        m = data.shape[1]
        n = data.shape[2]
        p = data.shape[3]

        i = pm.index(0, data.shape[0]-1, name="i")
        j = pm.index(0, data.shape[1]-1, name="j")
        k = pm.index(0, data.shape[2]-1, name="k")
        l = pm.index(0, data.shape[3]-1, name="l")
        out[((i*m + j)*n + k)*p + l] = data[i, j, k, l]

def reduce_sum(data, axes=None, keepdims=None, shape=None, name=None, **kwargs):
    i = pm.index(0, data.shape[axes] - 1)
    return pm.sum([i], data[i], name=name)

def elem_greater(a, b, shape=None, name=None, **kwargs):
    indices = tuple([pm.index(0, s - 1) for s in shape])
    a_idx = _get_indices(a, indices, shape)
    b_idx = _get_indices(b, indices, shape)

    return (a[a_idx] < b[b_idx]).set_name(name)

def elem_sub(a, b, shape=None, name=None, **kwargs):
    indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in shape])
    a_idx = _get_indices(a, indices, shape)
    b_idx = _get_indices(b, indices, shape)
    return (a[a_idx] - b[b_idx]).set_name(name)

def elem_mul(a, b, shape=None, name=None, **kwargs):

    indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in shape])
    a_idx = _get_indices(a, indices, shape)
    b_idx = _get_indices(b, indices, shape)

    return (a[a_idx] * b[b_idx]).set_name(name)

def elem_sigmoid(x, shape=None, name=None, **kwargs):
    indices = tuple([pm.index(0, s - 1) for s in shape])
    ret = pm.sigmoid(x[indices]).set_name(name)
    return ret

def cast(data, to=None, shape=None, name=None, **kwargs):
    indices = tuple([pm.index(0, s - 1) for s in shape])
    out = pm.temp(name=name, shape=shape, dtype=to)
    out[indices] = data[indices]
    return out

def unsqueeze(x, axis, *args, name=None, **kwargs):
    x.graph.nodes[name] = x
    return x

# TODO: Check this works after changes
def squeeze(x, axis, *args, shape=None, name=None, **kwargs):
    x.graph.nodes[name] = x
    return x

def matmul(a, b, shape=None, name=None, **kwargs):
    i = pm.index(0, a.shape[0] - 1)
    j = pm.index(0, b.shape[0] - 1)
    k = pm.index(0, b.shape[1] - 1)
    return pm.sum([j], a[i, j]*b[j, k], name=name)

def lvmatmul(a, b, shape=None, name=None, **kwargs):
    i = pm.index(0, a.shape[0] - 1)
    j = pm.index(0, b.shape[1] - 1)
    return pm.sum([i], a[i]*b[i, j], name=name)

def rvmatmul(a, b, shape=None, name=None, **kwargs):
    i = pm.index(0, a.shape[0] - 1)
    j = pm.index(0, b.shape[0] - 1)
    return pm.sum([j], a[i, j]*b[j], name=name)

def get_matmul(a, b, **kwargs):

    if len(a.shape) == len(b.shape):
        return matmul(a,b,**kwargs)
    elif len(a.shape) > len(b.shape):
        return rvmatmul(a, b, **kwargs)
    else:
        return lvmatmul(a, b, **kwargs)

def get_elem(a, b, **kwargs):

    if len(a.shape) == len(b.shape):
        return matmul(a,b,**kwargs)
    elif len(a.shape) > len(b.shape):
        return rvmatmul(a, b, **kwargs)
    else:
        return lvmatmul(a, b, **kwargs)

# TODO: Check this works after changes
def reshape(data, *args, shape=None, name=None, **kwargs):
    data._shape = shape
    data.graph.nodes[name] = data
    return data

# TODO: Check this works after changes
def gemm(a, b, shape=None, name=None, **kwargs):
    i = pm.index(0, shape[0] - 1)
    j = pm.index(0, b.shape[0] - 1)
    k = pm.index(0, shape[1] - 1)
    return pm.sum([j], a[i, j]*b[j, k], name=name)

def transpose(data, shape=None, name=None, **kwargs):
    indices = tuple([pm.index(0, s - 1) for s in shape])
    out = pm.temp(name=name, shape=shape)
    out[indices] = data[tuple(reversed(indices))]

    return out

def identity(data, shape=None, name=None, **kwargs):
    data.set_name(name)
    return data

def _get_indices(node, all_indices, tgt_shape):
    indices = []

    if node.shape == pm.DEFAULT_SHAPES[0]:
        return tuple(indices)

    for idx, i in enumerate(all_indices):
        if len(node.shape) > idx and tgt_shape[idx] == node.shape[idx]:
            indices.append(i)

    if tgt_shape != node.shape:
        for idx, i in enumerate(node.shape):
            if i != tgt_shape[idx]:
                indices.insert(idx, 0)
    return tuple(indices)


# TODO: Add reshape operator, constant operator, gemm
NODE_NAMES = {"SVMClassifier": svm_classifier_train,
              "Conv": conv,
              "MatMul": get_matmul,
              "MaxPool": max_pool2d,
              "Relu": relu,
              "LinearClassifier": linear_classifier,
              "ZipMap": zipmap,
              "LinearRegressor": linear_regressor_train,
              "Cast": cast,
              "Normalizer": normalizer,
              "Constant": pm.parameter,
              "Reshape": reshape,
              "Gemm": gemm,
              "Identity": identity,
              "Dropout": dropout,
              "LogSoftmax": log_softmax,
              "Sigmoid": elem_sigmoid,
              "Mul": elem_mul,
              "ReduceSum": reduce_sum,
              "Unsqueeze": unsqueeze,
              "Squeeze": squeeze,
              "Sub": elem_sub,
              "Transpose": transpose,
              "Greater": elem_greater}
