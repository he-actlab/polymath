import polymath as pm
import numpy as np
import functools

class dense(pm.Template):
    def define_graph(self, x, w, y, **kwargs):
        i = pm.index(0, (w.shape[1] - 1), name="i")
        j = pm.index(0, (w.shape[0] - 1), name="j")
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
        k = pm.index(0, data.shape[1] - 1, name="k")

        ihp = (data.shape[2] + pad*2)
        iwp = data.shape[3] + pad*2
        ihp_ = pm.index(0, ihp-1, name="ihp")
        iwp_ = pm.index(0, iwp-1, name="iwp")
        padded = pm.temp(name="padded", shape=(data.shape[0], data.shape[1], ihp, iwp))
        padded[b, k, ihp_, iwp_] = 0
        padded[b, k, iy + pad, ix + pad] = data[b, k, iy, ix]

        out[b, c, y, x] = pm.sum([dy, dx, k], (padded[b, k, dy + stride*y, dx + stride*x] * w[c, k, dy, dx]).set_name("p*w"))

class avg_pool2d(pm.Template):
    def define_graph(self, inp, out, kh, kw, stride, pad, **kwargs):

        oh = ((inp.shape[2] + 2 * pad - kh) // stride + 1)
        ow = ((inp.shape[3] + 2 * pad - kw) // stride + 1)
        out.set_shape((inp.shape[0], inp.shape[1], oh, ow))

        b = pm.index(0, inp.shape[0]-1)
        c = pm.index(0, inp.shape[1]-1)
        y = pm.index(0, oh-1)
        x = pm.index(0, ow-1)
        m = pm.index(0, kh-1)
        n = pm.index(0, kw-1)
        # padded = pm.temp(name="padded", shape=(ns, ic, ihp, iwp))
        # padded[b, k, ihp_, iwp_] = 0
        # padded[b, k, iy + pad, ix + pad] = data[b, k, iy, ix]
        out[b, c, y, x] = ((1/(kh*kw)) * pm.sum([m, n], inp[b, c, stride*y + m, stride*x + n])).set_name("final")

class max_pool2d(pm.Node):
    pass

class linear_classifier(pm.Node):
    pass

class normalizer(pm.Node):
    pass

class cast(pm.Node):
    pass

class zipmap(pm.Template):
    pass


class linear_regressor(pm.Template):

    def define_graph(self, x, w, y_pred, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")


class logistic_regressor(pm.Template):

    def define_graph(self, x, w, y_pred, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h"))


class mc_logistic_regressor_train(pm.Template):

    def define_graph(self, x, w, y, y_pred, mu, m, **kwargs):
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.placeholder(name="h", shape=(m))
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

class reshape(pm.Template):
    def define_graph(self, *args, **kwargs):
        pass

class gemm(pm.Template):
    def define_graph(self, *args, **kwargs):
        pass

class dropout(pm.Template):
    def define_graph(self, *args, **kwargs):
        pass


class log_softmax(pm.Template):
    def define_graph(self, *args, **kwargs):
        pass

class softmax(pm.Template):
    def define_graph(self, data, out, **kwargs):
        e = pm.parameter(name="e", default=np.e)
        out.set_shape(data.shape)
        i = pm.index(0, data.shape[0]-1)
        j = pm.index(0, data.shape[0]-1)
        out[i] = (e ** (data[i]))/ pm.sum([j], e**data[j], name="num")

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

# TODO: Add reshape operator, constant operator, gemm
NODE_NAMES = {"SVMClassifier": svm_classifier_train,
              "Conv": conv,
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
              "Dropout": dropout,
              "LogSoftmax": log_softmax}

# class _TemplateWrapper(object):
#     def __init__(self, def_func):
#         self.def_func = def_func
#
#     def __call__(self, *args, **kwargs):
#         temp = pm.Template()
#
# def templateop(target=None):
#     """
#     Decorator for creating nodes from functions.
#     """
#     # This is called when the decorator is used with arguments
#     if target is None:
#         return functools.partial(templateop)
#
#     # This is called when the decorator is used without arguments
#     @functools.wraps(target)
#     def _wrapper(*args, **kwargs_inner):
#         temp = pm.Template(*args, **kwargs_inner)
#         temp.define_graph = target
#         temp.
#         return temp
#     return _wrapper