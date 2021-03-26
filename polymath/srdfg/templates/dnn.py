import polymath as pm
from .template_utils import _get_indices, _get_single_node_indices
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools

class avg_pool(pm.Template):
    def define_graph(self, data, out, kh, kw, stride=(1,1), pad=(0,0)):
        sx, sy = stride
        oh = ((data.shape[-2] + 2 * pad[0] - kh) // stride[0] + 1)
        ow = ((data.shape[-1] + 2 * pad[1] - kw) // stride[1] + 1)

        y = pm.index(0, oh - 1, name="y")
        x = pm.index(0, ow - 1, name="x")
        m = pm.index(0, kh - 1, name="m")
        n = pm.index(0, kw - 1, name="n_")
        ihp = (data.shape[-2] + pad[0] * 2)
        iwp = data.shape[-1] + pad[1] * 2
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        iy = pm.index(0, data.shape[-2] - 1, name="iy")
        ix = pm.index(0, data.shape[-1] - 1, name="ix")

        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1, name="b")
            c = pm.index(0, data.shape[1] - 1, name="c")

            o_indices = [b, c]
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], data.shape[1], oh, ow))

        else:
            c = pm.index(0, data.shape[0] - 1, name="c")
            o_indices = [c]
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((data.shape[0], oh, ow))
        o_indices = tuple(o_indices)
        padded = pm.temp(name="padded", shape=p_shape)
        padded[o_indices + (ihp_, iwp_)] = 0
        padded[o_indices + (iy + pad[0], ix + pad[1])] = data[o_indices + (iy, ix)]
        out[o_indices + (y, x)] = pm.sum([m, n], padded[o_indices + (sx*y + m, sy*x + n)]) * (1/(kh*kw))

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class dense(pm.Template):
    def define_graph(self, x, w, y):
        i = pm.index(0, (w.shape[1] - 1), name="i")
        j = pm.index(0, (w.shape[0] - 1), name="j")
        y.set_shape((w.shape[0]))
        y[j] = pm.sum([i], w[j, i] * x[i], name="h")

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class dense_sigmoid(pm.Template):
    def define_graph(self, x, w, y):
        i = pm.index(0, (w.shape[1] - 1), name="i")
        j = pm.index(0, (w.shape[0] - 1), name="j")
        y[j] = pm.sigmoid(pm.sum([i], w[j, i] * x[i], name="h"))

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)



class relu(pm.Template):
    def define_graph(self, inp, out):
        out.set_shape(inp.shape)
        # indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in inp.shape])
        indices = tuple([pm.index(0, s - 1) for s in inp.shape])
        out[indices] = (0 < inp[indices]) * inp[indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class leaky_relu(pm.Template):
    def define_graph(self, inp, out, alpha=1e-2):
        out.set_shape(inp.shape)
        indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in inp.shape])
        out[indices] = (0 < inp[indices]) * inp[indices] + (0 >= inp[indices]) * inp[indices] * alpha

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class relu1d(pm.Template):
    def define_graph(self, inp, out):
        i = pm.index(0, inp.shape[0] - 1, name="i")
        out.set_shape(inp.shape)
        out.write((0 < inp[i]) * inp[i])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class conv_bias(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0):
        oh = ((data.shape[-2] + 2 * pad - w.shape[-2]) // stride + 1)
        ow = ((data.shape[-1] + 2 * pad - w.shape[-1]) // stride + 1)
        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1, name="y_")
        x = pm.index(0, ow - 1, name="x_")
        dy = pm.index(0, w.shape[2] - 1, name="dy")
        dx = pm.index(0, w.shape[3] - 1, name="dx")
        iy = pm.index(0, data.shape[-2] - 1, name="iy")
        ix = pm.index(0, data.shape[-1] - 1, name="ix")
        k = pm.index(0, data.shape[-3] - 1, name="k")
        ihp = (data.shape[-2] + pad * 2)
        iwp = data.shape[-1] + pad * 2
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], w.shape[0], oh, ow))

        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((w.shape[0], oh, ow))
        padded = pm.temp(name="padded", shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad, ix + pad)] = data[p_indices + (iy, ix)]

        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

class avg_pool2d(pm.Template):
    def define_graph(self, inp, out, kh, kw, stride=1, pad=0):
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

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)


class batch_flatten(pm.Template):
    def define_graph(self, data, out):
        out.set_shape((data.shape[0]*data.shape[1]*data.shape[2]*data.shape[3],))
        m = data.shape[1]
        n = data.shape[2]
        p = data.shape[3]

        i = pm.index(0, data.shape[0]-1, name="i")
        j = pm.index(0, data.shape[1]-1, name="j")
        k = pm.index(0, data.shape[2]-1, name="k")
        l = pm.index(0, data.shape[3]-1, name="l")
        out[((i*m + j)*n + k)*p + l] = data[i, j, k, l]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class batch_norm(pm.Template):
    def define_graph(self, x, scale, b, mean, var, out, eps=1e-05, momentum=0.9, spatial=1):
        indices = _get_single_node_indices(out, shape=out.shape)
        if len(out.shape) > 3:
            i = indices[1]
        else:
            i = indices[0]
        out[indices] = scale[i]*(x[indices] - mean[i])/pm.sqrt(var[i] + eps) + b[i]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4])

    @property
    def outputs(self):
        return (self.args[5],)


class elem_sigmoid(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.sigmoid(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_cast(pm.Template):
    def define_graph(self, x, out, to):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.cast(to, x[indices], shape=out.shape)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class softmax(pm.Template):
    def define_graph(self, data, out, axis=0):
        out.set_shape(data.shape)
        i = pm.index(0, data.shape[axis]-1, name="i")
        j = pm.index(0, data.shape[axis]-1, name="j")
        indices = [pm.index(0, s - 1, name=f"{data.name}[{i}]") for i, s in enumerate(data.shape)]
        indices_denom = indices
        indices_denom[axis] = j
        indices[axis] = i
        indices = tuple(indices)
        indices_denom = tuple(indices_denom)
        mval = pm.max([i], data[indices], name="max_test")
        e_x = pm.exp((data[indices] - mval), name="e_x")
        out[indices] = e_x[indices] / pm.sum([indices_denom[axis]], e_x[indices_denom], name="denom")

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_tanh(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.tanh(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)



class dropout(pm.Template):
    # TODO: Fix and test indices here
    def define_graph(self, x, y, ratio=0.0):
        indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in y.shape])
        y[indices] = x[indices] * 1.0 / (1 - ratio)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)



class global_avg_pool(pm.Template):
    def define_graph(self, x, out):
        # indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in shape])
        indices = _get_single_node_indices(out, shape=out.shape)
        m = pm.index(0, x.shape[2]-1)
        n = pm.index(0, x.shape[3]-1)
        h = x.shape[2]
        w = x.shape[3]
        out[indices] = (1/(h*w)) * pm.sum([m, n], x[indices[0], indices[1], m, n])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class conv(pm.Template):
    def define_graph(self, data, w, out, stride=1, pad=0):
        oh = ((data.shape[-2] + 2 * pad - w.shape[-2]) // stride + 1)
        ow = ((data.shape[-1] + 2 * pad - w.shape[-1]) // stride + 1)
        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2]-1, name="dy")
        dx = pm.index(0, w.shape[3]-1, name="dx")
        iy = pm.index(0, data.shape[-2]-1, name="iy")
        ix = pm.index(0, data.shape[-1]-1, name="ix")
        k = pm.index(0, data.shape[-3]-1, name="k")
        ihp = (data.shape[-2] + pad*2)
        iwp = data.shape[-1] + pad*2
        ihp_ = pm.index(0, ihp-1, name="ihp")
        iwp_ = pm.index(0, iwp-1, name="iwp")
        # im2col
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0]-1, name="b")
            o_indices = [b,c]
            p_indices = [b, k,]
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], w.shape[0], oh, ow))

        else:
            o_indices = [c]
            p_indices = [k]
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((w.shape[0], oh, ow))

        padded = pm.temp(name="padded", shape=p_shape)
        padded[tuple(p_indices + [ihp_, iwp_])] = 0
        padded[tuple(p_indices + [iy + pad, ix + pad])] = data[tuple(p_indices + [ iy, ix])]
        out[tuple(o_indices + [y, x])] = pm.sum([dy, dx, k], (padded[tuple(p_indices + [dy + stride*y, dx + stride*x])] * w[c, k, dy, dx]))

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)


class lrn(pm.Template):
    def define_graph(self, x, y, alpha, beta, bias, nsize):
        n = pm.index(0, x.shape[0] - 1)
        c = pm.index(0, x.shape[1] - 1)
        h = pm.index(0, x.shape[2] - 1)
        w = pm.index(0, x.shape[3] - 1)
        c_ = pm.index(0, x.shape[1] - 1)
        ext = pm.temp(name="extended", shape=tuple([*x.shape, x.shape[-3]]))

        bounds = pm.output(name="bounds", shape=(x.shape[1], x.shape[1]))
        radius = nsize//2
        hbool = ((((x.shape[1] > (c + radius + 1)) * (c + radius)) + (x.shape[1] <= (c + radius + 1)) * (
                    x.shape[1] - 1)) >= c_)
        lbool = ((((c - radius) > 0) * (c - radius)) + (((c - radius) <= 0) * 0) <= c_)
        bounds[c, c_] = hbool*lbool
        ext[n, c, h, w, c_] = x[n, c_, h, w] * bounds[c, c_]
        # y[n, c, h, w] = x[n,c,h,w] / ((bias + (alpha/nsize) * pm.sum([c_], ext[n, c, h, w, c_]**2))**beta)
        y[n, c, h, w] = x[n,c,h,w] / ((bias + (alpha / nsize) * pm.sum([c_], ext[n, c, h, w, c_]**2)) ** beta)

    @property
    def inputs(self):
        return (self.args[0], self.args[2], self.args[3], self.args[4], self.args[5])

    @property
    def outputs(self):
        return (self.args[1],)



class max_pool(pm.Template):
    def define_graph(self, data, out, kh, kw, stride=(1,1), pad=(0,0)):

        oh = ((data.shape[-2] + 2 * pad[0] - kh) // stride[0] + 1)
        ow = ((data.shape[-1] + 2 * pad[1] - kw) // stride[1] + 1)

        y = pm.index(0, oh-1, name="y")
        x = pm.index(0, ow-1, name="x")
        m = pm.index(0, kh-1, name="m")
        n = pm.index(0, kw-1, name="n_")
        ihp = (data.shape[-2] + pad[0] * 2)
        iwp = data.shape[-1] + pad[1] * 2
        ihp_ = pm.index(0, ihp-1, name="ihp")
        iwp_ = pm.index(0, iwp-1, name="iwp")
        iy = pm.index(0, data.shape[-2] - 1, name="iy")
        ix = pm.index(0, data.shape[-1] - 1, name="ix")

        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0]-1, name="b")
            c = pm.index(0, data.shape[1] - 1, name="c")

            o_indices = (b,c)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], data.shape[1], oh, ow))

        else:
            c = pm.index(0, data.shape[0] - 1, name="c")
            o_indices = (c,)
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((data.shape[0], oh, ow))

        padded = pm.temp(name="padded", shape=p_shape)
        padded[o_indices, ihp_, iwp_] = 0
        padded[o_indices, iy + pad[0], ix + pad[1]] = data[o_indices, iy, ix]
        out[o_indices, y, x] = pm.max([m, n], padded[o_indices, stride[0]*y + m, stride[1]*x + n])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def stride(self):
        return self.args[4]

    @property
    def kernel_size(self):
        return (self.args[2], self.args[3])

    @property
    def pad(self):
        return self.args[4]