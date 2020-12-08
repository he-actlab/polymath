import polymath as pm
from polymath.mgdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools

class avg_pool(pm.Template):
    def define_graph(self, data, out, kh, kw, stride, pad, **kwargs):
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
    def define_graph(self, x, w, y, **kwargs):
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
    def define_graph(self, x, w, y, **kwargs):
        i = pm.index(0, (w.shape[1] - 1), name="i")
        j = pm.index(0, (w.shape[0] - 1), name="j")
        y[j] = pm.sigmoid(pm.sum([i], w[j, i] * x[i], name="h"))

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

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
    def define_graph(self, inp, out, alpha=1e-2, **kwargs):
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
    def define_graph(self, inp, out, **kwargs):
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
    def define_graph(self, data, w, bias, out, stride, pad):
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

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

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

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class batch_norm(pm.Template):
    def define_graph(self, x, scale, b, mean, var, out, eps=1e-05, momentum=0.9, spatial=1, shape=None, name=None, **kwargs):
        indices = _get_single_node_indices(out, shape=shape)
        if len(shape) > 3:
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

class matmul(pm.Template):
    def define_graph(self, a, w, out, shape=None, name=None, **kwargs):
        indices = _get_single_node_indices(a)
        sum_idx = indices[-1]
        o_idx = pm.index(0, w.shape[0]-1) if w.shape[-1] == a.shape[-1] else pm.index(0, w.shape[1]-1)
        w_idx = (o_idx, sum_idx) if w.shape[-1] == a.shape[-1] else (sum_idx, o_idx)
        out_idx = indices[:-1] + (o_idx,)
        out[out_idx] = pm.sum([sum_idx], a[indices]*w[w_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_sigmoid(pm.Template):
    def define_graph(self, x, out, shape=None, name=None):
        indices = _get_single_node_indices(out, shape=shape)
        out[indices] = pm.sigmoid(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_cast(pm.Template):
    def define_graph(self, x, out, to, shape=None, name=None):
        indices = _get_single_node_indices(out, shape=shape)
        out[indices] = pm.cast(to, x[indices], shape=shape)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class softmax(pm.Template):
    def define_graph(self, data, out, axis=0, shape=None, name=None, **kwargs):
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
    def define_graph(self, x, out, shape=None, name=None):
        indices = _get_single_node_indices(out, shape=shape)
        out[indices] = pm.tanh(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class transpose(pm.Template):
    def define_graph(self, data, out, shape=None, name=None, **kwargs):
        # indices = tuple([pm.index(0, s - 1) for s in data.shape])
        indices = _get_single_node_indices(data)
        rev_idx = tuple(reversed(indices))
        out[rev_idx] = data[indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class reduce_sum(pm.Template):
    def define_graph(self, data, out, axes=(0,), keepdims=True, shape=None, name=None, **kwargs):
        indices = _get_single_node_indices(data)
        # indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axes])
        out_idx = tuple([indices[i] for i in axes if i not in axes])
        out[out_idx] = pm.sum([sum_idx], data[indices])


# def reduce_sum(data, axes=None, keepdims=None, shape=None, name=None, **kwargs):
#     i = pm.index(0, data.shape[axes] - 1)
#     return pm.sum([i], data[i], name=name)


class elem_greater(pm.Template):
    def define_graph(self, a, b, out, shape=None, name=None, **kwargs):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # a_idx, b_idx, indices = _get_binop_idx(a, b, out)

        out[indices] = (a[a_idx] > b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_sub(pm.Template):
    def define_graph(self, a, b, out, shape=None, name=None, **kwargs):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # a_idx, b_idx, indices = _get_binop_idx(a, b, out)
        out[indices] = (a[a_idx] - b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_add(pm.Template):
    def define_graph(self, a, b, out, shape=None, name=None, **kwargs):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # a_idx, b_idx, indices = _get_binop_idx(a, b, out)
        out[indices] = (a[a_idx] + b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_mul(pm.Template):
    def define_graph(self, a, b, out, shape=None, name=None, **kwargs):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        out[indices] = (a[a_idx] * b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)



# TODO: Need to fix this functionality to create a new node
def unsqueeze(x, *args, axes=None, shape=None, name=None, **kwargs):
    out = pm.unsqueeze(x, axis=axes, name=name, shape=shape)
    return out

# TODO: Check this works after changes
def squeeze(x, *args, axes=None, shape=None, name=None, **kwargs):
    out = pm.squeeze(x, axis=axes, name=name, shape=shape)
    return out

def lvmatmul(a, b, shape=None, name=None, **kwargs):
    i = pm.index(0, a.shape[0] - 1)
    j = pm.index(0, b.shape[1] - 1)
    return pm.sum([i], a[i]*b[i, j], name=name)

def rvmatmul(a, b, shape=None, name=None, **kwargs):
    i = pm.index(0, a.shape[0] - 1)
    j = pm.index(0, b.shape[0] - 1)
    return pm.sum([j], a[i, j]*b[j], name=name)

class coarse_flatten(pm.Template):
    def define_graph(self, data, out, axis=1, shape=None, **kwargs):
        o_indices = _get_single_node_indices(out, shape=shape)
        i_indices = _get_single_node_indices(data, shape=shape)
        out[o_indices] = data[i_indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class ppo(pm.Template):
    def define_graph(self, obs, action, states,
                     gamma=0.99,
                     clip=0.2,
                     ent_coeff=0.01,
                     lam=0.95,
                     adam_eps=1e-5):
        pass


class dropout(pm.Template):
    # TODO: Fix and test indices here
    def define_graph(self, x, y, ratio=0.0, name=None, shape=None, **kwargs):
        indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in shape])
        y[indices] = x[indices] * 1.0 / (1 - ratio)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

# TODO: Check this works after changes
def reshape(data, *args, shape=None, name=None, **kwargs):
    data._shape = shape
    data.graph.nodes[name] = data
    return data

# TODO: Convert this to a template node
def resize(data, *args, shape=None, name=None, **kwargs):

    data._shape = shape
    data.graph.nodes[name] = data
    return data

def identity(data, shape=None, name=None, **kwargs):
    data.set_name(name)
    return data

def _get_single_node_indices(node, shape=None):
    if node.shape == pm.DEFAULT_SHAPES[0]:
        return tuple([])
    else:
        if not shape:
            shape = node.shape
        indices = tuple([pm.index(0, s - 1) for s in shape])
        return indices

def _get_reduce_node_indices(a, b, output, axis):
    if output.shape == pm.DEFAULT_SHAPES[0]:
        return tuple([])
    else:
        if not output.shape:
            raise RuntimeError
        indices = tuple([pm.index(0, s - 1) for s in output.shape])
        return indices


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True

# Use numpy broadcasting rules
def _get_elem_indices(node_a, node_b, node_c, zero_indices=True):
    broadcastable = is_broadcastable(node_a.shape, node_b.shape)

    a_idx = []
    b_idx = []
    out_idx = []
    nmap = {}
    reverse = True

    if not broadcastable:
        reverse = False

        a_idx = [None] * len(node_a.shape)
        b_idx = [None] * len(node_b.shape)
        a_map = {}
        b_map = {}
        for s in node_c.shape:
            idx = pm.index(0, s-1)
            out_idx.append(idx)
            if s in node_a.shape:
                start = 0
                if s in a_map:
                    start = a_map[s]
                sidx = node_a.shape.index(s, start)
                a_idx[sidx] = idx
                a_map[s] = sidx

            if s in node_b.shape:
                start = 0
                if s in b_map:
                    start = b_map[s]
                sidx = node_b.shape.index(s, start)
                b_idx[sidx] = idx
                b_map[s] = sidx
                
        for i in range(len(a_idx)):
            if a_idx[i] is None:
                assert node_a.shape[i] == 1
                a_idx[i] = 0

        for i in range(len(b_idx)):
            if b_idx[i] is None:
                assert node_b.shape[i] == 1
                b_idx[i] = 0

    else:
        if node_a.shape == node_b.shape and node_c.shape == node_a.shape:
            indices = _get_single_node_indices(node_a)
            return indices, indices, indices
        elif node_a.shape == pm.DEFAULT_SHAPES[0] and node_b.shape == pm.DEFAULT_SHAPES[0]:
            idx = format_idx([])
            return idx, idx, idx
        elif node_a.shape == pm.DEFAULT_SHAPES[0]:
            idx = format_idx([])
            indices = _get_single_node_indices(node_b)
            return idx, indices, indices
        elif node_b.shape == pm.DEFAULT_SHAPES[0]:
            idx = format_idx([])
            indices = _get_single_node_indices(node_a)
            return indices, idx, indices

        if len(node_a.shape) > len(node_b.shape):
            small_node = node_b
            lg_node = node_a
            nmap["small"] = b_idx
            nmap["large"] = a_idx
        else:
            small_node = node_a
            lg_node = node_b
            nmap["small"] = a_idx
            nmap["large"] = b_idx


        for i in range(-1, -len(lg_node.shape) - 1, -1):
            if len(small_node.shape) < abs(i):
                idx = pm.index(0, lg_node.shape[i] - 1)
                nmap["large"].append(idx)
                out_idx.append(idx)
            elif node_a.shape[i] == node_b.shape[i]:
                if node_a.shape[i] != 1:
                    idx = pm.index(0, node_a.shape[i] - 1)
                    a_idx.append(idx)
                    b_idx.append(idx)
                    out_idx.append(idx)
            elif node_a.shape[i] == 1:
                idx = pm.index(0, node_b.shape[i] - 1)
                if zero_indices:
                    a_idx.append(0) # TESTING
                b_idx.append(idx)
                out_idx.append(idx)
            elif node_b.shape[i] == 1:
                idx = pm.index(0, node_a.shape[i] - 1)
                a_idx.append(idx)
                if zero_indices:
                    b_idx.append(0) # TESTING
                out_idx.append(idx)
            else:
                raise RuntimeError(f"Unable to broadcast indices:\n"
                                   f"{node_a.name}: {node_a.shape}\n"
                                   f"{node_b.name}: {node_b.shape}\n")
    return format_idx(a_idx, reverse), format_idx(b_idx, reverse), format_idx(out_idx, reverse)

def _get_binop_idx(node_a, node_b, out_node):
    # TODO: Figure out what to do about multiple dimensions with the same value
    cnt = 0
    op1 = []
    op2 = []
    all_ops = []

    for i in node_a.shape:
        if i == 1:
            op1.append(0)
            # all_ops.append(0)
        else:
            idx = pm.index(0, i - 1)
            op1.append(idx)
            all_ops.append(idx)
            cnt += 1

    for i in node_b.shape:
        if i in node_a.shape:
            idx = node_a.shape.index(i)
            op2.append(op1[idx])
        elif i == 1:
            op2.append(0)
            # all_ops.append(0)
        else:
            idx = pm.index(0, i - 1)
            op2.append(idx)
            all_ops.append(idx)
            cnt += 1
    if out_node.is_shape_finalized():
        all_ops = []
        for s in out_node.shape:
            if s in node_a.shape:
                idx = node_a.shape.index(s)
                all_ops.append(idx)
            else:
                assert s in node_b.shape, f"Output shape value {s} not in other shapes"
                idx = node_b.shape.index(s)
                all_ops.append(idx)

    return op1, op2, all_ops

def format_idx(x, reverse=True):
    if reverse:
        return tuple(list(reversed(x)))
    else:
        return tuple(x)

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

class global_avg_pool(pm.Template):
    def define_graph(self, x, out, shape=None, name=None, **kwargs):
        # indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in shape])
        indices = _get_single_node_indices(out, shape=shape)
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
    def define_graph(self, data, w, out, stride, pad):
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

class gemm(pm.Template):
    def define_graph(self, a, b, c, y,  shape=None, name=None, alpha=1.0, beta=0.0, transA=None, transB=None, **kwargs):
        i = pm.index(0, a.shape[0] - 1)
        j = pm.index(0, b.shape[0] - 1)
        k = pm.index(0, b.shape[1] - 1)
        y[i, k] = pm.sum([j], a[i, j]*b[j, k]) + c[i, k]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

class elem_gather(pm.Template):
    def define_graph(self, data, indices, output, axis=0, shape=None, name=None):
        # TODO: Fix this to use manual implementation
        output.write(pm.gather(data, indices, axis=axis))

class elem_expand(pm.Template):
    def define_graph(self, data, new_shape, output, axis=0, shape=None, name=None):
        # TODO: Fix this to use manual implementation
        in_dims = data.shape[0]
        new_dims = new_shape[0]
        update_shape_bool = in_dims < new_dims
        in_shape = in_dims * update_shape_bool + (1-update_shape_bool)

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class lrn(pm.Template):
    def define_graph(self, x, y, alpha, beta, bias, nsize, shape=None, name=None, **kwargs):
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
    def define_graph(self, data, out, kh, kw, stride, pad, **kwargs):

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

def get_transpose(data, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    transpose(data, out, shape=shape)
    return out
    # return transpose(data, out, shape=shape)

def get_elem_sigmoid(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return elem_sigmoid(x, out, shape=shape)
    elem_sigmoid(x, out, shape=shape)
    return out

def get_softmax(x, axis=1, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return softmax(x, out, axis=axis, shape=shape)
    softmax(x, out, axis=axis, shape=shape)
    return out

def get_elem_tanh(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return elem_tanh(x, out, shape=shape)
    elem_tanh(x, out, shape=shape)
    return out


# TODO: Need to convert this to a node with an output
def get_elem_cast(data, to=None, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    elem_cast(data, out, to, shape=shape)
    # return pm.cast(to, data[indices], name=name, shape=shape)
    return out

def get_elem_add(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return elem_add(a, b, out, shape=shape)
    elem_add(a, b, out, shape=shape)
    return out

def get_elem_sub(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    elem_sub(a, b, out, shape=shape)
    return out

def get_elem_greater(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    elem_greater(a, b, out, shape=shape)
    return out

def get_elem_mul(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    elem_mul(a, b, out, shape=shape)
    return out


def get_reduce_sum(x, shape=None, name=None, out=None, axes=(0,), **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    if isinstance(axes, Integral):
        axes = (axes,)
    elif isinstance(axes, list):
        axes = tuple(axes)
    else:
        assert isinstance(axes, tuple)
    reduce_sum(x, out, shape=shape, axes=axes, **kwargs)
    return out


def get_matmul(a, b, out=None, **kwargs):

    if not out:
        out = pm.output(shape=kwargs['shape'], name=kwargs['name'])
    matmul(a, b, out)
    return out

def get_elem(a, b, **kwargs):

    if len(a.shape) == len(b.shape):
        return matmul(a, b, **kwargs)
    elif len(a.shape) > len(b.shape):
        return rvmatmul(a, b, **kwargs)
    else:
        return lvmatmul(a, b, **kwargs)

def get_lrn(x, alpha=None, beta=None, bias=None, size=None, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.lrn(x, out, alpha=alpha, beta=beta, bias=bias, nsize=size)
    return out

# TODO: Add concat to transformations
def get_concat(*inputs, axis=None, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    indices = [pm.index(0, s - 1) if s > 1 else 0 for s in shape]
    for idx, i in enumerate(inputs):
        indices[axis] = pm.index(idx*i.shape[axis], (idx+1)*i.shape[axis]-1)
        j = pm.index(0, i.shape[axis]-1)
        out[tuple(indices)] = i[tuple(indices[:axis] + [j] + indices[axis+1:])]
    return out

def get_conv(x, w, bias=None, dilations=None, group=None, kernel_shape=None, pads=None, auto_pad=None,
             strides=None,
             shape=None,
             name=None,
             out=None):
    if not out:
        out = pm.output(shape=shape, name=name)

    if auto_pad:

        h_out = np.ceil(x.shape[-2] / strides[0])
        w_out = np.ceil(x.shape[-1] / strides[1])
        ph = max(0, (h_out - 1) * strides[0] + kernel_shape[0] - x.shape[-2])
        pw = max(0, (w_out - 1) * strides[1] + kernel_shape[1] - x.shape[-1])
        pads = [0,0,0,0]
        if auto_pad == "SAME_LOWER":
            pads[0] = np.floor(ph//2)
            pads[1] = ph - pads[0]
            pads[2] = np.floor(pw//2)
            pads[3] = pw - pads[2]
        elif auto_pad == "SAME_UPPER":
            pads[1] = np.floor(ph // 2)
            pads[0] = ph - pads[1]
            pads[3] = np.floor(pw // 2)
            pads[2] = pw - pads[3]

    if bias:
        pm.conv_bias(x, w, bias, out, int(strides[0]), int(pads[-2]))
        return out
    else:
        pm.conv(x, w, out, int(strides[0]), int(pads[-2]))
        return out

def get_batch_norm(x, s, b, mean, var, spatial=None, momentum=None,  epsilon=None, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.batch_norm(x, s, b, mean, var, out, epsilon, momentum, shape=shape)
    return out

def get_relu(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.relu(x, out)
    return out

def get_leaky_relu(x, alpha=0.01, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.leaky_relu(x, out, alpha=alpha)
    return out

def get_global_avg_pool(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.global_avg_pool(x, out, shape=shape)
    return out

def get_avg_pool(x, auto_pad=None, ceil_mode=0, kernel_shape=None, pads=None,
                 strides=None,
                 shape=None,
                 name=None,
                 out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    if auto_pad:
        if ceil_mode == 0:
            h_out = np.floor(x.shape[-2] / strides[0])
            w_out = np.floor(x.shape[-1] / strides[1])
        else:
            h_out = np.ceil(x.shape[-2] / strides[0])
            w_out = np.ceil(x.shape[-1] / strides[1])
        ph = max(0, (h_out - 1) * strides[0] + kernel_shape[0] - x.shape[-2])
        pw = max(0, (w_out - 1) * strides[1] + kernel_shape[1] - x.shape[-1])
        pads = [0, 0, 0, 0]
        if auto_pad == "SAME_LOWER":
            pads[0] = np.floor(ph // 2)
            pads[1] = ph - pads[0]
            pads[2] = np.floor(pw // 2)
            pads[3] = pw - pads[2]
        elif auto_pad == "SAME_UPPER":
            pads[1] = np.floor(ph // 2)
            pads[0] = ph - pads[1]
            pads[3] = np.floor(pw // 2)
            pads[2] = pw - pads[3]

    pm.avg_pool(x, out, kernel_shape[0], kernel_shape[1], (int(strides[0]), int(strides[1])),
                (int(pads[0]), int(pads[2])), shape=shape)
    return out

def get_max_pool(x, ceil_mode=0, kernel_shape=None, pads=None, auto_pad=None,
                 strides=None,
                 shape=None,
                 name=None,
                 out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    int_fn = np.ceil if ceil_mode != 0 else np.floor
    if auto_pad:
        h_out = int_fn(x.shape[-2] / strides[0])
        w_out = int_fn(x.shape[-1] / strides[1])
        ph = max(0, (h_out - 1) * strides[0] + kernel_shape[0] - x.shape[-2])
        pw = max(0, (w_out - 1) * strides[1] + kernel_shape[1] - x.shape[-1])
        pads = [0,0,0,0]
        if auto_pad == "SAME_LOWER":
            pads[0] = np.floor(ph//2)
            pads[1] = ph - pads[0]
            pads[2] = np.floor(pw//2)
            pads[3] = pw - pads[2]
        elif auto_pad == "SAME_UPPER":
            pads[1] = np.floor(ph // 2)
            pads[0] = ph - pads[1]
            pads[3] = np.floor(pw // 2)
            pads[2] = pw - pads[3]
    pm.max_pool(x, out, kernel_shape[0], kernel_shape[1], (int(strides[0]), int(strides[1])), (int(pads[0]),int(pads[2])), shape=shape)
    return out


def get_dropout(x, ratio=None, training_mode=False, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    if training_mode:
        pm.dropout(x, out, ratio=ratio, shape=shape)
    else:
        pm.dropout(x, out, shape=shape)
    return out


def get_flatten(x, axis=1, name=None, shape=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    coarse_flatten(x, out, axis=axis, shape=shape)
    return out

def get_gather(data, indices, axis=0, name=None, shape=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    elem_gather(data, indices, out, axis=axis, shape=shape)
    return out

def get_gemm(a, b , c=None, shape=None, name=None, alpha=None,
             beta=None,
             transA=None,
             transB=None,
             out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    if c:
        pm.gemm(a, b, c, out, shape=shape, alpha=alpha, beta=beta, transA=transA, transB=transB)
    else:
        t_c = pm.temp(shape=shape)
        i = pm.index(0, shape[0]-1)
        j = pm.index(0, shape[1]-1)
        t_c[i, j] = 0
        pm.gemm(a, b, t_c, out, shape=shape, alpha=alpha, beta=beta, transA=transA, transB=transB)
    return out


# TODO: Make range operation
def get_range(start, limit, delta, shape=None, name=None):
    value = np.arange(start, limit, delta)
    assert value.shape == shape
    y = pm.parameter(name=name, shape=shape, default=value)
    return y

# TODO: Fix this to be an actual operation
def get_shape(x, *args, name=None, shape=None, **kwargs):
    x.graph.nodes[name] = x.shape
    return x.graph.nodes[name]

# TODO: Fix this operation
def get_expand(input, shape_input, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    return out

# TODO: Add reshape operator, constant operator, gemm
NODE_NAMES = {"SVMClassifier": svm_classifier_train,
              "Conv": get_conv,
              "MatMul": get_matmul,
              "Concat": get_concat,
              "GlobalAveragePool": get_global_avg_pool,
              "AveragePool": get_avg_pool,
              "Flatten": get_flatten,
              "LRN": get_lrn,
              "Relu": get_relu,
              "LeakyRelu": get_leaky_relu,
              "BatchNormalization": get_batch_norm,
              "MaxPool": get_max_pool,
              "Gemm": get_gemm,
              "Dropout": get_dropout,
              "Mul": get_elem_mul,
              "Sub": get_elem_sub,
              "Add": get_elem_add,
              "Softmax": get_softmax,
              "Transpose": get_transpose,
              "Sigmoid": get_elem_sigmoid,
              "Tanh": get_elem_tanh,
              "Greater": get_elem_greater,
              "Shape": get_shape,
              "Gather": get_gather,
              "Range": get_range,
              "Expand": get_expand,
              "LinearRegressor": linear_regressor_train,
              "Cast": get_elem_cast,
              "Constant": pm.parameter,
              "Reshape": reshape,
              "Identity": identity,
              "ReduceSum": get_reduce_sum,
              "Unsqueeze": unsqueeze,
              "Squeeze": squeeze,
              "Resize": resize,
              }
