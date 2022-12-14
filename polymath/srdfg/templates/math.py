import polymath as pm
from .template_utils import _get_single_node_indices, _get_elem_indices
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools


class reduce_sum(pm.Template):
    def define_graph(self, data, out, axes=(0,), keepdims=True):
        # indices = _get_single_node_indices(data)
        indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axes])
        out_idx = tuple([indices[i] for i in range(len(indices)) if i not in axes])
        out[out_idx] = pm.sum([sum_idx], data[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class reduce_prod(pm.Template):
    def define_graph(self, data, out, axes=(0,), keepdims=True):
        indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axes])
        out_idx = tuple([indices[i] for i in range(len(indices)) if i not in axes])
        out[out_idx] = pm.prod([sum_idx], data[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class reduce_min(pm.Template):
    def define_graph(self, data, out, axes=(0,), keepdims=True):
        # indices = _get_single_node_indices(data)
        indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axes])
        out_idx = tuple([indices[i] for i in range(len(indices)) if i not in axes])
        out[out_idx] = pm.min([sum_idx], data[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def axis(self):
        return self.kwargs['axes']

    @property
    def axes(self):
        return self.kwargs['axes']

class reduce_max(pm.Template):
    def define_graph(self, data, out, axes=(0,), keepdims=True):
        # indices = _get_single_node_indices(data)
        indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axes])
        out_idx = tuple([indices[i] for i in range(len(indices)) if i not in axes])
        out[out_idx] = pm.max([sum_idx], data[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class reduce_mean(pm.Template):
    def define_graph(self, data, out, axes=(0,), keepdims=True):
        # indices = _get_single_node_indices(data)
        indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axes])
        out_idx = tuple([indices[i] for i in range(len(indices)) if i not in axes])
        denom = 1
        for i in axes:
            denom *= data.shape[i]
        out[out_idx] = pm.sum([sum_idx], data[indices]) / (denom)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def axis(self):
        return self.kwargs['axes']

    @property
    def axes(self):
        return self.kwargs['axes']

class elem_greater(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # a_idx, b_idx, indices = _get_binop_idx(a, b, out)

        out[indices] = (a[a_idx] > b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_less(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)

        out[indices] = (a[a_idx] < b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_not(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.logical_not(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_nonzero(pm.Template):
    def define_graph(self, x, out):
        pass
        # indices = _get_single_node_indices(out, shape=out.shape)
        # out[indices] = ((x[indices] != 0) * indices)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)


class reciprocal(pm.Template):
    def define_graph(self, x, out):
        pass
        # indices = _get_single_node_indices(out, shape=out.shape)
        # out[indices] = ((x[indices] != 0) * indices)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_or(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        out[indices] = (a[a_idx] or b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_and(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        out[indices] = (a[a_idx] and b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_equal(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # out[indices] = (a[a_idx] == b[b_idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_min(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        out[indices] = (a[a_idx] > b[b_idx]) * b[a_idx] + (a[a_idx] <= b[b_idx]) * a[a_idx]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_max(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        out[indices] = (a[a_idx] > b[b_idx]) * a[a_idx] + (a[a_idx] <= b[b_idx]) * b[a_idx]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_sub(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # a_idx, b_idx, indices = _get_binop_idx(a, b, out)
        out[indices] = (a[a_idx] - b[b_idx])

    @property
    def inputs(self):
        if "const" in self.op_name:
            if isinstance(self.args[0], pm.parameter):
                return (self.args[1],)
            else:
                return (self.args[0],)
        else:
            return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_add(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # a_idx, b_idx, indices = _get_binop_idx(a, b, out)
        out[indices] = (a[a_idx] + b[b_idx])

    @property
    def inputs(self):
        if "const" in self.op_name:
            if isinstance(self.args[0], pm.parameter):
                return (self.args[1],)
            else:
                return (self.args[0],)
        else:
            return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_mul(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        out[indices] = (a[a_idx] * b[b_idx])

    @property
    def inputs(self):
        if "const" in self.op_name:
            if isinstance(self.args[0], pm.parameter):
                return (self.args[1],)
            else:
                return (self.args[0],)
        else:
            return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_div(pm.Template):
    def define_graph(self, a, b, out):
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        out[indices] = (a[a_idx] / b[b_idx])

    @property
    def inputs(self):
        if "const" in self.op_name:
            if isinstance(self.args[0], pm.parameter):
                return (self.args[1],)
            else:
                return (self.args[0],)
        else:
            return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_pow(pm.Template):
    def define_graph(self, val, out, exp=None):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = (val[indices] ** exp)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def exp(self):
        return self.kwargs['exp']

    @property
    def outputs(self):
        return (self.args[1],)

class matmul(pm.Template):
    def define_graph(self, a, w, out):
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

class mean_var(pm.Template):
    def define_graph(self, data, mean, var, axis=None):
        indices = tuple([pm.index(0, s-1) for s in data.shape])
        if axis is None:
            axis = tuple(list(range(len(data.shape))))
        r_idx = [indices[i] for i in axis]
        o_idx = tuple([indices[i] for i in range(len(data.shape)) if i not in axis])

        denom = 1
        for i in axis:
            denom *= data.shape[i]

        mean[o_idx] = pm.sum(r_idx, data[indices]) / (denom)
        var[o_idx] = pm.sum(r_idx, pm.square(data[indices] - mean[o_idx])) / (denom)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1], self.args[2])

def lvmatmul(a, b, shape=None, name=None, **kwargs):
    i = pm.index(0, a.shape[0] - 1)
    j = pm.index(0, b.shape[1] - 1)
    return pm.sum([i], a[i]*b[i, j], name=name)

def rvmatmul(a, b, shape=None, name=None, **kwargs):
    i = pm.index(0, a.shape[0] - 1)
    j = pm.index(0, b.shape[0] - 1)
    return pm.sum([j], a[i, j]*b[j], name=name)

class gemm(pm.Template):
    def define_graph(self, a, b, c, y, alpha=1.0, beta=0.0, transA=None, transB=None, strict_shapes=False):
        if strict_shapes:
            assert b.shape[0] == a.shape[1]
            assert len(y.shape) == 0 or y.shape[0] == a.shape[0]
            assert c.shape[0] == b.shape[1]
            assert bool(transB) == bool(transA) and bool(transA) == False, f"Strict shape check failed: {transA} != {transB}"

        if transA:
            i = pm.index(0, a.shape[1] - 1)
            j = pm.index(0, b.shape[0] - 1)
            k = pm.index(0, b.shape[1] - 1)
            y[i, k] = pm.sum([j], a[j, i]*b[j, k]) + c[i, k]
        elif transB:
            i = pm.index(0, a.shape[0] - 1)
            j = pm.index(0, b.shape[1] - 1)
            k = pm.index(0, b.shape[0] - 1)
            y[i, k] = pm.sum([j], a[i, j]*b[k, j]) + c[i, k]
        else:
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


class gemm_no_bias(pm.Template):
    def define_graph(self, a, b, y, alpha=1.0, beta=0.0, transA=False, transB=False, strict_shapes=False):
        if strict_shapes:
            assert b.shape[0] == a.shape[1]
            assert len(y.shape) == 0 or y.shape[0] == a.shape[0]
            assert bool(transB) == bool(transA) and bool(transA) == False, f"Strict shape check failed: {transA} != {transB}"

        if transA:
            i = pm.index(0, a.shape[1] - 1)
            j = pm.index(0, b.shape[0] - 1)
            k = pm.index(0, b.shape[1] - 1)
            y[i, k] = pm.sum([j], a[j, i]*b[j, k])
        elif transB:
            i = pm.index(0, a.shape[0] - 1)
            j = pm.index(0, b.shape[1] - 1)
            k = pm.index(0, b.shape[0] - 1)
            y[i, k] = pm.sum([j], a[i, j]*b[k, j])
        else:
            i = pm.index(0, a.shape[0] - 1)
            j = pm.index(0, b.shape[0] - 1)
            k = pm.index(0, b.shape[1] - 1)
            y[i, k] = pm.sum([j], a[i, j]*b[j, k])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)