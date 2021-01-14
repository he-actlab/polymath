import polymath as pm
from .template_utils import _get_single_node_indices
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools

class transpose(pm.Template):
    def define_graph(self, data, out):
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

class coarse_flatten(pm.Template):
    def define_graph(self, data, out, axis=1):
        o_indices = _get_single_node_indices(out, shape=out.shape)
        i_indices = _get_single_node_indices(data, shape=out.shape)
        out[o_indices] = data[i_indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)


class elem_gather(pm.Template):
    def define_graph(self, data, indices, output, axis=0):
        # TODO: Fix this to use manual implementation
        output.write(pm.gather(data, indices, axis=axis))

class elem_expand(pm.Template):
    def define_graph(self, data, new_shape, output, axis=0):
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

# TODO: Need to fix this functionality to create a new node
def onnx_unsqueeze(x, *args, axes=None, shape=None, name=None, **kwargs):
    out = pm.unsqueeze(x, axis=axes, name=name, shape=shape)
    return out

# TODO: Check this works after changes
def onnx_squeeze(x, *args, axes=None, shape=None, name=None, **kwargs):
    out = pm.squeeze(x, axis=axes, name=name, shape=shape)
    return out

# TODO: Check this works after changes
def onnx_reshape(data, *args, shape=None, name=None, **kwargs):
    data._shape = shape
    data.graph.nodes[name] = data
    return data

# TODO: Convert this to a template node
def onnx_resize(data, *args, shape=None, name=None, **kwargs):

    data._shape = shape
    data.graph.nodes[name] = data
    return data

def onnx_identity(data, shape=None, name=None, **kwargs):
    data.set_name(name)
    return data