import polymath as pm
from .template_utils import _get_single_node_indices
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools

class tensor_transpose(pm.Template):
    def define_graph(self, data, out, perm=None):

        temp = pm.transpose(data, perm)
        indices = _get_single_node_indices(temp)
        out[indices] = temp[indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def perm(self):
        return self.kwargs["perm"] or tuple(reversed(range(len(self.args[0].shape))))

class tensor_flip(pm.Template):
    def define_graph(self, data, out, axis=None):
        temp = pm.flip(data, axis)
        indices = _get_single_node_indices(temp)
        out[indices] = temp[indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)


class tensor_reshape(pm.Template):
    def define_graph(self, data, out, new_shape):
        temp = pm.reshape(data, new_shape)
        indices = _get_single_node_indices(temp)
        out[indices] = temp[indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class tensor_pad(pm.Template):
    def define_graph(self, data, out, pad_start, pad_end=None):
        assert isinstance(pad_start, (list, tuple)) and len(pad_start) >= 1
        if isinstance(pad_start[0], (list, tuple)):
            assert pad_end is None
            pad_end = tuple([pad_start[i][1] for i in range(len(pad_start))])
            pad_start = tuple([pad_start[i][0] for i in range(len(pad_start))])

        temp = pm.pad(data, pad_start, pad_end=pad_end)
        indices = _get_single_node_indices(temp)
        out.set_shape(temp.shape, override=True)
        out[indices] = temp[indices]

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


class tensor_squeeze(pm.Template):
    def define_graph(self, data, out):
        pass

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)


class elem_gather(pm.Template):
    def define_graph(self, data, output, indices=None, axis=0):
        # TODO: Fix this to use manual implementation
        assert indices is not None
        output.write(pm.gather(data, np.asarray([indices]), axis=axis))

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def indices(self):
        return self.kwargs['indices']



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

class resize(pm.Template):
    def define_graph(self, data, scales, output, mode=0):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

def pad_tensor_inlined(data, out, pad_start, pad_end=None):
    assert isinstance(pad_start, (list, tuple)) and len(pad_start) >= 1
    if isinstance(pad_start[0], (list, tuple)):
        assert pad_end is None
        pad_end = tuple([pad_start[i][1] for i in range(len(pad_start))])
        pad_start = tuple([pad_start[i][0] for i in range(len(pad_start))])

    temp = pm.pad(data, pad_start, pad_end=pad_end)
    indices = _get_single_node_indices(temp)
    out.set_shape(temp.shape, override=True)
    out[indices] = temp[indices]

def flip_tensor_inlined(data, out, axis=None):
    temp = pm.flip(data, axis)
    indices = _get_single_node_indices(temp)
    out[indices] = temp[indices]

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