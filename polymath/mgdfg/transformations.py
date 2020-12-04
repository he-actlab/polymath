import numpy as np
from .index import index
from .domain import Domain
from .base import Node, nodeop, func_op, slice_op, var_index, call
from polymath import DEFAULT_SHAPES, UNSET_SHAPE


class Transformation(Node):

    def __init__(self, target, node, domain, **kwargs):
        super(Transformation, self).__init__(node, target=f"{target.__module__}.{target.__name__}", domain=domain, **kwargs)
        self.target = target

    def _evaluate(self, val, **kwargs):
        if "target" in kwargs:
            kwargs.pop("target")
        if "domain" in kwargs:
            kwargs.pop("domain")
        val = self.target(val, **kwargs)

        if not self.is_shape_finalized():
            self.shape = val.shape
        return val

    @property
    def domain(self):
        return self.kwargs["domain"]

    def compute_shape(self):
        raise NotImplemented

    def __call__(self, val, **kwargs):
        return call(self, val, **kwargs)

    def __repr__(self):
        return "<nonlinear '%s' target=%s>"% \
               (self.name, self.op_name)

class unsqueeze(Transformation):
    def __init__(self, input_node, axis=0, shape=None, **kwargs):
        if shape:
            new_shape = shape
        else:
            new_shape = list(input_node.shape)
            new_shape.insert(axis, 1)
        new_domain = Domain(tuple(new_shape))
        super(unsqueeze, self).__init__(_unsqueeze, input_node, new_domain, axis=axis, shape=new_shape, **kwargs)

    def compute_shape(self):
        assert all([not isinstance(s, Node) for s in self.args[0].shape])
        new_shape = list(self.args[0].shape)
        new_shape.insert(self.axis, 1)
        return tuple(new_shape)

    @property
    def axis(self):
        return self.kwargs['axis']

class squeeze(Transformation):
    def __init__(self, input_node, axis=None, shape=None, **kwargs):
        new_shape = list(input_node.shape)
        if shape:
            new_shape = shape
        elif axis:
            if isinstance(axis, int):
                axis = (axis,)
            for a in axis:
                if isinstance(new_shape[a], int):
                    assert new_shape[a] == 1
                new_shape.pop(a)
        else:
            for i, a in enumerate(new_shape):
                if a == 1:
                    new_shape.pop(i)
        new_domain = Domain(tuple(new_shape))
        super(squeeze, self).__init__(_squeeze, input_node, new_domain, axis=axis, shape=new_shape, **kwargs)

    def compute_shape(self):
        assert all([not isinstance(s, Node) for s in self.args[0].shape])
        new_shape = list(self.args[0].shape)
        if self.axis:

            for a in self.axis:
                if isinstance(new_shape[a], int):
                    assert new_shape[a] == 1
                new_shape.pop(a)
        else:
            for i, a in enumerate(new_shape):
                if a == 1:
                    new_shape.pop(i)
        return tuple(new_shape)

    @property
    def axis(self):
        return self.kwargs['axis']

class flatten(Transformation):
    def __init__(self, input_node, axis=1, shape=None, **kwargs):
        if shape:
            new_shape = shape
        elif axis == 0:
            size = 1
            for i in input_node.shape:
                size = size*i
            new_shape = (1, size)
        else:
            dim0 = 1
            for i in input_node.shape[0:axis]:
                dim0 = dim0*i
            dim1 = 1
            for i in input_node.shape[axis:]:
                dim1 = dim1*i
            new_shape = (dim0, dim1)
        new_domain = Domain(tuple(new_shape))
        super(flatten, self).__init__(_flatten, input_node, new_domain, axis=axis, shape=new_shape, **kwargs)

    def compute_shape(self):
        assert all([not isinstance(s, Node) for s in self.args[0].shape])
        if self.axis == 0:
            size = 1
            for i in self.args[0].shape:
                size = size * i
            new_shape = (1, size)
        else:
            dim0 = 1
            for i in self.args[0].shape[0:self.axis]:
                dim0 = dim0 * i
            dim1 = 1
            for i in self.args[0].shape[self.axis:]:
                dim1 = dim1 * i
            new_shape = (dim0, dim1)
        return tuple(new_shape)

    @property
    def axis(self):
        return self.kwargs['axis']

def _unsqueeze(value, axis=0):
    return np.expand_dims(value, axis=axis)

def _squeeze(value, axis=0):
    return np.squeeze(value, axis=axis)

def _flatten(value, axis=1):
    new_shape = (1, -1) if axis == 0 else (np.prod(value.shape[0:axis]).astype(int), -1)
    return np.reshape(value, new_shape)


