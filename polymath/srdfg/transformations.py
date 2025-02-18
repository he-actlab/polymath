import numpy as np
from .index import index
from .domain import Domain
from .base import Node, nodeop, func_op, slice_op, var_index, call
from polymath import DEFAULT_SHAPES, UNSET_SHAPE


class Transformation(Node):

    def __init__(self, target, *args, **kwargs):
        super(Transformation, self).__init__(*args, target=f"{target.__module__}.{target.__name__}", **kwargs)
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
        return "<transformation '%s' target=%s>"% \
               (self.name, self.op_name)

    def __getitem__(self, key):

        if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 0:
            return self
        elif self.is_shape_finalized() and len(self.nodes) > 0:
            if isinstance(key, (int, Node)):
                key = tuple([key])
            if len(key) != len(self.shape):
                raise KeyError(f"Invalid key shape for {self.name}:\n"
                               f"Shape: {self.shape}\n"
                               f"Key: {key}")

            name = f"{self.name}{key}"
            if name not in self.nodes.keys():
                raise KeyError(f"{name} not in {self.name} keys:\n"
                               f"Node keys: {list(self.nodes.keys())}")
            ret = self.nodes[name]
            return ret
        else:
            name = []
            if isinstance(key, Node):
                name.append(key.name)
            elif hasattr(key, "__len__") and not isinstance(key, str):
                for k in key:
                    if isinstance(k, Node):
                        name.append(k.name)
                    else:
                        name.append(str(k))
            else:
                name.append(key)
            name = self.var.name + "[" + "][".join(name) + "]"
            if name in self.graph.nodes:
                return self.graph.nodes[name]
            elif isinstance(key, (list)):
                return var_index(self, key, name=name, graph=self.graph)
            elif isinstance(key, tuple):
                return var_index(self, list(key), name=name, graph=self.graph)
            else:
                return var_index(self, [key], name=name, graph=self.graph)

class unsqueeze(Transformation):
    def __init__(self, input_node, axis=0, shape=None, **kwargs):
        if shape:
            new_shape = shape
        else:
            new_shape = list(input_node.shape)
            new_shape.insert(axis, 1)
        new_domain = Domain(tuple(new_shape))
        super(unsqueeze, self).__init__(_unsqueeze, input_node, domain=new_domain, axis=axis, shape=new_shape, **kwargs)

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

        if 'domain' in kwargs:
            new_domain = kwargs.pop('domain')
        else:
            new_domain = Domain(tuple(new_shape))
        super(squeeze, self).__init__(_squeeze, input_node, domain=new_domain, axis=axis, shape=new_shape, **kwargs)

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
        if 'domain' in kwargs:
            new_domain = kwargs.pop('domain')
        else:
            new_domain = Domain(tuple(new_shape))
        super(flatten, self).__init__(_flatten, input_node, domain=new_domain, axis=axis, shape=new_shape, **kwargs)

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

class gather(Transformation):
    def __init__(self, input_node, indices, axis=1, shape=None, **kwargs):
        if shape:
            new_shape = shape
        elif len(input_node.shape) == 1 and input_node.shape[0] == 1:
            new_shape = list(input_node.shape)
        else:
            new_shape = list(input_node.shape)
            new_shape.pop(axis)
            curr_axis = axis
            for i in indices.shape:
                new_shape.insert(curr_axis, i)
                curr_axis += 1
        if 'domain' in kwargs:
            new_domain = kwargs.pop('domain')
        else:
            new_domain = Domain(tuple(new_shape))
        super(gather, self).__init__(_gather, input_node, indices, domain=new_domain, axis=axis, shape=new_shape, **kwargs)

    def compute_shape(self):
        assert all([not isinstance(s, Node) for s in self.args[0].shape])
        assert all([not isinstance(s, Node) for s in self.args[1].shape])
        new_shape = list(self.args[0].shape)
        new_shape[self.axis].pop()
        curr_axis = self.axis
        for i in self.args[1].shape:
            new_shape[curr_axis].insert(curr_axis, i)
            curr_axis += 1
        return tuple(new_shape)

    def _evaluate(self, val, indices, **kwargs):
        if "target" in kwargs:
            kwargs.pop("target")
        if "domain" in kwargs:
            kwargs.pop("domain")

        val = self.target(val, indices, self.axis)

        if not self.is_shape_finalized():
            self.shape = val.shape
        return val

    @property
    def axis(self):
        return self.kwargs['axis']


class gather_elements(Transformation):
    def __init__(self, input_node, indices, axis=1, shape=None, **kwargs):
        if shape:
            new_shape = shape
        else:
            new_shape = list(input_node.shape)
            new_shape.pop(axis)
            curr_axis = axis
            for i in indices.shape:
                new_shape.insert(curr_axis, i)
                curr_axis += 1
        if 'domain' in kwargs:
            new_domain = kwargs.pop('domain')
        else:
            new_domain = Domain(tuple(new_shape))
        super(gather_elements, self).__init__(_gather_elements, input_node, indices, domain=new_domain, axis=axis, shape=new_shape, **kwargs)

    def compute_shape(self):
        assert all([not isinstance(s, Node) for s in self.args[0].shape])
        assert all([not isinstance(s, Node) for s in self.args[1].shape])
        new_shape = list(self.args[0].shape)
        new_shape[self.axis].pop()
        curr_axis = self.axis
        for i in self.args[1].shape:
            new_shape[curr_axis].insert(curr_axis, i)
            curr_axis += 1
        return tuple(new_shape)

    def _evaluate(self, val, indices, **kwargs):
        if "target" in kwargs:
            kwargs.pop("target")
        if "domain" in kwargs:
            kwargs.pop("domain")

        val = self.target(val, indices, self.axis)

        if not self.is_shape_finalized():
            self.shape = val.shape
        return val

    @property
    def axis(self):
        return self.kwargs['axis']

class reshape(Transformation):
    def __init__(self, input_node, new_shape, **kwargs):

        assert new_shape is not None
        if 'domain' in kwargs:
            new_domain = kwargs.pop('domain')
        else:
            new_domain = Domain(tuple(new_shape))
        super(reshape, self).__init__(_reshape, input_node, new_shape, domain=new_domain, shape=new_shape, **kwargs)

    def _evaluate(self, val, new_shape, **kwargs):
        if "target" in kwargs:
            kwargs.pop("target")
        if "domain" in kwargs:
            kwargs.pop("domain")

        val = self.target(val, self.shape)
        return val

class transpose(Transformation):
    def __init__(self, input_node, axis, **kwargs):

        assert isinstance(axis, (tuple, list, np.ndarray))
        if len(input_node.shape) > 1:
            new_shape = tuple([input_node.shape[i] for i in axis])
        else:
            new_shape = input_node.shape

        if 'domain' in kwargs:
            new_domain = kwargs.pop('domain')
        else:
            new_domain = Domain(new_shape)

        super(transpose, self).__init__(_transpose, input_node, axis, domain=new_domain, shape=new_shape, **kwargs)

    def _evaluate(self, val, axis, **kwargs):
        if "target" in kwargs:
            kwargs.pop("target")
        if "domain" in kwargs:
            kwargs.pop("domain")

        val = self.target(val, axis)
        return val

    @property
    def axis(self):
        return self.args[1]


class pad(Transformation):
    def __init__(self, input_node, pad_start, pad_end=None, **kwargs):
        n = len(input_node.shape)
        if pad_end is None:
            padding_val = tuple((pad_start[i], pad_start[i]) for i in range(len(pad_start)))
        else:
            padding_val = tuple((pad_start[i], pad_end[i]) for i in range(len(pad_start)))

        pad_end = pad_end if pad_end else pad_start
        new_shape = tuple([input_node.shape[i] + pad_start[i] + pad_end[i] for i in range(n)])

        if 'domain' in kwargs:
            new_domain = kwargs.pop('domain')
        else:
            new_domain = Domain(tuple(new_shape))

        super(pad, self).__init__(_pad, input_node, padding_val, domain=new_domain, shape=new_shape, **kwargs)

    def _evaluate(self, val, padding_val, **kwargs):
        if "target" in kwargs:
            kwargs.pop("target")
        if "domain" in kwargs:
            kwargs.pop("domain")

        val = self.target(val, self.padding_val)
        return val

    @property
    def padding_val(self):
        return self.args[1]

class flip(Transformation):
    def __init__(self, input_node, axis, **kwargs):


        if 'domain' in kwargs:
            new_domain = kwargs.pop('domain')
        else:
            new_domain = Domain(tuple(input_node.shape))

        super(flip, self).__init__(_flip, input_node, axis, domain=new_domain, shape=input_node.shape, **kwargs)

    def _evaluate(self, val, axis, **kwargs):
        if "target" in kwargs:
            kwargs.pop("target")
        if "domain" in kwargs:
            kwargs.pop("domain")

        val = self.target(val, self.axis)
        return val

    @property
    def axis(self):
        return self.args[1]

def _gather(value, indices, axis=0):
    return np.take(value, indices, axis=axis)

def _gather_elements(value, indices, axis=0):
    return np.take_along_axis(value, indices, axis=axis)

def _reshape(value, shape):
    return np.reshape(value, shape)

def _unsqueeze(value, axis=0):
    return np.expand_dims(value, axis=axis)

def _squeeze(value, axis=0):
    return np.squeeze(value, axis=axis)

def _flatten(value, axis=1):
    new_shape = (1, -1) if axis == 0 else (np.prod(value.shape[0:axis]).astype(int), -1)
    return np.reshape(value, new_shape)

def _transpose(value, axis=None):
    return np.transpose(value, axis)

def _pad(value, pad_width):
    return np.pad(value, pad_width, mode="constant")

def _flip(value, axis):
    return np.flip(value, axis=axis)