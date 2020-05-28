from polymath.mgdfg.base import *
import numpy as np
from numbers import Integral, Real
import functools
from polymath.mgdfg.nodes import parameter
from .util import _flatten_iterable, _fnc_hash

class NonLinear(Node):
    """
        Node wrapper for stateless functions.

        Parameters
        ----------
        target : callable
            function to evaluate the node
        args : tuple
            positional arguments passed to the target
        kwargs : dict
            keywoard arguments passed to the target
        """

    def __init__(self, target, val, **kwargs):
        kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs \
            else f"{self.__class__.__name__}"
        if "domain" in kwargs:
            kwargs.pop("domain")
        domain = val.domain if isinstance(val, Node) else Domain((1,))
        kwargs['shape'] = (1,) if isinstance(val, Real) else val.shape
        super(NonLinear, self).__init__(val, target=f"{target.__module__}.{target.__name__}", domain=domain, **kwargs)
        self.target = target

    def __getitem__(self, key):

        if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 0:
            return self
        elif self.is_shape_finalized() and len(self.nodes) > 0:
            if isinstance(key, (int, Node)):
                key = tuple([key])
            if len(key) != len(self.shape):
                raise KeyError(f"Cannot access item with key {key} for node with shape {self.shape}\n"
                               f"\tNode: {self.name}\n\t"
                               f"Op: {self.op_name}\n\t")
            name = f"{self.name}{key}"
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
            name = str(tuple(name)).replace("'", "")
            name = f"{self.var.name}{name}"
            if name in self.graph.nodes:
                return self.graph.nodes[name]
            elif isinstance(key, (list)):
                return var_index(self, key, name=name, graph=self.graph)
            elif isinstance(key, tuple):
                return var_index(self, list(key), name=name, graph=self.graph)
            else:
                return var_index(self, [key], name=name, graph=self.graph)

    def _evaluate(self, val, **kwargs):
        if "target" in kwargs:
            kwargs.pop("target")
        if "domain" in kwargs:
            kwargs.pop("domain")
        val = self.target(val)
        if isinstance(val, Real) or len(val.shape) == 0:
            val = np.asarray([val])

        if not self.is_shape_finalized():
            self.shape = val.shape
        return val

    @property
    def domain(self):
        return self.kwargs["domain"]

    def __call__(self, val, **kwargs):
        return call(self, val, **kwargs)

    def __repr__(self):
        return "<nonlinear '%s' target=%s>"% \
               (self.name, self.op_name)


class sigmoid(NonLinear):
    def __init__(self, input_node, **kwargs):
        super(sigmoid, self).__init__(_sigmoid, input_node, **kwargs)


class log2(NonLinear):
    def __init__(self, input_node, **kwargs):
        super(log2, self).__init__(_log2, input_node, **kwargs)

class exp(NonLinear):
    def __init__(self, input_node, **kwargs):
        super(exp, self).__init__(_exp, input_node, **kwargs)


class abs(NonLinear):
    def __init__(self, input_node, **kwargs):
        super(abs, self).__init__(_abs, input_node, **kwargs)

class floor(NonLinear):
    def __init__(self, input_node, **kwargs):
        super(floor, self).__init__(_floor, input_node, **kwargs)

class ceil(NonLinear):
    def __init__(self, input_node, **kwargs):
        super(ceil, self).__init__(_ceil, input_node, **kwargs)

class sqrt(NonLinear):
    def __init__(self, input_node, **kwargs):
        super(sqrt, self).__init__(_sqrt, input_node, **kwargs)

# class cast(NonLinear):
#     def __init__(self, np_dtype, input_node, **kwargs):
#         super(cast, self).__init__(lambda x: _cast(x, np_dtype), input_node, **kwargs)

def _log2(value):
    return np.log2(value)

def _sigmoid(value):
    return 1 / (1 + np.exp(-value))

def _exp(value):
    return np.exp(value)

def _abs(value):
    return np.abs(value)

def _sqrt(value):
    return np.sqrt(value)

def _floor(value):
    return int(np.floor(value))

def _ceil(value):
    return int(np.ceil(value))

def _cast(value, npdtype):
    if not isinstance(value, np.ndarray):
        return np.asarray(value).astype(npdtype)
    else:
        return value.astype(npdtype)





