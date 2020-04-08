from polymath.mgdfg.base import *
import numpy as np
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
        domain = val.domain if isinstance(val, Node) else Domain((1,))
        kwargs['shape'] = val.shape
        super(NonLinear, self).__init__(val, target=f"{target.__module__}.{target.__name__}", domain=domain, **kwargs)
        self.target = target

    def __getitem__(self, key):
        if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 0:
            return self
        elif self.is_shape_finalized() or len(self.nodes) > 0:
            if isinstance(key, (int, Node)):
                key = tuple([key])
            assert len(key) == len(self.shape)
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

            name = self.var.name + "[" + "][".join(name) + "]"
            if isinstance(key, (list)):
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
        return self.target(val)

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

def _sigmoid(value):
    return 1 / (1 + np.exp(-value))