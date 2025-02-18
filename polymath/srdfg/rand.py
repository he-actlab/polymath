import numpy as np
from .index import index
from .domain import Domain
from .base import Node, nodeop, func_op, slice_op, var_index, call
from polymath import DEFAULT_SHAPES, UNSET_SHAPE

class Random(Node):

    def __init__(self, target, *args, **kwargs):
        super(Random, self).__init__(*args, target=f"{target.__module__}.{target.__name__}", **kwargs)
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
        return "<random '%s' target=%s>"% \
               (self.name, self.op_name)

class choice(Random):
    def __init__(self, input_node, **kwargs):
        super(choice, self).__init__(_choice, input_node, **kwargs)

def _choice(a, size=None, replace=True, p=None):
    return np.random.choice(a, size=size, replace=replace, p=p)