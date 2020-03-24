from polymath.mgdfg.base import *
import numpy as np
from polymath.mgdfg.util import is_iterable
import builtins
from itertools import product
import operator

class index(Node):  # pylint: disable=C0103,W0223
    """
    Return a slice of a variable.

    .. note::.
    """
    def __init__(self, lbound, ubound, **kwargs):  # pylint: disable=W0235
        kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs else "index"
        super(index, self).__init__(lbound, ubound, **kwargs)

    @property
    def domain(self):
        return tuple([self])

    @property
    def dom_names(self):
        return tuple([self.name])

    @property
    def lbound(self):
        l, _ = self.args
        return l

    @property
    def ubound(self):
        _, u = self.args
        return u

    def _evaluate(self, lbound, ubound, **kwargs):
        value = np.asarray([i for i in range(int(lbound), int(ubound) + 1)])
        return value

    def set_scalar_subgraph(self, _):
        return

    def __add__(self, other):
        return index_op(operator.add, self, other, graph=self.graph)

    def __radd__(self, other):
        return index_op(operator.add, other, self, graph=self.graph)

    def __sub__(self, other):
        return index_op(operator.sub, self, other, graph=self.graph)

    def __rsub__(self, other):
        return index_op(operator.sub, other, self, graph=self.graph)

    def __mul__(self, other):
        return index_op(operator.mul, self, other, graph=self.graph)

    def __rmul__(self, other):
        return index_op(operator.mul, other, self, graph=self.graph)

    def __truediv__(self, other):
        return index_op(operator.floordiv, self, other, graph=self.graph)

    def __rtruediv__(self, other):
        return index_op(operator.floordiv, other, self, graph=self.graph)

    def __floordiv__(self, other):
        return index_op(operator.floordiv, self, other, graph=self.graph)

    def __rfloordiv__(self, other):
        return index_op(operator.floordiv, other, self, graph=self.graph)

    def __repr__(self):
        return "<index '%s'>" % (self.name)

class index_op(index):

    def __init__(self, target, *args, **kwargs):  # pylint: disable=W0235
        kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs \
            else f"index_{target.__name__}"
        if "domain" in kwargs:
            domain = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
        else:
            domain = []
            for a in args:
                if isinstance(a, index):
                    domain += [i for i in a.domain]
            domain = tuple(domain)
        super(index_op, self).__init__(*args, target=f"{target.__module__}.{target.__name__}", domain=domain, **kwargs)
        self.target = target

    # #TODO: Fix domain and bounds
    @property
    def domain(self):
        return self.kwargs["domain"]

    @property
    def dom_names(self):
        names = []
        for a in self.args:
            if isinstance(a, index):
                names += [i.name if isinstance(i, index) else i for i in a.domain]
        return tuple(names)

    @property
    def lbound(self):
        return self.args

    @property
    def ubound(self):
        return self.args

    def _evaluate(self, op1, op2, **kwargs):

        if is_iterable(op1) and is_iterable(op2):
            assert isinstance(self.args[0], index) or np.allclose(self.args[0], op1)
            assert isinstance(self.args[1], index) or np.allclose(self.args[1], op2)
            combined_indices = list(product(*(op1, op2)))
            value = np.array(list(map(lambda x: self.target(x[0], x[1]), combined_indices)))
        else:
            value = self.target(op1, op2)
        return value