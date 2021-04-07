from .base import Node
import numpy as np
from .util import is_iterable
from polymath.srdfg.domain import Domain
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
        if not "domain" in kwargs or len(kwargs['domain'].doms) == 0:
            domain = Domain(tuple([self]), dom_set=tuple([self]))
            self.kwargs['domain'] = domain

        if 'stride' not in kwargs:
            self.kwargs['stride'] = 1


    def as_shape(self):
        return self.ubound - self.lbound + 1

    @property
    def domain(self):
        return self.kwargs["domain"]

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

    @property
    def stride(self):
        return self.kwargs['stride']

    def _evaluate(self, lbound, ubound, **kwargs):
        if isinstance(self.lbound, index) or is_iterable(self.lbound):
            lbound = len(lbound) - 1

        if isinstance(self.ubound, index) or is_iterable(self.ubound):
            ubound = len(ubound) - 1
        value = np.asarray([i for i in range(int(lbound), int(ubound) + 1, self.stride)])
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

    def __rshift__(self, other):
        return index_op(operator.rshift, self, other, graph=self.graph)

    def __rrshift__(self, other):

        return index_op(operator.rshift, other, self, graph=self.graph)

    def __lshift__(self, other):
        return index_op(operator.lshift, self, other, graph=self.graph)

    def __rlshift__(self, other):
        return index_op(operator.lshift, other, self, graph=self.graph)

    def __truediv__(self, other):
        return index_op(operator.floordiv, self, other, graph=self.graph)

    def __rtruediv__(self, other):
        return index_op(operator.floordiv, other, self, graph=self.graph)

    def __floordiv__(self, other):
        return index_op(operator.floordiv, self, other, graph=self.graph)

    def __rfloordiv__(self, other):
        return index_op(operator.floordiv, other, self, graph=self.graph)

    def __le__(self, other):
        return index_op(operator.le, self, other, graph=self.graph)

    def __ge__(self, other):
        return index_op(operator.ge, self, other, graph=self.graph)

    def __lt__(self, other):
        return index_op(operator.lt, self, other, graph=self.graph)

    def __gt__(self, other):
        return index_op(operator.gt, self, other, graph=self.graph)

    def __repr__(self):
        return "<index '%s'>" % (self.name)

class index_op(index):

    def __init__(self, target, *args, **kwargs):  # pylint: disable=W0235
        kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs \
            else f"index_{target.__name__}"

        super(index_op, self).__init__(*args, target=f"{target.__module__}.{target.__name__}", **kwargs)
        if "domain" in kwargs:
            domain = Domain(tuple(self.kwargs.pop("domain"))) if isinstance(self.kwargs["domain"], list) else self.kwargs.pop("domain")
            self.kwargs['domain'] = domain
        else:
            dset = []
            for a in args:
                if isinstance(a, index):
                    dset += [i for i in a.domain.dom_set if i not in dset]
            self.kwargs['domain'] = Domain(tuple([self]), dom_set=tuple(dset))
        self.op1_dom = self.args[0].domain if isinstance(self.args[0], Node) else Domain(self.args[0])
        self.op2_dom = self.args[1].domain if isinstance(self.args[1], Node) else Domain(self.args[1])
        self.target = target

    def as_shape(self):
        return 0

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
        if is_iterable(op1) and is_iterable(op2) and all([i.shape != (1,) for i in [op1,op2]]):
            if self.op1_dom.names != self.op2_dom.names:
                pairs = list(product(*(op1, op2)))
            else:
                pairs = list(zip(*(op1, op2)))
            _ = self.domain.compute_index_pairs()
            assert isinstance(self.args[0], index) or np.allclose(self.args[0], op1)
            assert isinstance(self.args[1], index) or np.allclose(self.args[1], op2)
            value = np.array(list(map(lambda x: self.target(x[0], x[1]), pairs)), dtype=np.int)
        else:
            value = self.target(op1, op2)

        return value

    def __repr__(self):
        return "<index_%s '%s'>" % (self.target.__name__, self.name)