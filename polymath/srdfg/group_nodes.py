import operator
import builtins
import numpy as np
from .nodes import slice_op
from .base import Node, add, sub, mul, min_, max_, var_index, DEFAULT_SHAPES
from .util import _flatten_iterable

class GroupNode(Node):
    builtin_np = ["sum", "prod", "amax", "amin", "argmin", "argmax"]
    scalar_op_map = {"sum": operator.add, "prod": operator.mul, "amax": builtins.max, "amin": builtins.min, "argmin": np.argmin, "argmax": np.argmax, "bitreverse": lambda a, b: (a << 1) | (b & 1)}
    def __init__(self, target, bounds, input_node, **kwargs):
        self.output_nodes = []
        target_name = f"{target.__module__}.{target.__name__}"
        if "domain" in kwargs:
            domain = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
        elif isinstance(input_node, Node):
            domain = input_node.domain.reduction_domain(_flatten_iterable(bounds))
        else:
            raise ValueError(f"Group operations unable to handle non node inputs currently: {input_node} - {target_name}")

        if "axes" in kwargs:
            axes = kwargs.pop("axes") if isinstance(kwargs["axes"], tuple) else tuple(kwargs.pop("axes"))
        else:
            axes = input_node.domain.compute_set_reduction_index(domain)

        super(GroupNode, self).__init__(bounds, input_node, target=target_name, domain=domain, axes=axes, **kwargs)
        self.target = target
        if self.target.__name__ == "reduce":
            self.scalar_target = self.scalar_op_map[self.__class__.__name__]
        else:
            self.scalar_target = self.scalar_op_map[self.target.__name__]
        self.input_node = input_node

    def __getitem__(self, key):
        if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 0:
            return self
        # elif self.is_shape_finalized() and self.shape == DEFAULT_SHAPES[0]:
        #     return self
        elif self.is_shape_finalized() and len(self.nodes) > 0:
            if isinstance(key, int):
                key = tuple([key])
            idx = np.ravel_multi_index(key, dims=self.shape, order='C')
            ret = self.output_nodes[idx]
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
                        name.append(k)
            else:
                name.append(key)

            name = f"{self.var.name}{tuple(name)}"
            if name in self.graph.nodes:
                return self.graph.nodes[name]
            elif isinstance(key, (list)):
                return var_index(self, key, name=name, graph=self.graph)
            elif isinstance(key, tuple):
                return var_index(self, list(key), name=name, graph=self.graph)
            else:
                return var_index(self, [key], name=name, graph=self.graph)

    @property
    def axes(self):
        return self.kwargs["axes"]

    @property
    def domain(self):
        return self.kwargs["domain"]

    def _evaluate(self, bounds, input_res, **kwargs):

        sum_axes = self.axes
        if self.target.__name__ in ["argmax", "argmin"] and isinstance(sum_axes, tuple):
            assert len(sum_axes) == 1
            sum_axes = sum_axes[0]
        if not hasattr(input_res, "__len__"):
            value = input_res * np.prod([len(bound) for bound in bounds])
        elif self.target.__name__ in self.builtin_np:
            value = self.target(input_res.reshape(self.args[1].domain.computed_set_shape), axis=sum_axes)
        else:
            value = self.target(input_res.reshape(self.args[1].domain.computed_set_shape), axis=sum_axes, initial=self.initial)

        if len(value.shape) == 0:
            value = np.asarray([value])

        # if value.shape == DEFAULT_SHAPES[0]:
        #     value = value[0]

        if not self.is_shape_finalized():
            self.shape = value.shape

        # if len(value.shape) == 0:
        #     value = np.asarray([value])


        return value

    def __add__(self, other):
        return slice_op(operator.add, self, other, graph=self.graph) if not self.domain.is_scalar else add(self, other,
                                                                                                       graph=self.graph)

    def __radd__(self, other):
        return slice_op(operator.add, other, self, graph=self.graph) if not self.domain.is_scalar else add(other, self,
                                                                                                       graph=self.graph)
    def __sub__(self, other):
        return slice_op(operator.sub, self, other, graph=self.graph) if not self.domain.is_scalar else sub(self, other,
                                                                                                       graph=self.graph)
    def __rsub__(self, other):
        return slice_op(operator.sub, other, self, graph=self.graph) if not self.domain.is_scalar else sub(other, self,
                                                                                                       graph=self.graph)
    def __pow__(self, other):
        return slice_op(builtins.pow, self, other, graph=self.graph)

    def __rpow__(self, other):
        return slice_op(builtins.pow, other, self, graph=self.graph)

    def __mul__(self, other):
        return slice_op(operator.mul, self, other, graph=self.graph) if not self.domain.is_scalar else mul(self, other, graph=self.graph)

    def __rmul__(self, other):
        return slice_op(operator.mul, other, self, graph=self.graph) if not self.domain.is_scalar else mul(other, self, graph=self.graph)

    def __truediv__(self, other):
        return slice_op(operator.truediv, self, other, graph=self.graph)

    def __rtruediv__(self, other):
        return slice_op(operator.truediv, other, self, graph=self.graph)

    def __floordiv__(self, other):
        return slice_op(operator.floordiv, self, other, graph=self.graph)

    def __rfloordiv__(self, other):
        return slice_op(operator.floordiv, other, self, graph=self.graph)

    def __mod__(self, other):
        return slice_op(operator.mod, self, other, graph=self.graph)

    def __rmod__(self, other):
        return slice_op(operator.mod, other, self, graph=self.graph)

    def __lshift__(self, other):
        return slice_op(operator.lshift, self, other, graph=self.graph)

    def __rlshift__(self, other):
        return slice_op(operator.lshift, other, self, graph=self.graph)

    def __rshift__(self, other):
        return slice_op(operator.rshift, self, other, graph=self.graph)

    def __rrshift__(self, other):
        return slice_op(operator.rshift, other, self, graph=self.graph)

    def __and__(self, other):
        return slice_op(operator.and_, self, other, graph=self.graph)

    def __rand__(self, other):
        return slice_op(operator.and_, other, self, graph=self.graph)

    def __or__(self, other):
        return slice_op(operator.or_, self, other, graph=self.graph)

    def __ror__(self, other):
        return slice_op(operator.or_, other, self, graph=self.graph)

    def __xor__(self, other):
        return slice_op(operator.xor, self, other, graph=self.graph)

    def __rxor__(self, other):
        return slice_op(operator.xor, other, self, graph=self.graph)

    def __lt__(self, other):
        return slice_op(operator.lt, self, other, graph=self.graph)

    def __le__(self, other):
        return slice_op(operator.lt, other, self, graph=self.graph)

    def __ne__(self, other):
        return slice_op(operator.ne, self, other, graph=self.graph)

    def __gt__(self, other):
        return slice_op(operator.gt, self, other, graph=self.graph)

    def __ge__(self, other):
        return slice_op(operator.ge, self, other, graph=self.graph)

    def __repr__(self):
        return f"<group_{self.op_name} '{self.name}'>"

class sum(GroupNode):
    def __init__(self, bounds, input_node, **kwargs):
        super(sum, self).__init__(np.sum, bounds, input_node, **kwargs)

class min(GroupNode):

    def __init__(self, bounds, input_node, **kwargs):
        super(min, self).__init__(np.min, bounds, input_node, **kwargs)

class prod(GroupNode):

    def __init__(self, bounds, input_node, **kwargs):
        super(prod, self).__init__(np.prod, bounds, input_node, **kwargs)

class max(GroupNode):

    def __init__(self, bounds, input_node, **kwargs):
        super(max, self).__init__(np.max, bounds, input_node, **kwargs)

class argmax(GroupNode):

    def __init__(self, bounds, input_node, **kwargs):
        super(argmax, self).__init__(np.argmax, bounds, input_node, **kwargs)

class argmin(GroupNode):

    def __init__(self, bounds, input_node, **kwargs):
        super(argmin, self).__init__(np.argmin, bounds, input_node, **kwargs)


class bitreverse(GroupNode):
    def __init__(self, bounds, input_node, **kwargs):
        shifter = lambda a, b: (a << 1) | (b & 1)
        np_shifter = np.frompyfunc(shifter, 2, 1).reduce
        self.initial = 0
        super(bitreverse, self).__init__(np_shifter, bounds, input_node, **kwargs)