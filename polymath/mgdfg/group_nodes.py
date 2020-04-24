
from polymath.mgdfg.base import *
from polymath.mgdfg.nodes import parameter
from polymath.mgdfg.domain import Domain
from .util import _flatten_iterable, _fnc_hash

class GroupNode(Node):
    builtin_np = ["sum", "prod", "max", "min", "argmin", "argmax"]
    scalar_op_map = {"sum": operator.add, "prod": operator.mul, "max": max_, "min": min_, "argmin": min_, "argmax": max_, "bitreverse": lambda a, b: (a << 1) | (b & 1)}
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
        elif self.is_shape_finalized() and len(self.nodes) > 0:
            if isinstance(key, int):
                key = tuple([key])
            idx = np.ravel_multi_index(key, dims=self.shape, order='F')
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

    @property
    def axes(self):
        return self.kwargs["axes"]

    @property
    def domain(self):
        return self.kwargs["domain"]

    def _evaluate(self, bounds, input_res, **kwargs):
        sum_axes = self.axes
        if not hasattr(input_res, "__len__"):
            value = input_res * np.prod([len(bound) for bound in bounds])
        elif self.target.__name__ in self.builtin_np:
            value = self.target(input_res.reshape(self.args[1].domain.computed_set_shape), axis=sum_axes)
        else:
            value = self.target(input_res.reshape(self.args[1].domain.computed_set_shape), axis=sum_axes, initial=self.initial)
        return value

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