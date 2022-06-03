import logging
import pickle
import sys
from .base import Node
import operator
import builtins
import functools
from numbers import Integral, Real
import numpy as np
from .index import index
from .domain import Domain
from .base import Node, nodeop, func_op, slice_op, var_index
import uuid
from .util import _noop_callback, deprecated, _flatten_iterable, is_iterable, _is_node_type_instance
from polymath import DEFAULT_SHAPES, UNSET_SHAPE

class placeholder(Node):  # pylint: disable=C0103,R0903
    """
    Placeholder that needs to be given in the context to be evaluated.
    """
    def __init__(self, name=None, type_modifier=None, default=None, uid=None, root_name=None, **kwargs):
        # TODO: Remove this and change to state, input, param, output
        self.type_modifier = type_modifier or "declaration"
        kwargs["uid"] = uid if uid else uuid.uuid4().hex
        root_name = root_name or name

        if 'layout' not in kwargs:
            kwargs['layout'] = 'nchw'

        super(placeholder, self).__init__(name=name, root_name=root_name, type_modifier=self.type_modifier, default=default, **kwargs)
        assert isinstance(self.shape, tuple)
        self._domain = Domain(self.shape)
        if self.shape == UNSET_SHAPE:
            self.shape = DEFAULT_SHAPES[0]

    @property
    def domain(self):
        return self._domain

    def evaluate(self, context, callback=None):
        callback = callback or _noop_callback
        with callback(self, context):
            value = self.get_context_value(context)

            if isinstance(value, (list, tuple, np.ndarray)) and not self.is_shape_finalized():

                value = value if isinstance(value, np.ndarray) else np.asarray(value)
                assert len(value.shape) == len(self.shape)
                for idx, dim in enumerate(value.shape):

                    if isinstance(self.shape[idx], Node):
                        context[self.shape[idx]] = dim
                        _ = self.shape[idx].evaluate(context)
        return value

    @property
    def root_name(self):
        return self.kwargs["root_name"]

    def get_context_value(self, context):
        if self in context:
            return context[self]
        n_hash = self.func_hash()

        for k, v in context.items():
            if n_hash == k.func_hash():
                context[self] = context[k]
                return context[self]
        if (self not in context.keys() or context[self] is None):
            raise ValueError(f"missing value for placeholder \n\t {self.name} \n\t {self.op_name}\n\t"
                             f"in graph '{self.graph.name}' with context\n'{context.keys()}'")

    def __getitem__(self, key):
        key = _flatten_iterable(key)
        if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 0:
            return self
        elif self.is_shape_finalized() and self.shape == DEFAULT_SHAPES[0]:
            return self
        elif self.is_shape_finalized() and len(self.nodes) > 0:
            idx = np.ravel_multi_index(key, dims=self.shape, order='C')
            ret = self.nodes.item_by_index(idx)
            return ret
        else:
            indices = str(tuple([i.name if isinstance(i, Node) else i for i in key])).replace("'","")
            name = f"{self.name}{indices}"
            if self.graph and name in self.graph.nodes:
                return self.graph.nodes[name]
            else:
                return var_index(self, key, name=name, graph=self.graph)

    def _evaluate(self, *args, **kwargs):  # pylint: disable=W0221
        raise ValueError("missing value for placeholder '%s'" % self.name)

    def evaluate_shape(self, context):
        shapes = []
        for s in self.shape:
            if isinstance(s, Node):
                sval = s.evaluate(context)
                if isinstance(sval, float):
                    sval = np.int(sval)
                context[s] = sval
                shapes.append(context[s])
                if s.value is None:
                    s.value = context[s]
            else:
                assert isinstance(s, Integral)
                shapes.append(s)
        self._shape = tuple(shapes)

    def __repr__(self):
        return "<placeholder '%s'>" % self.name

class input(placeholder):
    def __init__(self, name=None, **kwargs):
        if "type_modifier" in kwargs:
            kwargs.pop("type_modifier")
        super(input, self).__init__(name=name, type_modifier="input", **kwargs)

    def __repr__(self):
        return "<input '%s'>" % self.name

class output(placeholder):
    def __init__(self, name=None, **kwargs):
        if "type_modifier" in kwargs:
            kwargs.pop("type_modifier")
        if "write_count" not in kwargs:
            kwargs["write_count"] = 0
        super(output, self).__init__(name=name, type_modifier="output", **kwargs)
        self.alias = self.name

    def __setitem__(self, key, value):

        key = _flatten_iterable(key)
        name = f"{self.name}{self.write_count}"
        prev_name = f"{self.name}{self.write_count - 1}" if self.write_count > 0 else self.name
        if prev_name not in self.graph.nodes:
            raise KeyError(f"{prev_name} not found in graph {self.graph.name} nodes:\n"
                           f"Keys: {list(self.graph.nodes.keys())}")
        x = write(value, list(key), self.graph.nodes[prev_name], name=name, alias=self.name, graph=self.graph)
        self.write_count += 1

    def __getitem__(self, key):
        key = _flatten_iterable(key)
        # if self.name == "sqrtz":
        if isinstance(key, (tuple, list, np.ndarray)) and len(key) == 0:
            return self
        elif self.is_shape_finalized() and all([not isinstance(i, Node) for i in key]) and len(self.nodes) > 0:

        # elif self.is_shape_finalized() and all([not isinstance(i, Node) for i in key]):
            idx = np.ravel_multi_index(key, dims=self.shape, order='C')
            ret = self.current_value().nodes.item_by_index(idx)
            return ret
        else:

            idx_name = str(tuple([i.name if isinstance(i, Node) else i for i in key])).replace("'","")
            name = f"{self.current_value().name}{idx_name}"

            if name in self.graph.nodes:
                return self.graph.nodes[name]
            else:
                return var_index(self.current_value(), key, name=name, graph=self.graph)

    def set_shape(self, shape=None, init=False, override=False):
        if isinstance(shape, Integral):
            new_shape = tuple([shape])
        elif isinstance(shape, Node):
            new_shape = tuple([shape])
        elif not shape or len(shape) == 0:
            new_shape = UNSET_SHAPE
        else:
            shapes = []
            for dim in shape:
                if isinstance(dim, (Node, Integral)):
                    shapes.append(dim)
                elif isinstance(dim, float):
                    shapes.append(int(dim))
                else:
                    raise TypeError(f"Shape value must be placeholder or integer value for {self.name}\n"
                                    f"\tDim: {dim}"
                                    f"\n\t{self.kwargs} ")

            new_shape = tuple(shapes)
        if self.is_shape_finalized() and new_shape != self._shape and not override:
            raise RuntimeError(f"Overwriting shape which has already been set for node {self.name}\n"
                               f"Initial shape: {self._shape}\n"
                               f"New shape: {new_shape}")
        self._shape = new_shape


    # TODO: Need to freeze the graph after exiting scope
    def current_value(self):
        return self if self.write_count == 0 or len(Node._graph_stack) == 1 else self.graph.nodes[f"{self.name}{self.write_count - 1}"]
        # return self if self.write_count == 0 else self.graph.nodes[f"{self.name}{self.write_count - 1}"]

    def reset_writes(self):
        for i in range(self.write_count):
            self.graph.nodes.pop(f"{self.name}{i}")
        self.write_count = 0

    def evaluate(self, context, callback=None):
        callback = callback or _noop_callback
        with callback(self, context):
            if self not in context:
                if not self.is_shape_finalized() or self.shape == UNSET_SHAPE:
                    self.evaluate_shape(context)

                context[self] = np.empty(shape=self.shape)
                for i in range(self.write_count):
                    name = f"{self.name.replace('/', str(i) + '/')}{i}"
                    w_node = self.graph.nodes[name]
                    context[w_node] = w_node.evaluate(context)
                fname = f"{self.name.replace('/', str(self.write_count-1) + '/')}{self.write_count - 1}"
                final = self.graph.nodes[fname]
                value = context[final]
            else:
                value = context[self]
        return value

    def write(self, value):
        name = f"{self.name}{self.write_count}"
        prev_name = f"{self.name}{self.write_count - 1}" if self.write_count > 0 else self.name
        x = write(value, [], self.graph.nodes[prev_name], name=name, alias=self.name, graph=self.graph)
        self.write_count += 1

    @property
    def write_count(self):
        return self.kwargs["write_count"]

    @write_count.setter
    def write_count(self, value):
        self.kwargs["write_count"] = value

    def __repr__(self):
        return "<output '%s'>" % self.name

class state(placeholder):
    def __init__(self, name=None, init_value=None, **kwargs):
        if "type_modifier" in kwargs:
            kwargs.pop("type_modifier")

        if "write_count" not in kwargs:
            kwargs["write_count"] = 0

        if init_value is not None:
            if isinstance(init_value, (list, tuple)):
                init_value = np.asarray(init_value)
                kwargs['shape'] = init_value.shape
            elif isinstance(init_value, np.ndarray):
                kwargs['shape'] = init_value.shape
            elif isinstance(init_value, Integral):
                init_value = np.asarray(init_value)
                kwargs['shape'] = init_value.shape
            else:
                kwargs['shape'] = tuple([len(init_value)])
            kwargs['init_value'] = init_value
        else:
            kwargs['init_value'] = init_value

        super(state, self).__init__(name=name, type_modifier="state", **kwargs)
        self.alias = self.name

    def __setitem__(self, key, value):
        key = _flatten_iterable(key)
        name = f"{self.name}{self.write_count}"
        prev_name = f"{self.name}{self.write_count - 1}" if self.write_count > 0 else self.name
        prev_node = self.graph.nodes[prev_name]

        x = write(value, list(key), prev_node, name=name, alias=self.name, graph=self.graph)
        self.write_count += 1

    def __getitem__(self, key):
        key = _flatten_iterable(key)

        if self.is_shape_finalized() and len(self.nodes) > 0:
            idx = np.ravel_multi_index(key, dims=self.shape, order='C')

            ret = self.current_value().nodes.item_by_index(idx)
            return ret
        else:

            idx_name = str(tuple([i.name if isinstance(i, Node) else i for i in key])).replace("'","")
            name = f"{self.current_value().name}{idx_name}"
            if name in self.graph.nodes:
                return self.graph.nodes[name]
            else:
                return var_index(self.current_value(), key, name=name, graph=self.graph)

    # TODO: Need to freeze the graph after exiting scope
    def current_value(self):
        if self.write_count == 0 or len(Node._graph_stack) == 1:
            return self
        elif f"{self.name}{self.write_count - 1}" not in self.graph.nodes:
            name = f"{self.name}{self.write_count - 1}"

            raise RuntimeError(f"Unable to find node {name} in graph {self.graph} for node {self.name}."
                               f"All Nodes: {list(self.graph.nodes.keys())}")
        else:
            return self.graph.nodes[f"{self.name}{self.write_count - 1}"]

    def evaluate(self, context, callback=None):
        callback = callback or _noop_callback

        with callback(self, context):

            value = self.get_context_value(context)

            if isinstance(value, (list, tuple, np.ndarray)) and not self.is_shape_finalized():
                value = value if isinstance(value, np.ndarray) else np.asarray(value)
                assert len(value.shape) == len(self.shape)
                for idx, dim in enumerate(value.shape):
                    if isinstance(self.shape[idx], Node):
                        context[self.shape[idx]] = dim
                        _ = self.shape[idx].evaluate(context)
        return value

    def __repr__(self):
        return "<state '%s'>" % self.name

    @property
    def write_count(self):
        return self.kwargs["write_count"]

    @write_count.setter
    def write_count(self, value):
        self.kwargs["write_count"] = value

    @property
    def init_value(self):
        return self.kwargs['init_value']



class temp(placeholder):
    def __init__(self, name=None, **kwargs):
        if "write_count" not in kwargs:
            kwargs["write_count"] = 0
        if "type_modifier" in kwargs:
            kwargs.pop("type_modifier")
        self.type_modifier = "temp"

        super(temp, self).__init__(name=name, type_modifier=self.type_modifier, **kwargs)
        self.alias = self.name

    def __setitem__(self, key, value):
        key = _flatten_iterable(key)
        name = f"{self.name}{self.write_count}"
        prev_name = f"{self.name}{self.write_count - 1}" if self.write_count > 0 else self.name
        _ = write(value, list(key), self.graph.nodes[prev_name], name=name, alias=self.name, graph=self.graph)
        self.write_count += 1

    def __getitem__(self, key):
        key = _flatten_iterable(key)
        if len(key) == 0:
            return self
        elif self.is_shape_finalized() and self.shape == DEFAULT_SHAPES[0]:
            return self
        elif self.is_shape_finalized() and len(self.nodes) > 0:
            idx = np.ravel_multi_index(key, dims=self.shape, order='C')
            ret = self.current_value().nodes.item_by_index(idx)

            return ret
        else:
            idx_name = str(tuple([i.name if isinstance(i, Node) else i for i in key])).replace("'","")
            name = f"{self.current_value().name}{idx_name}"

            if name in self.graph.nodes:
                return self.graph.nodes[name]
            else:
                return var_index(self.current_value(), key, name=name, graph=self.graph)

    def current_value(self):
        return self if self.write_count == 0 or len(Node._graph_stack) == 1 else self.graph.nodes[f"{self.name}{self.write_count - 1}"]
        # return self if self.write_count == 0 else self.graph.nodes[f"{self.name}{self.write_count - 1}"]

    def evaluate(self, context, callback=None):
        callback = callback or _noop_callback
        with callback(self, context):
            if self not in context:
                if not self.is_shape_finalized() or self.shape == DEFAULT_SHAPES[0]:
                    self.evaluate_shape(context)
                # TODO: Find a way to use the input value or determine correct datatype
                context[self] = np.zeros(shape=self.shape)
                for i in range(self.write_count):
                    name = f"{self.name.replace(f'{self.root_name}/', f'{self.root_name}{str(i)}/')}{i}"
                    w_node = self.graph.nodes[name]
                    context[w_node] = w_node.evaluate(context)
                old_str = f"{self.root_name}/"
                repl_str = f"{self.root_name}{self.write_count - 1}/"
                fname = f"{self.name.replace(old_str, repl_str)}{self.write_count-1}"
                final = self.graph.nodes[fname]
                value = context[final]
            else:
                value = context[self]
        return value


    def __repr__(self):
        return "<temp '%s'>" % self.name

    @property
    def write_count(self):
        return self.kwargs["write_count"]

    @write_count.setter
    def write_count(self, value):
        self.kwargs["write_count"] = value

class write(Node):
    def __init__(self, src, dst_key, dst, **kwargs):
        if "type_modifier" in kwargs:
            kwargs.pop("type_modifier")
        self.type_modifier = dst.type_modifier

        if "shape" in kwargs:
            kwargs.pop("shape")

        if "domain" in kwargs:
            domain = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
        elif len(dst_key) == 0:
            domain = src.domain
        else:
            domain = Domain(dst_key)

        if "write_graph" in kwargs:
            kwargs.pop("write_graph")
        active_graph = Node.get_active_graph()
        graph_location = []
        while active_graph:
            graph_location.insert(0, active_graph.name)
            active_graph = active_graph.graph
        super(write, self).__init__(src, dst_key, dst, write_graph=graph_location, domain=domain, shape=dst.shape, type_modifier=self.type_modifier,
                                    **kwargs)
        self.alias = kwargs["alias"]
        self.shape_domain = Domain(self.dest.shape)

    @property
    def domain(self):
        return self.kwargs["domain"]

    def is_scalar_write(self, dst_key):
        return len(dst_key) == 1 and dst_key[0].shape == (1,)

    def scalar_dest(self):
        return all([isinstance(v, int) for v in self.args[1]])

    def constant_value_write(self):
        if isinstance(self.args[0], var_index):
            return self.args[0].scalar_result() and self.scalar_dest()
        elif isinstance(self.args[0], func_op):
            return self.scalar_dest()
        elif isinstance(self.args[0], slice_op):
            is_scalar_arg = []
            for a in self.args[0].args:
                if isinstance(a, Node):
                    is_scalar_arg.append(a.shape == DEFAULT_SHAPES[0] or len(a.shape) == 0)
            return all(is_scalar_arg) and self.scalar_dest()
        else:
            return False

    def _evaluate(self, src, dst_key, dst, context=None, **kwargs):
        if not self.is_shape_finalized():
            self._shape = self.args[2].shape
        # TODO: This should not be default shapes, fix

        if self.shape in DEFAULT_SHAPES:
            value = src
        elif not is_iterable(src) and not self.is_scalar_write(dst_key):
            value = np.full(self.shape, src)
        elif not is_iterable(src) and self.is_scalar_write(dst_key):
            dst_idx = self.shape_domain.compute_shape_domain(indices=dst_key)[0]
            value = dst.copy().astype(src.dtype)
            value[dst_idx] = src
        else:
            dst_indices = self.shape_domain.compute_shape_domain(indices=dst_key)
            key_indices = self.domain.compute_pairs()
            src_dom = self.args[0].domain

            src_indices = self.domain.map_sub_domain(src_dom, is_write=True)
            value = np.zeros(shape=dst.shape, dtype=src.dtype)

            # TODO: Fix logic here to check for validity
            if len(src_indices[0]) != len(src.shape):
                src = src.squeeze()

            for i in dst_indices:
                if i in key_indices:
                    idx = key_indices.index(i)
                    src_idx = src_indices[idx]
                    value[i] = src[src_idx]
                else:
                    value[i] = dst[i]

        return value

    def __getitem__(self, key):
        key = _flatten_iterable(key)
        if len(key) == 0:
            return self
        elif self.is_shape_finalized():
            idx = np.ravel_multi_index(key, dims=self.shape, order='C')
            ret = self.nodes.item_by_index(idx)
            return ret
        else:
            idx_name = str(tuple([i.name if isinstance(i, Node) else i for i in key])).replace("'","")
            name = f"{self.name}{idx_name}"
            if name in self.graph.nodes:
                return self.graph.nodes[name]
            else:
                return var_index(self, key, name=name, graph=self.graph)

    @property
    def source(self):
        return self.args[0]

    @property
    def dest(self):
        return self.args[2]

    def __repr__(self):
        return "<write '%s'>" % self.name

class parameter(placeholder):
    """
    Node returning the input value.
    """
    def __init__(self, name=None, default=None, **kwargs):
        if default and hasattr(default, "__len__") and not isinstance(default, str):
            raise TypeError(f"Cannot use parameter for non scalar value {default}."
                            f"Use variable instead of parameter.")
        if "type_modifier" in kwargs:
            kwargs.pop("type_modifier")
        if "shape" in kwargs:
            kwargs.pop("shape")
        super(parameter, self).__init__(name=name, default=default, type_modifier="param", shape=DEFAULT_SHAPES[0], **kwargs)
        self.default = default

    evaluate = Node.evaluate

    def _evaluate(self, value=None, **kwargs):
        if value is None and self.default is None:
            raise ValueError(f"Value not supplied for parameter {self.name}"
                             f" and default value is not set.")
        return value or self.default

    def __repr__(self):
        return "<parameter '%s'>" % self.name

class variable(Node):  # pylint: disable=C0103,R0903
    """
    Placeholder that needs to be given in the context to be evaluated.
    """
    def __init__(self, value, name=None, **kwargs):
        assert hasattr(value, "__len__")
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
            kwargs['shape'] = value.shape
        else:
            kwargs['shape'] = tuple([len(value)])
        super(variable, self).__init__(value, name=name, **kwargs)

    def evaluate(self, context, callback=None):
        callback = callback or _noop_callback
        shape = tuple([self.evaluate_node(s, context, callback=callback) for s in self.shape])
        with callback(self, context):
            value, = self.args
            if isinstance(value, str):
                pass
            elif not isinstance(value, (list, tuple)):
                value = np.zeros(shape) + value
            context[self] = value
        self.finalized = True
        return value


    def _evaluate(self, value, context=None, **kwargs):
        if isinstance(value, str):
            pass
        elif not isinstance(value, (list, tuple)):
            value = np.zeros(self.shape) + value
        return value

    def __repr__(self):
        return "<variable '%s' args=%s kwargs=%s>"%  \
               (self.name, self.args, self.kwargs)


class predicate(Node):  # pylint: disable=C0103,W0223
    """
    Return `x` if `predicate` is `True` and `y` otherwise.

    .. note::
        The conditional node will only execute one branch of the computation graph depending on
        `predicate`.
    """
    def __init__(self, pred, x, y=None, *, name=None, dependencies=None, shape=None):  # pylint: disable=W0235
        super(predicate, self).__init__(pred, x, y, name=name, dependencies=dependencies, shape=shape)

    def evaluate(self, context, callback=None):
        # Evaluate all dependencies first
        callback = callback or _noop_callback
        self.evaluate_dependencies(context, callback)

        pred, x, y = self.args  # pylint: disable=E0632,C0103
        # Evaluate the predicate and pick the right node
        pred = self.evaluate_node(pred, context, callback=callback)
        with callback(self, context):
            value = self.evaluate_node(x if pred else y, context, callback=callback)
            context[self] = value
        self.finalized = True
        return value


class try_(Node):  # pylint: disable=C0103,W0223
    """
    Try to evaluate `node`, fall back to alternative nodes in `except_`, and ensure that
    `finally_` is evaluated.

    .. note::
        The alternative nodes will only be executed if the target node fails.

    Parameters
    ----------
    node : Node
        Node to evaluate.
    except_ : list[(type, Node)]
        List of exception types and corresponding node to evaluate if it occurs.
    finally_ : Node
        Node to evaluate irrespective of whether `node` fails.
    """
    def __init__(self, node, except_=None, finally_=None, **kwargs):
        except_ = except_ or []
        super(try_, self).__init__(node, except_, finally_, **kwargs)

    def evaluate(self, context, callback=None):
        # Evaluate all dependencies first
        callback = callback or _noop_callback
        self.evaluate_dependencies(context, callback=callback)

        node, except_, finally_ = self.args # pylint: disable=E0632,C0103
        with callback(self, context):
            try:
                value = self.evaluate_node(node, context, callback=callback)
                context[self] = value
                self.finalized = True
                return value
            except:
                # Check the exceptions
                _, ex, _ = sys.exc_info()
                for type_, alternative in except_:
                    if isinstance(ex, type_):
                        value = self.evaluate_node(alternative, context, callback=callback)
                        context[self] = value
                        self.finalized = True
                        return value
                raise
            finally:
                if finally_:
                    self.evaluate_node(finally_, context)

def _pickle_load(filename):
    with open(filename, 'rb') as fp:  # pylint: disable=invalid-name
        return pickle.load(fp)


def _pickle_dump(value, filename):
    with open(filename, 'wb') as fp:  # pylint: disable=invalid-name
        pickle.dump(value, fp)

class lazy_constant(Node):  # pylint: disable=invalid-name
    """
    Node that returns the output of `target` lazily.

    Parameters
    ----------
    target : callable
        Function to evaluate when the node is evaluated.
    kwargs : dict
        Keyword arguments passed to the constructor of `Node`.
    """
    def __init__(self, target, **kwargs):
        super(lazy_constant, self).__init__(**kwargs)
        self.target = target
        if not callable(self.target):
            raise ValueError("`target` must be callable")
        self.value = None

    def _evaluate(self):  # pylint: disable=W0221
        if self.value is None:
            self.value = self.target()
        return self.value


@nodeop
def identity(value):
    """
    Node returning the input value.
    """
    return value

@nodeop
def assert_(condition, message=None, *args, val=None):  # pylint: disable=keyword-arg-before-vararg
    """
    Return `value` if the `condition` is satisfied and raise an `AssertionError` with the specified
    `message` and `args` if not.
    """
    if message:
        assert condition, message % args
    else:
        assert condition

    return val

@nodeop
def str_format(format_string, *args, **kwargs):
    """
    Use python's advanced string formatting to convert the format string and arguments.

    References
    ----------
    https://www.python.org/dev/peps/pep-3101/
    """
    return format_string.format(*args, **kwargs)