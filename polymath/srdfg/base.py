
from polymath import UNSET_SHAPE, DEFAULT_SHAPES
import builtins
import operator
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
import functools
from numbers import Integral, Real
import contextlib
import traceback
import uuid
import numpy as np
import importlib
from .graph import Graph
from .domain import Domain
from .util import _noop_callback, _flatten_iterable, node_hash, \
    _is_node_type_instance, is_iterable

class Node(object):
    """
    Base class for nodes.

    Parameters
    ----------
    args : tuple
        Positional arguments passed to the `_evaluate` method.
    name : str or None
        Name of the node or `None` to use a random, unique identifier.
    shape : tuple or None
        Shape of the output for a node. This can be a tuple of integers or parameter node names.
    graph : Node or None
        Parent graph of this node. If graph is `None`, this is the top-level graph.
    op_name : str
        Operation name which describes the node functionality.
    value : Any or None
        If a node has a default value to use for execution, it can be set using `value`.
    kwargs : dict
        Keyword arguments passed to the `_evaluate` method.
    """
    _graph_stack = deque([None])
    _eval_stack = []
    stack_size = 5
    evaluated_nodes = 0
    def __init__(self, *args,
                 name=None,
                 shape=None,
                 graph=None,
                 dependencies=None,
                 op_name=None,
                 value=None,
                 **kwargs):
        self.nodes = Graph()
        self.value = value
        self.dependencies = []
        self._args = []
        self._predeecessors = []
        self._succesors = []
        self.args = args
        if "name" in kwargs:
            kwargs.pop("name")
        self.added_attrs = []
        # TODO: CHange this to underscore private variable
        self.kwargs = kwargs
        self.graph = graph
        self._shape = OrderedDict()
        self.shape = shape or tuple([])


        # Get a list of all dependencies relevant to this node
        self.dependencies = [] if dependencies is None else dependencies
        if self.graph:
            self.dependencies.extend(self.graph.dependencies)
        # Choose a name for the node and add the node to the graph
        self._name = None
        self.name = name or uuid.uuid4().hex
        self._op_name = None
        self.op_name = op_name
        # Get the stack context so we can report where the node was defined
        self._stack = traceback.extract_stack(limit=1)


    @property
    def graph(self):
        """
        polymath.srdfg.graph.Graph : Parent graph of this node. If graph is `None`, this is the top-level graph.
        """
        return self._graph

    def preds(self):
        return self._preds

    def succs(self):
        return self._preds

    def add_predecessor(self, pred):
        if isinstance(pred, Node):
            self._predecessors.append(pred.gname)
        else:
            self._predecessors.append(pred)

    def add_successor(self, succ):
        if isinstance(succ, Node):
            self._succesors.append(succ.gname)
        else:
            self._succesors.append(succ)

    def set_edges(self):
        for e in self.args:
            self.add_predecessor(e)
            if isinstance(e, Node):
                e.add_successor(self)

    @property
    def domain(self):
        return Domain(tuple([]))

    @property
    def args(self):
        """
        tuple : Positional arguments which are used for executing this node.
        """
        return tuple(self._args)

    @property
    def argnames(self):
        return [a.name if isinstance(a, Node) else a for a in self.args]

    @property
    def shape(self):
        """
        tuple : Shape of the output for a node. This can be a tuple of integers or parameter node names.
        """
        return self._shape

    @property
    def var(self):
        return self

    @property
    def name(self):
        """str : Unique name of the node"""
        return self._name

    @property
    def op_name(self):
        """
        str : Operation name which describes the node functionality.

        """
        return self._op_name

    @op_name.setter
    def op_name(self, op_name):

        if op_name:
            self._op_name = op_name
        elif self.__class__.__name__ == "Node":
            self._op_name = self.name
        else:
            self._op_name = self.__class__.__name__

    @name.setter
    def name(self, name):
        self.set_name(name)

    @args.setter
    def args(self, args):
        new_args = []
        for arg in args:
            if isinstance(arg, Node):
                if self.__class__.__name__ == "Node":
                    self.nodes[arg.name] = self.graph[arg.name]
            new_args.append(arg)
        self._args = tuple(new_args)

    @shape.setter
    def shape(self, shape):
        self.set_shape(shape, init=True)

    @graph.setter
    def graph(self, graph):
        self._graph = Node.get_active_graph(graph)

    @property
    def gname(self):
        scope_names = [self.name]
        cgraph = self.graph
        while cgraph:
            scope_names.append(cgraph.name)
            cgraph = cgraph.graph
        return "/".join(list(reversed(scope_names)))

    def __enter__(self):
        Node._graph_stack.append(self)
        return self

    def __exit__(self, *args):
        assert self == Node._graph_stack.pop()

    def __repr__(self):
        return "<node '%s'>" % self.name

    def add_attribute(self, key, value):
        self.added_attrs.append(key)
        self.kwargs[key] = value

    def is_shape_finalized(self):
        if self.shape == UNSET_SHAPE or isinstance(self.shape, OrderedDict):
            return False
        for s in self.shape:
            if not isinstance(s, Integral):
                return False
        return True

    def set_shape(self, shape=None, init=False, override=False):
        if isinstance(shape, float):
            new_shape = tuple([np.int(shape)])
        elif isinstance(shape, Integral):
            new_shape = tuple([shape])
        elif isinstance(shape, Node):
            new_shape = tuple([shape])
        elif not shape or len(shape) == 0:
            # TODO: Change in order to enable "is shape finalized" to work
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
            raise RuntimeError(f"Overwriting shape which has already been set for node\n"
                               f"Initial shape: {self._shape}\n"
                               f"New shape: {new_shape}")

        self._shape = new_shape


    @staticmethod
    def get_active_graph(graph=None):
        """
        Obtain the currently active graph instance by returning the explicitly given graph or using
        the default graph.

        Parameters
        ----------
        graph : Node or None
            Graph to return or `None` to use the default graph.

        Raises
        ------
        ValueError
            If no `Graph` instance can be obtained.
        """

        graph = graph or Node._graph_stack[-1]
        return graph

    def instantiate_node(self, node):  # pylint:disable=W0621
        """
        Instantiate nodes by retrieving the node object associated with the node name.

        Parameters
        ----------
        node : Node or str
            Node instance or name of an node.

        Returns
        -------
        instantiated_node : Node
            Node instance.

        Raises
        ------
        ValueError
            If `node` is not an `Node` instance or an node name.
        RuntimeError
            If `node` is an `Node` instance but does not belong to this graph.
        """
        if isinstance(node, str):
            return self.nodes[node]

        if isinstance(node, Node):
            if node.name not in self.nodes and (node.graph != self):
                raise RuntimeError(f"node '{node}' does not belong to {self} graph, instead belongs to"
                                   f" {node.graph}")
            return node

        raise ValueError(f"'{node}' is not an `Node` instance or node name")

    def instantiate_graph(self, context, **kwargs):
        """
        Instantiate a graph by replacing all node names with node instances.

        .. note::
           This function modifies the context in place. Use :code:`context=context.copy()` to avoid
           the context being modified.

        Parameters
        ----------
        context : dict[Node or str, object]
            Context whose keys are node instances or names.
        kwargs : dict[str, object]
            Additional context information keyed by variable name.

        Returns
        -------
        normalized_context : dict[Node, object]
            Normalized context whose keys are node instances.

        Raises
        ------
        ValueError
            If the context specifies more than one value for any node.
        ValueError
            If `context` is not a mapping.
        """
        if context is None:
            context = {}
        elif not isinstance(context, Mapping):
            raise ValueError("`context` must be a mapping.")

        nodes = list(context)
        # Add the keyword arguments
        for node in nodes:  # pylint:disable=W0621
            value = context.pop(node)
            node = self.instantiate_node(node)
            if node in context:
                raise ValueError(f"duplicate unequal value for node '{node}'")
            context[node] = value
            if node.op_name in ["placeholder", "state", "input", "output", "temp"] and not node.is_shape_finalized():
                context[node] = node.evaluate(context)

        for name, value in kwargs.items():
            node = self.nodes[name]
            if node in context:
                raise ValueError(f"duplicate value for node '{node}'")
            context[node] = value
            if node.op_name in ["placeholder", "state", "input", "output", "temp"] and not node.is_shape_finalized():
                context[node] = node.evaluate(context)

        return context

    def run(self, fetches, context=None, *, callback=None, **kwargs):
        """
        Evaluate one or more nodes given a dictionary of node names with their values.

        .. note::
           This function modifies the context in place. Use :code:`context=context.copy()` to avoid
           the context being modified.

        Parameters
        ----------
        fetches : list[str or Node] or str or Node
            One or more `Node` instances or names to evaluate.
        context : dict or None
            Context in which to evaluate the nodes.
        callback : callable or None
            Callback to be evaluated when an node is evaluated.
        kwargs : dict
            Additional context information keyed by variable name.

        Returns
        -------
        values : Node or tuple[object]
            Output of the nodes given the context.

        Raises
        ------
        ValueError
            If `fetches` is not an `Node` instance, node name, or a sequence thereof.
        """
        if isinstance(fetches, (str, Node)):
            fetches = [fetches]
            single = True
        elif isinstance(fetches, Sequence):
            single = False
        else:
            raise ValueError("`fetches` must be an `Node` instance, node name, or a "
                             "sequence thereof.")
        fetches = [self.instantiate_node(node) for node in fetches]
        context = self.instantiate_graph(context, **kwargs)
        for c in context:
            if c in fetches and c.op_name in ["output", "state", "temp"]:
                write_name = "/".join([f"{i}{c.write_count-1}" for i in c.name.split("/")]) if c.write_count > 0 else c.name
                fetches[fetches.index(c)] = c.graph.nodes[write_name]

        values = [fetch.evaluate_node(fetch, context, callback=callback) for fetch in fetches]

        return values[0] if single else tuple(values)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, data):
        self.__dict__.update(data)

    def set_name(self, name):
        """
        Set the name of the node and update the graph.

        Parameters
        ----------
        value : str
            Unique name of the node.

        Returns
        -------
        self : Node
            This node.

        Raises
        ------
        ValueError
            If an node with `value` already exists in the associated graph.
        KeyError
            If the current name of the node cannot be found in the associated graph.
        """

        name = name or uuid.uuid4().hex
        # TODO: Need a way to check if the existing node is not equal to the current ndoe as ewll
        if self.graph and name in self.graph.nodes:
            raise ValueError(f"duplicate name '{name}' in {self.graph.name}:\n\t"
                             f"Existing: {self.graph.nodes[name].args}\n\t"
                             f"New: {self.args}")

        if self.graph:
            graph = self.graph
            if self._name and self._name in graph.nodes:
                graph.update_graph_key(self._name, name)
            else:
                graph.nodes[name] = self

        self._name = name
        return self

    def evaluate_dependencies(self, context, callback=None):
        """
        Evaluate the dependencies of this node and discard the values.

        Parameters
        ----------
        context : dict
            Normalised context in which to evaluate the node.
        callback : callable or None
            Callback to be evaluated when an node is evaluated.
        """
        for node in self.dependencies:
            node.evaluate(context, callback)

    def evaluate(self, context, callback=None):
        """
        Evaluate the node given a context.

        Parameters
        ----------
        context : dict
            Normalised context in which to evaluate the node.
        callback : callable or None
            Callback to be evaluated when an node is evaluated.

        Returns
        -------
        value : object
            Output of the node given the context.
        """
        # Evaluate all explicit dependencies first

        self.evaluate_dependencies(context, callback)

        if self in context:
            return context[self]

        # Evaluate the parents
        partial = functools.partial(self.evaluate_node, context=context, callback=callback)

        args = [partial(arg) for arg in self.args]
        kwargs = {key: partial(value) for key, value in self.kwargs.items() if key not in self.added_attrs}
        # Evaluate the node
        callback = callback or _noop_callback
        with callback(self, context):
            if self.__class__.__name__ == "Node":
                context[self] = self.value = self._evaluate(*args, context=context, **kwargs)
            else:
                context[self] = self.value = self._evaluate(*args, **kwargs)
        return self.value

    def _evaluate(self, *args, context=None, **kwargs):
        """
        Inheriting nodes should implement this function to evaluate the node.
        """
        return self(*args, context, **kwargs)

    @classmethod
    def evaluate_node(cls, node, context, **kwargs):
        """
        Evaluate an node or constant given a context.
        """
        Node.evaluated_nodes += 1
        try:
            if isinstance(node, Node):
                Node._eval_stack.append(node.name)
                return node.evaluate(context, **kwargs)
            partial = functools.partial(cls.evaluate_node, context=context, **kwargs)
            if isinstance(node, tuple):
                return tuple(partial(element) for element in node)
            if isinstance(node, list):
                return [partial(element) for element in node]
            if isinstance(node, dict):
                return {partial(key): partial(value) for key, value in node.items()}
            if isinstance(node, slice):
                return slice(*[partial(getattr(node, attr))
                                     for attr in ['start', 'stop', 'step']])
            return node
        except Exception as ex:  # pragma: no cover
            messages = []
            interactive = False
            if isinstance(node, Node) or not is_iterable(node):
                node = [node]

            for n in node:
                stack = []
                if isinstance(n, Node):

                    for frame in reversed(n._stack):  # pylint: disable=protected-access
                        # Do not capture any internal stack traces
                        fname = frame.filename
                        if 'polymath' in fname:
                            continue
                        # Stop tracing at the last interactive cell
                        if interactive and not fname.startswith('<'):
                            break  # pragma: no cover
                        interactive = fname.startswith('<')
                        stack.append(frame)
                    stack = "".join(traceback.format_list(reversed(stack)))
                message = "Failed to evaluate node `%s` defined at:\n\n%s" % (n, stack)
                messages.append(message)
            raise ex from EvaluationError("".join(messages))


    @classmethod
    def init_from_args(cls, *args,
                       name=None,
                       shape=None,
                       graph=None,
                       dependencies=None,
                       op_name=None,
                       value=None,
                       **kwargs):
        if len(args) == 0:
            n = cls(name=name,
                    shape=shape,
                    graph=graph,
                    op_name=op_name,
                    dependencies=dependencies,
                    value=value,
                    **kwargs)
        else:
            n = cls(*args,
                    name=name,
                    shape=shape,
                    graph=graph,
                    op_name=op_name,
                    dependencies=dependencies,
                    value=value,
                    **kwargs)
        return n

    def __bool__(self):
        return True

    def __hash__(self):
        return id(self)

    def func_hash(self):
        """
        This returns the functional hash of a particular node. The default hash returns an object id, whereas this function
        returns a hash of all attributes and subgraphs of a node.
        """
        return node_hash(self)

    def find_node(self, name):
        g = self.graph
        while g is not None and name not in g.nodes:
            g = g.graph
        if name in g.nodes:
            return g.nodes[name]
        raise RuntimeError(f"Cannot find {name} in graph nodes. Graph: {self.graph}")

    def __len__(self):
        #TODO: Update this to check for finalzied shape
        if self.shape == UNSET_SHAPE:
            raise TypeError(f'`shape` must be specified explicitly for nodes {self}')
        return self.shape[0]

    def __iter__(self):
        num = len(self)
        for i in range(num):
            yield self[i]

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __getattr__(self, name):
        return getattr_(self, name, graph=self.graph)

    def __getitem__(self, key):
        if self.__class__.__name__ != "Node":
            if isinstance(key, (slice, Integral)):
                return getitem(self, key, graph=self.graph)
            else:
                if isinstance(key, (list)):
                    return var_index(self, key, graph=self)
                elif isinstance(key, tuple):
                    return var_index(self, list(key), graph=self)
                else:
                    return var_index(self, [key], graph=self)
        else:
            return self.nodes[key]

    def __add__(self, other):
        return add(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__radd__(self)

    def __radd__(self, other):
        return add(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__add__(self)

    def __sub__(self, other):
        return sub(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rsub__(self)

    def __rsub__(self, other):
        return sub(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__sub__(self)

    def __pow__(self, other):
        return pow_(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rpow__(self)

    def __rpow__(self, other):
        return pow_(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rpow__(self)

    def __matmul__(self, other):
        return matmul(self, other, graph=self.graph)

    def __rmatmul__(self, other):
        return matmul(other, self, graph=self.graph)

    def __mul__(self, other):
        return mul(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rmul__(self)

    def __rmul__(self, other):
        return mul(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__mul__(self)

    def __truediv__(self, other):
        return truediv(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__truediv__(self)

    def __rtruediv__(self, other):
        return truediv(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rtruediv__(self)

    def __floordiv__(self, other):
        return floordiv(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rfloordiv__(self)

    def __rfloordiv__(self, other):
        return floordiv(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__floordiv__(self)

    def __mod__(self, other):
        return mod(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rmod__(self)

    def __rmod__(self, other):
        return mod(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__mod__(self)

    def __lshift__(self, other):
        return lshift(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rlshift__(self)

    def __rlshift__(self, other):
        return lshift(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__lshift__(self)

    def __rshift__(self, other):
        return rshift(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rrshift__(self)

    def __rrshift__(self, other):
        return rshift(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rshift__(self)

    def __and__(self, other):
        return and_(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rand__(self)

    def __rand__(self, other):
        return and_(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__and__(self)

    def __or__(self, other):
        return or_(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__ror__(self)

    def __ror__(self, other):
        return or_(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__or__(self)

    def __xor__(self, other):
        return xor(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__rxor__(self)

    def __rxor__(self, other):
        return xor(other, self, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__xor__(self)

    def __lt__(self, other):
        return lt(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__gt__(self)

    def __le__(self, other):
        return le(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__ge__(self)


    def __ne__(self, other):
        return ne(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__ne__(self)

    def __gt__(self, other):
        return gt(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__lt__(self)

    def __ge__(self, other):
        return ge(self, other, graph=self.graph) if not _is_node_type_instance(other, ("slice_op", "var_index", "index")) else other.__le__(self)

    def __invert__(self):
        return inv(self, graph=self.graph)

    def __neg__(self):
        return neg(self, graph=self.graph)

    def __abs__(self):
        return abs_(self, graph=self.graph)

    def __pos__(self):
        return pos(self, graph=self.graph)

    def __reversed__(self):
        return reversed_(self, graph=self.graph)

    def update_template_index(self, temp):
        assert isinstance(temp, Node) and hasattr(temp, 'inputs')
        assert all([i.name in self.nodes for i in temp.inputs])
        self.nodes.pop(temp.name)
        node_list = list(self.nodes.keys())
        min_idx = max([node_list.index(i.name) for i in temp.inputs])

        self.insert_node(temp, min_idx + 1)

    def update_graph_key(self, old_key, new_key):
        n = list(map(lambda k: (new_key, self.nodes[k]) if k == old_key else (k, self.nodes[k]), self.nodes.keys()))
        self.nodes = Graph(n)

    def replace_graph_key(self, old_key, new_node):
        n = list(map(lambda k: (new_node.name, new_node) if k == old_key else (k, self.nodes[k]), self.nodes.keys()))
        self.nodes = Graph(n)

    def insert_node(self, node, idx):
        node_list = list(self.nodes.items())

        node_list.insert(idx, (node.name, node))
        self.nodes = Graph(node_list)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

class EvaluationError(RuntimeError):
    """
    Failed to evaluate an node.
    """


class var_index(Node):  # pylint: disable=C0103,W0223
    """
    Node representing values of a variable corresponding to input index values.

    Parameters
    ----------
    var : Node
        The multi-dimensional variable used for indexing into.
    idx : tuple
        Tuple of either integer values or index/index_op nodes.
    """
    def __init__(self, var, idx, name=None, **kwargs):  # pylint: disable=W0235
        if "domain" in kwargs:
            domain = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
        else:
            domain = Domain(idx)

        super(var_index, self).__init__(var, idx, name=name, domain=domain, **kwargs)

    @property
    def domain(self):
        return self.kwargs["domain"]

    @property
    def var(self):
        var, index_list = self.args
        return var

    def set_name(self, name):
        """
        Set the name for a variable index, making sure to replicate the new name with
        a unique stringwhich corresponds to the variable, index combination.

        Parameters
        ----------
        value : str
            Unique name of the node.

        Returns
        -------
        self : Node
            This node.

        Raises
        ------
        ValueError
            If an node with `value` already exists in the associated graph.
        KeyError
            If the current name of the node cannot be found in the associated graph.
        """

        # TODO: Need a way to check if the existing node is not equal to the current ndoe as ewll
        if self.graph and name in self.graph.nodes:
            raise ValueError(f"duplicate name '{name}' in {self.graph.name}:"
                             f"Existing: {self.graph.nodes[name].args}\n"
                             f"New: {self.args}")

        if self.graph:
            graph = self.graph
            if self._name is not None and self._name in graph.nodes:
                graph.update_graph_key(self._name, name)
            else:
                graph.nodes[name] = self

        self._name = name
        return self

    def __getitem__(self, key):
        if self.is_shape_finalized() and len(self.nodes) >= np.prod(self.shape):
            if isinstance(key, Integral):
                key = tuple([key])
            idx = np.ravel_multi_index(key, dims=self.shape, order='C')
            ret = self.nodes.item_by_index(idx)
            return ret
        else:
            if isinstance(key, (list)):
                ret = var_index(self.var, tuple(key), graph=self)
            elif isinstance(key, tuple):
                ret = var_index(self.var, key, graph=self)
            else:
                ret = var_index(self.var, tuple([key]), graph=self)
            return ret

    def is_scalar(self, val=None):
        if val is not None and (not isinstance(val, np.ndarray) or (len(val.shape) == 1 and val.shape[0] == 1)):
            if self.var.shape != DEFAULT_SHAPES[0] and (len(self.var.shape) == 1 and not isinstance(self.var.shape[0],Node)):
                raise ValueError(f"Invalid shape var for var index {self} with variable shape {self.var.shape}")
            return True
        else:
            return self.var.shape == DEFAULT_SHAPES[0]

    def scalar_result(self):
        return all([isinstance(v, int) for v in self.args[1]])

    def _evaluate(self, var, indices, **kwargs):

        if self.is_scalar(var):
            out_shape = (1,)
            indices = (0,)
            single = True
        else:
            out_shape = self.domain.shape_from_indices(indices)
            indices = self.domain.compute_pairs()
            single = False
        if isinstance(var, (Integral, Real, str)):
            var = np.asarray([var])
        elif not isinstance(var, (np.ndarray, list)):
            raise TypeError(f"Variable {var} with type {type(var)} is not a list or numpy array, and cannot be sliced for {self.name}")
        elif isinstance(var, list):
            var = np.asarray(var)
        if len(var.shape) != len(out_shape) and np.prod(var.shape) == np.prod(out_shape):
            if len(out_shape) > len(var.shape):
                for i in range(len(out_shape)):
                    if out_shape[i] == 1:
                        var = np.expand_dims(var, axis=i)
            else:
                var = np.squeeze(var)

        if len(var.shape) != len(out_shape) and np.prod(var.shape) != np.prod(out_shape):
            raise ValueError(f"Index list does not match {var.shape} in {self.var.name} - {self.var.op_name}"
                             f"dimensions for slice {self.args[0].name} with {out_shape}.\n"
                             f"Domain: {self.domain}\n"
                             f"Eval Stack: {Node._eval_stack}")

        if not single and not all([(idx_val - 1) >= indices[-1][idx] for idx, idx_val in enumerate(var.shape)]):

            raise ValueError(f"var_index {self.name} has indices which are greater than the variable shape:\n"
                             f"\tArgs: {self.args}\n"
                             f"\tVar shape: {var.shape}\n"
                             f"\tNode shape: {self.var.shape}\n"
                             f"\tIndex Upper bounds: {indices[-1]}")

        indices = list(map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indices))

        res = var[indices] if single else np.asarray([var[idx] for idx in indices]).reshape(out_shape)
        if out_shape == (1,) and len(indices) == 1:
            res = res[0]
        self.domain.set_computed(out_shape, indices)

        return res

    def __add__(self, other):
        return slice_op(operator.add, self, other, graph=self.graph)

    def __radd__(self, other):
        return slice_op(operator.add, other, self, graph=self.graph)

    def __sub__(self, other):
        return slice_op(operator.sub, self, other, graph=self.graph)

    def __rsub__(self, other):
        return slice_op(operator.sub, other, self, graph=self.graph)

    def __pow__(self, other):
        return slice_op(builtins.pow, self, other, graph=self.graph)

    def __rpow__(self, other):
        return slice_op(builtins.pow, other, self, graph=self.graph)

    def __mul__(self, other):
        return slice_op(operator.mul, self, other, graph=self.graph)

    def __rmul__(self, other):
        return slice_op(operator.mul, other, self, graph=self.graph)

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
        return "<var_index name=%s, index=%s>" % (self.name, self.args)

class slice_op(Node):
    """
    Node representing multi-dimensional operations performed on a node.

    Parameters
    ----------
    target : cal
        The multi-dimensional variable used for indexing into.
    idx : tuple
        Tuple of either integer values or index/index_op nodes.
    """
    def __init__(self, target, *args, **kwargs):

        if "domain" in kwargs:
            domain = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
        else:
            all_args = _flatten_iterable(args)
            slice1_var, slice1_idx, slice2_var, slice2_idx = self.get_index_nodes(all_args[0], all_args[1])
            domain = slice1_idx.combine_set_domains(slice2_idx)

        if "op_name" in kwargs:
            kwargs.pop("op_name")

        target_name = f"{target.__module__}.{target.__name__}"
        super(slice_op, self).__init__(*args, target=target_name, domain=domain, op_name=f"slice_{target.__name__}", **kwargs)
        self.target = target


    @property
    def domain(self):
        return self.kwargs["domain"]

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
            if isinstance(key, list):
                key = tuple(key)
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
                        name.append(k)

            else:
                name.append(key)
            name = tuple(name)
            name = self.var.name + str(name)
            if name in self.graph.nodes:
                return self.graph.nodes[name]
            elif isinstance(key, (list)):
                return var_index(self, key, name=name, graph=self.graph)
            elif isinstance(key, tuple):
                return var_index(self, list(key), name=name, graph=self.graph)
            else:
                return var_index(self, [key], name=name, graph=self.graph)

    def set_shape(self, shape=None, init=False, override=False):
        s = []
        assert isinstance(shape, (tuple, list))
        if all([isinstance(sv, Integral) for sv in shape]) and len(self.domain) == np.product(shape) and len(shape) > 0:
            self._shape = shape if isinstance(shape, tuple) else tuple(shape)
        else:

            for idx, d in enumerate(self.domain.dom_set):
                if shape and isinstance(shape[idx], (func_op, Integral)):
                    s.append(shape[idx])
                elif shape and isinstance(shape[idx], float):
                    s.append(int(shape[idx]))
                elif isinstance(d, float):
                    s.append(int(d))
                elif isinstance(d, var_index):
                    s.append(d.domain)
                else:
                    s.append(d)

            self._shape = tuple(s)

    def is_scalar(self, val):
        return not isinstance(val, np.ndarray) or (len(val.shape) == 1 and val.shape[0] == 1)

    def scalar_result(self):
        return False

    def _evaluate(self, op1, op2, context=None, **kwargs):
        if self.is_scalar(op1) or self.is_scalar(op2):
            value = self.target(op1, op2)
        else:
            arg0_dom = self.args[0].domain
            arg1_dom = self.args[1].domain
            op1_idx = self.domain.map_sub_domain(arg0_dom) if isinstance(self.args[0], Node) else tuple([])
            op2_idx = self.domain.map_sub_domain(arg1_dom) if isinstance(self.args[1], Node) else tuple([])
            op1 = np.asarray(list(map(lambda x: op1[x], op1_idx))).reshape(self.domain.computed_shape)
            op2 = np.asarray(list(map(lambda x: op2[x], op2_idx))).reshape(self.domain.computed_shape)
            value = self.target(op1, op2)
        return value



    def get_index_nodes(self, slice1_var=None, slice2_var=None):
        if slice1_var is None and slice2_var is None:
            slice1_var, slice2_var = self.args

        if isinstance(slice1_var, (slice_op, var_index)) or _is_node_type_instance(slice1_var, "GroupNode"):
            slice1_idx = slice1_var.domain
        elif _is_node_type_instance(slice1_var, "index"):
            slice1_idx = slice1_var.domain
        else:
            slice1_idx = Domain(tuple([]))

        if isinstance(slice2_var, (slice_op, var_index)) or _is_node_type_instance(slice2_var, "GroupNode"):
            slice2_idx = slice2_var.domain
        elif _is_node_type_instance(slice2_var, "index"):
            slice2_idx = slice2_var.domain
        else:
            slice2_idx = Domain(tuple([]))
        return slice1_var, slice1_idx, slice2_var, slice2_idx

    def __add__(self, other):
        return slice_op(operator.add, self, other, graph=self.graph)

    def __radd__(self, other):
        return slice_op(operator.add, other, self, graph=self.graph)

    def __sub__(self, other):
        return slice_op(operator.sub, self, other, graph=self.graph)

    def __rsub__(self, other):
        return slice_op(operator.sub, other, self, graph=self.graph)

    def __pow__(self, other):
        return slice_op(builtins.pow, self, other, graph=self.graph)

    def __rpow__(self, other):
        return slice_op(builtins.pow, other, self, graph=self.graph)

    def __mul__(self, other):
        return slice_op(operator.mul, self, other, graph=self.graph)

    def __rmul__(self, other):
        return slice_op(operator.mul, other, self, graph=self.graph)

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
        return "<slice_%s '%s'>" % (self.target.__name__, self.name)


class func_op(Node):  # pylint: disable=C0103,R0903
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
    def __init__(self, target, *args, **kwargs):
        kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs \
            else f"{target.__name__}"
        if "domain" in kwargs:
            domain = tuple(kwargs.pop("domain")) if isinstance(kwargs["domain"], list) else kwargs.pop("domain")
        elif len(args) == 2:
            all_args = _flatten_iterable(args)
            slice1_var, slice1_idx, slice2_var, slice2_idx = self.get_index_nodes(all_args[0], all_args[1])
            domain = slice1_idx.combine_set_domains(slice2_idx)
        else:
            domain = Domain(tuple([]))
        self._target = None
        super(func_op, self).__init__(*args, target=f"{target.__module__}.{target.__name__}", domain=domain, **kwargs)
        self.target = target
        self.added_attrs += ["domain", "target"]

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, fnc):
        self._target = fnc
        self.op_name = f"{fnc.__name__}"
        self.kwargs["target"] = f"{fnc.__module__}.{fnc.__name__}"

    def __getitem__(self, key):
        return self

    @property
    def domain(self):
        return self.kwargs["domain"]

    def get_index_nodes(self, slice1_var=None, slice2_var=None):
        if slice1_var is None and slice2_var is None:
            slice1_var, slice2_var = self.args

        if isinstance(slice1_var, (slice_op, var_index)) or _is_node_type_instance(slice1_var, "GroupNode"):
            slice1_idx = slice1_var.domain
        else:
            slice1_idx = Domain(tuple([]))

        if isinstance(slice2_var, (slice_op, var_index)) or _is_node_type_instance(slice2_var, "GroupNode"):
            slice2_idx = slice2_var.domain
        else:
            slice2_idx = Domain(tuple([]))
        return slice1_var, slice1_idx, slice2_var, slice2_idx

    def _evaluate(self, *args, **kwargs):

        for aa in list(kwargs.keys()):
            if aa in self.added_attrs:
                kwargs.pop(aa)
        return self.target(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return call(self, *args, **kwargs)

    def __repr__(self):
        return "<func_op '%s' target=%s args=<%d items>>" % \
            (self.name, self.kwargs["target"], len(self.args))

def nodeop(target=None, **kwargs):
    """
    Decorator for creating nodes from functions.
    """
    # This is called when the decorator is used with arguments
    if target is None:
        return functools.partial(nodeop, **kwargs)

    # This is called when the decorator is used without arguments
    @functools.wraps(target)
    def _wrapper(*args, **kwargs_inner):
        return func_op(target, *args, **kwargs_inner, **kwargs)
    return _wrapper


@nodeop
def call(func, *args, **kwargs):
    """
    Call `func` with positional arguments `args` and keyword arguments `kwargs`.

    Parameters
    ----------
    func : callable
        Function to call when the node is executed.
    args : list
        Sequence of positional arguments passed to `func`.
    kwargs : dict
        Mapping of keyword arguments passed to `func`.
    """
    return func(*args, **kwargs)

@contextlib.contextmanager
def control_dependencies(dependencies, graph=None):
    """
    Ensure that all `dependencies` are executed before any nodes in this scope.

    Parameters
    ----------
    dependencies : list
        Sequence of nodes to be evaluted before evaluating any nodes defined in this
        scope.
    """
    # Add dependencies to the graph
    graph = Node.get_active_graph(graph)
    graph.dependencies.extend(dependencies)
    yield
    # Remove dependencies from the graph
    del graph.dependencies[-len(dependencies):]

#pylint: disable=C0103
abs_ = nodeop(builtins.abs)
dict_ = nodeop(builtins.dict)
help_ = nodeop(builtins.help)
min_ = nodeop(builtins.min)
setattr_ = nodeop(builtins.setattr)
all_ = nodeop(builtins.all)
dir_ = nodeop(builtins.dir)
hex_ = nodeop(builtins.hex)
next_ = nodeop(builtins.next)
slice_ = nodeop(builtins.slice)
any_ = nodeop(builtins.any)
divmod_ = nodeop(builtins.divmod)
id_ = nodeop(builtins.id)
object_ = nodeop(builtins.object)
sorted_ = nodeop(builtins.sorted)
ascii_ = nodeop(builtins.ascii)
enumerate_ = nodeop(builtins.enumerate)
input_ = nodeop(builtins.input)
oct_ = nodeop(builtins.oct)
staticmethod_ = nodeop(builtins.staticmethod)
bin_ = nodeop(builtins.bin)
eval_ = nodeop(builtins.eval)
int_ = nodeop(builtins.int)
open_ = nodeop(builtins.open)
str_ = nodeop(builtins.str)
bool_ = nodeop(builtins.bool)
exec_ = nodeop(builtins.exec)
isinstance_ = nodeop(builtins.isinstance)
ord_ = nodeop(builtins.ord)
sum_ = nodeop(builtins.sum)
bytearray_ = nodeop(builtins.bytearray)
filter_ = nodeop(builtins.filter)
issubclass_ = nodeop(builtins.issubclass)
pow_ = nodeop(builtins.pow)
super_ = nodeop(builtins.super)
bytes_ = nodeop(builtins.bytes)
float_ = nodeop(builtins.float)
iter_ = nodeop(builtins.iter)
print_ = nodeop(builtins.print)
tuple_ = nodeop(builtins.tuple)
callable_ = nodeop(builtins.callable)
format_ = nodeop(builtins.format)
len_ = nodeop(builtins.len)
property_ = nodeop(builtins.property)
type_ = nodeop(builtins.type)
chr_ = nodeop(builtins.chr)
frozenset_ = nodeop(builtins.frozenset)
list_ = nodeop(builtins.list)
range_ = nodeop(builtins.range)
vars_ = nodeop(builtins.vars)
classmethod_ = nodeop(builtins.classmethod)
getattr_ = nodeop(builtins.getattr)
locals_ = nodeop(builtins.locals)
repr_ = nodeop(builtins.repr)
zip_ = nodeop(builtins.zip)
compile_ = nodeop(builtins.compile)
globals_ = nodeop(builtins.globals)
map_ = nodeop(builtins.map)
reversed_ = nodeop(builtins.reversed)
complex_ = nodeop(builtins.complex)
hasattr_ = nodeop(builtins.hasattr)
max_ = nodeop(builtins.max)
round_ = nodeop(builtins.round)
delattr_ = nodeop(builtins.delattr)
hash_ = nodeop(builtins.hash)
memoryview_ = nodeop(builtins.memoryview)
set_ = nodeop(builtins.set)
add = nodeop(operator.add)
and_ = nodeop(operator.and_)
attrgetter = nodeop(operator.attrgetter)
concat = nodeop(operator.concat)
contains = nodeop(operator.contains)
countOf = nodeop(operator.countOf)
delitem = nodeop(operator.delitem)
eq = nodeop(operator.eq)
floordiv = nodeop(operator.floordiv)
ge = nodeop(operator.ge)
getitem = nodeop(operator.getitem)
gt = nodeop(operator.gt)
index = nodeop(operator.index)
indexOf = nodeop(operator.indexOf)
inv = nodeop(operator.inv)
invert = nodeop(operator.invert)
ior = nodeop(operator.ior)
ipow = nodeop(operator.ipow)
irshift = nodeop(operator.irshift)
is_ = nodeop(operator.is_)
is_not = nodeop(operator.is_not)
itemgetter = nodeop(operator.itemgetter)
le = nodeop(operator.le)
length_hint = nodeop(operator.length_hint)
lshift = nodeop(operator.lshift)
lt = nodeop(operator.lt)
matmul = nodeop(operator.matmul)
methodcaller = nodeop(operator.methodcaller)
mod = nodeop(operator.mod)
mul = nodeop(operator.mul)
ne = nodeop(operator.ne)
neg = nodeop(operator.neg)
not_ = nodeop(operator.not_)
or_ = nodeop(operator.or_)
pos = nodeop(operator.pos)
rshift = nodeop(operator.rshift)
setitem = nodeop(operator.setitem)
sub = nodeop(operator.sub)
truediv = nodeop(operator.truediv)
truth = nodeop(operator.truth)
xor = nodeop(operator.xor)
import_ = nodeop(importlib.import_module)





