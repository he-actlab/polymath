# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
# Copyright 2017 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import contextlib
import logging
import math
import time
from graphviz import Digraph
import numpy as np
import inspect
from itertools import product
from collections import deque, OrderedDict
import pickle
from pytools import ProcessTimer
from pathlib import Path
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from polymath.srdfg.base import Node
LOGGER = logging.getLogger(__name__)


class lazy_import:  # pylint: disable=invalid-name, too-few-public-methods
    """
    Lazily import the given module.
    Parameters
    ----------
    module : str
        Name of the module to import
    """
    def __init__(self, module):
        self.module = module
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            self._module = __import__(self.module)
        return getattr(self._module, name)

class batch_iterable:  # pylint: disable=invalid-name, too-few-public-methods
    """
    Split an iterable into batches of a specified size.
    Parameters
    ----------
    iterable : iterable
        Iterable to split into batches.
    batch_size : int
        Size of each batch.
    transpose : bool
        Whether to transpose each batch.
    """
    def __init__(self, iterable, batch_size, transpose=False):
        self.iterable = iterable
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive but got '%s'" % batch_size)
        self.batch_size = batch_size
        self.transpose = transpose

    def __len__(self):
        return math.ceil(len(self.iterable) / self.batch_size)

    def __iter__(self):
        batch = []
        for item in self.iterable:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield tuple(zip(*batch)) if self.transpose else batch
                batch = []
        if batch:
            yield tuple(zip(*batch)) if self.transpose else batch


class Profiler:  # pylint: disable=too-few-public-methods
    """
    Callback for profiling computational graphs.
    Attributes
    ----------
    times : dict[Operation, float]
        Mapping from nodes to execution times.
    """
    def __init__(self):
        self.times = {}

    def get_slow_operations(self, num_operations=None):
        """
        Get the slowest nodes.
        Parameters
        ----------
        num_operations : int or None
            Maximum number of nodes to return or `None`
        Returns
        -------
        times : collections.OrderedDict
            Mapping of execution times keyed by nodes.
        """
        items = list(sorted(self.times.items(), key=lambda x: x[1], reverse=True))
        if num_operations is not None:
            items = items[:num_operations]
        return collections.OrderedDict(items)

    @contextlib.contextmanager
    def __call__(self, operation, context):
        start = time.time()
        yield
        self.times[operation] = time.time() - start

    def __str__(self):
        return "\n".join(['%s: %s' % item for item in self.get_slow_operations(10).items()])


@contextlib.contextmanager
def _noop_callback(*_):
    yield


def deprecated(func):  # pragma: no cover
    """
    Mark a callable as deprecated.
    """
    def _wrapper(*args, **kwargs):
        LOGGER.warning("%s is deprecated", func)
        return func(*args, **kwargs)
    return _wrapper




def _is_node_instance(node):
    classes = inspect.getmro(node.__class__)
    for cls in classes:
        if cls.__name__ == "Node":
            return True
    return False

def _is_node_type_instance(node, ntype):
    classes = inspect.getmro(node.__class__)
    for cls in classes:
        name = cls.__name__
        if isinstance(ntype, (tuple, list)) and name in ntype:
            return True
        elif name == ntype:
            return True
    return False


def visualize(graph: 'Node', filepath=None):
    out_graph = Digraph(graph.name)
    print(graph.name)
    added_nodes = []
    for name in graph.nodes.keys():
        if name == graph.name:
            continue
        node = graph.nodes[name]
        out_graph.node(name, label=f"Op:{node.op_name}\nname:{name}")
        added_nodes.append(name)
        all_args = _flatten_iterable(node.args)
        for arg in all_args:
            if _is_node_instance(arg):
                if arg.name not in added_nodes:
                    print(f"whoops: node: {node.name}\t arg:{arg.name}")
                out_graph.edge(arg.name, name)

            else:
                added_nodes.append(str(hash(arg)))
                out_graph.node(str(hash(arg)), label=str(arg))
                out_graph.edge(str(hash(arg)), name)
    if filepath:
        name = f"{filepath}/{graph.name}"
    else:
        name = f"{Path(__file__).parent}/{graph.name}"
    out_graph.render(name, view=False)

def _scope_name(scope_stack):
    name = []
    for n in scope_stack:
        if n:
            var_name = n.name.rsplit('/', 1)
            if var_name[-1] not in name:
                name.append(var_name[-1])
    return "/".join(name)

def _get_scalar_from_placeholder(comb_vals, is_shape=False):
    if comb_vals == tuple([0]):
        return []
    elif is_shape:
        if not all([isinstance(s, int) for s in comb_vals]) or comb_vals == tuple([0]):
            return []
        cmb = [[i for i in range(s)] for s in comb_vals]
    else:
        cmb = comb_vals
    res = list(product(*tuple(cmb)))

    return res

def debug_dif_print(attr_name, attr1, attr2):
    print(f"{attr_name} unequal for nodes:"
          f"\n\t{attr_name}1: {attr1}"
          f"\n\t{attr_name}2: {attr2}")

def _dif_hash(node1, node2, ctx):
    if _is_node_instance(node1):
        assert _is_node_instance(node2)
        if id(node1) in ctx:
            assert id(node2) in ctx
            return (ctx[id(node1)], ctx[id(node2)])
        assert node1.name not in node1.nodes
        assert node2.name not in node2.nodes
        shape_hash1 = hash(tuple([_fnc_hash(shape, ctx) for shape in node1.shape]))
        shape_hash2 = hash(tuple([_fnc_hash(shape, ctx) for shape in node2.shape]))
        if shape_hash1 != shape_hash2:
            debug_dif_print("shape", node1.shape, node2.shape)
        kwarg_hash1 = hash(tuple([(k, _fnc_hash(v, ctx)) for k, v in node1.kwargs.items()]))
        kwarg_hash2 = hash(tuple([(k, _fnc_hash(v, ctx)) for k, v in node2.kwargs.items()]))
        if kwarg_hash1 != kwarg_hash2:
            debug_dif_print("kwargs", node1.kwargs, node2.kwargs)

        arg_hash1 = hash(tuple([_fnc_hash(arg, ctx) for arg in _flatten_iterable(node1.args)]))
        arg_hash2 = hash(tuple([_fnc_hash(arg, ctx) for arg in _flatten_iterable(node2.args)]))
        if arg_hash1 != arg_hash2:
            debug_dif_print("args", node1.args, node2.args)
        graph_hash1 = []
        graph_hash2 = []
        for k, n in node1.nodes.items():

            n_h1 = _fnc_hash(n, ctx)
            n_h2 = _fnc_hash(node2.nodes[k], ctx)
            if n_h1 != n_h2:
                debug_dif_print(f"Node {k}", f"{n.name} - {n.op_name}", f"{node2.nodes[k].name} - {node2.nodes[k].op_name}")
            graph_hash1.append(n_h1)
            graph_hash2.append(n_h2)
        if node1.dependencies != node2.dependencies:
            debug_dif_print("Deps", node1.dependencies, node2.dependencies)

        if node1.op_name != node2.op_name:
            debug_dif_print("Op_name", node1.op_name, node2.op_name)

        ctx[id(node1)] = hash((arg_hash1,
                              shape_hash1,
                              node1.op_name,
                              tuple(node1.dependencies),
                              kwarg_hash1,
                              hash(tuple(graph_hash1))))

        ctx[id(node2)] = hash((arg_hash2,
                              shape_hash2,
                              node2.op_name,
                              tuple(node2.dependencies),
                              kwarg_hash2,
                              hash(tuple(graph_hash2))))
        return (ctx[id(node1)], ctx[id(node2)])
    elif isinstance(node1, (list, tuple)):
        if len(node1) == 0:
            assert len(node2) == 0
            return (node1, node2)
        else:
            return (tuple([_fnc_hash(n, ctx) for n in node1]), tuple([_fnc_hash(n, ctx) for n in node2]))
    elif isinstance(node1, dict):
        return (tuple([(k, _fnc_hash(v, ctx)) for k, v in node2.items()]), tuple([(k, _fnc_hash(v, ctx)) for k, v in node2.items()]))
    elif isinstance(node1, (slice, np.ndarray)):
        return (str(node1), str(node2))
    else:
        return node1, node2

def _fnc_hash(node, ctx):

    if _is_node_instance(node):
        if id(node) in ctx:
            return ctx[id(node)]
        assert node.name not in node.nodes

        shape_hash = hash(tuple([_fnc_hash(shape, ctx) for shape in node.shape]))
        kwarg_hash = hash(tuple([(k, _fnc_hash(v, ctx)) for k, v in node.kwargs.items()]))

        arg_hash = hash(tuple([_fnc_hash(arg, ctx) for arg in _flatten_iterable(node.args)]))
        graph_hash = []
        for _, n in node.nodes.items():
            n_h = _fnc_hash(n, ctx)
            graph_hash.append(n_h)
        ctx[id(node)] = hash((arg_hash,
                     shape_hash,
                     node.op_name,
                     tuple(node.dependencies),
                     kwarg_hash,
                     hash(tuple(graph_hash))))
        return ctx[id(node)]
    elif isinstance(node, (list, tuple)):
        if len(node) == 0:
            return node
        else:
            return tuple([_fnc_hash(n, ctx) for n in node])
    elif isinstance(node, dict):
        return tuple([(k, _fnc_hash(v, ctx)) for k, v in node.items()])
    elif isinstance(node, (slice, np.ndarray)):
        return str(node)
    else:
        return node

def update_graph_args(graph, hashes):
    iter_nodes = graph.nodes.copy()
    for k, v in iter_nodes.items():
        graph.nodes[k].args = _update_args(graph.nodes[k].args, hashes)


def _update_args(args, hashes):
    new_args = []
    for a in args:
        if _is_node_instance(a):
            a_hash = _fnc_hash(a, {})
            if a_hash in hashes:
                new_args.append(hashes[a_hash])
            else:
                new_args.append(a)
        elif isinstance(a, tuple):
            new_args.append(_update_args(a, hashes))
        else:
            new_args.append(a)
    return tuple(new_args)


def squeeze_shape(shape):
    assert isinstance(shape, tuple)
    if len(shape) > 1:
        return tuple(s for s in shape if s >1)
    else:
        return shape

def squeeze_indices(idx_vals, shape):
    indices = []
    for i in idx_vals:
        if _is_node_instance(i):
            val = i.value
        else:
            val = i

        if not is_iterable(val):
            indices.append(np.asarray([val]))
        else:
            indices.append(val)

        if len(shape) != len(idx_vals) and len(indices[-1]) == 1 and indices[-1][0] == 0:
            indices.pop()

    return indices

def extend_indices(idx_vals, shape):
    indices = []
    shape_idx = 0
    slice_idx = 0
    while shape_idx < len(shape):
        if shape[shape_idx] == 1 and len(shape) > len(idx_vals):
            indices.append(np.array([0]))
            shape_idx += 1
        else:
            if _is_node_instance(idx_vals[slice_idx]):
                val = idx_vals[slice_idx].value
            else:
                val = idx_vals[slice_idx]

            if not is_iterable(val):
                indices.append(np.asarray([val]))
            else:
                indices.append(val)
            shape_idx += 1
            slice_idx += 1

    return indices

def get_indices(idx_vals):
    indices = []
    for i in idx_vals:
        if _is_node_instance(i):
            val = i.value
        else:
            val = i

        if not is_iterable(val):
            indices.append(np.asarray([val]))
        else:
            indices.append(val)

    return np.asarray(indices)

def lower_graph(graph, supported_ops):
    var_hashes = {}
    graph_copy = pickle.loads(pickle.dumps(graph))
    scope_stack = deque([])
    scope_stack.append(graph_copy.graph)
    scope_stack.append(graph_copy)
    graph_copy.lower(scope_stack, supported_ops, var_hashes)
    return graph_copy

def _linear_subscript(shape, indices):
    if isinstance(indices, int):
        indices = tuple([indices])
    if len(shape) == 1:
        return indices[0]*shape[0]
    return indices[0] + shape[0]*_linear_subscript(shape[1:], indices[1:])

def _kwarg_hash(k, v):
    if isinstance(v, (slice, np.ndarray)):
        return (k, str(v))
    elif isinstance(v, (list, tuple)):
        vv = tuple([_arg_hash(v) for v in v])
        return tuple([k, vv])
    elif isinstance(v, dict):
        return tuple([_kwarg_hash(kk, vv) for kk, vv in v.items()])
    else:
        return (k,v)

def _arg_hash(arg):
    if isinstance(arg, (slice, np.ndarray)):
        return str(arg)
    elif isinstance(arg, list):
        return tuple(arg)
    elif isinstance(arg, dict):
        return tuple([(k, v) for k,v in arg.items()])
    else:
        return arg

def flatten(items):
    if _is_node_instance(items):
        yield items
    for x in items:
        if isinstance(x, (tuple, list)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def _flatten_iterable(items):
    if isinstance(items, np.ndarray):
        return items
    elif isinstance(items, (list, tuple)):
        return tuple(list(flatten(items)))
    else:
        return tuple(list(flatten([items])))

def _node_hash(node):
    shape_hash = tuple([hash(shape) for shape in node.shape])
    kwarg_hash = tuple([_kwarg_hash(k, v) for k, v in node.kwargs.items()])
    arg_hash = tuple([_arg_hash(arg) for arg in flatten(node.args)])
    return hash((arg_hash,
                 node.name,
                 shape_hash,
                 node.op_name,
                 tuple(node.dependencies),
                 kwarg_hash,
                 node.nodes))

def arg_hash(arg):
    if isinstance(arg, (slice, np.ndarray)):
        return str(arg)
    elif _is_node_instance(arg):
        return node_hash(arg)
    elif isinstance(arg, list):
        return tuple(arg)
    elif isinstance(arg, dict):
        return tuple([(k, v) for k,v in arg.items()])
    else:
        return arg

def kwarg_hash(k, v):
    if isinstance(v, (slice, np.ndarray)):
        return (k, str(v))
    elif _is_node_instance(v):
        return (k, node_hash(v))
    elif isinstance(v, (list, tuple)):
        vv = tuple([arg_hash(v) for v in v])
        return tuple([k, vv])
    elif isinstance(v, dict):
        return tuple([kwarg_hash(kk, vv) for kk, vv in v.items()])
    else:
        return (k,v)

def node_hash(node):
    shape = tuple([arg_hash(shape) for shape in node.shape])
    kwargs = tuple([kwarg_hash(k, v) for k, v in node.kwargs.items()])
    args = tuple([arg_hash(arg) for arg in flatten(node.args)])
    nodes = tuple([node_hash(n) for _, n in node.nodes.items()])
    return hash((args,
                 node.name,
                 shape,
                 node.op_name,
                 tuple(node.dependencies),
                 kwargs,
                 nodes))

def _compute_domain_pairs(domain):
    dom_pairs = []
    for i in domain:
        if _is_node_instance(i):
            if i.value is not None:
                dom_pairs.append(i.value)
            elif _is_node_type_instance(i, "index"):
                assert isinstance(i.lbound, int) and isinstance(i.ubound, int)
                dom_pairs.append([x for x in range(i.lbound, i.ubound + 1)])
            elif _is_node_type_instance(i, "index_op"):
                # np.array(list(product(*dom_pairs)))
                print(f"Cannot handle index op for computing domain pairs yet")
                # dom_pairs.append([x for x in range(i.lbound, i.ubound + 1)])
            else:
                raise ValueError(f"Could not use subscript for domain pair: {i.name} - {i.op_name}")
        elif isinstance(i, np.ndarray):
            dom_pairs.append(i.tolist())
        else:
            assert isinstance(i, (tuple, list))
            dom_pairs.append(i)
    dom_pairs = tuple(dom_pairs)
    return [tuple(i) for i in np.array(list(product(*dom_pairs)))]

class DebugTimer(object):

    def __init__(self, verbose=False):
        self.ptimer = ProcessTimer()
        self.verbose = verbose


    def __enter__(self):
        self.ptimer = ProcessTimer()
        self.ptimer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ptimer.__exit__(exc_type, exc_val, exc_tb)
        if self.verbose:
            print(f"Process elapsed: {self.ptimer.process_elapsed}")
            print(f"Wall elapsed: {self.ptimer.wall_elapsed}")

    @property
    def proc_elapsed(self):
        return self.ptimer.process_elapsed

    @property
    def wall_elapsed(self):
        return self.ptimer.wall_elapsed



def is_iterable(obj):
    return hasattr(obj, '__iter__') and not (isinstance(obj, str) or _is_node_instance(obj))


def compute_sum_indices(axes_idx, input_domain, sd):
    return (input_domain[:, axes_idx] == sd).all(axis=1).nonzero()[0]


def is_single_valued(node, value=None):
    if _is_node_instance(node) and node.shape in [(1,), (0,)]:
        return True
    elif value:
        if not isinstance(value, np.ndarray):
            return True
        elif is_iterable(value) and len(value) == 1:
            return True
    return False

