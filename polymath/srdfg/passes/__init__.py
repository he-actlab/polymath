import pickle

from polymath.srdfg.base import Node, Graph
from collections import OrderedDict
from collections import deque
import inspect
from polymath.srdfg.util import _flatten_iterable, DebugTimer
import tqdm
import sys
from pytools import ProcessTimer

sys.setrecursionlimit(5000)

KEY_ARGS = ["name", "shape", "graph", "dependencies", "op_name"]
class Visitor(object):

    def __init__(self, debug=False, pbar=None, cls_name=None):
        self.debug = debug
        self.pbar = pbar
        self.memo_map = {}
        self.cls_name = cls_name
        self.graph_stack = deque([None])

    def reset_visitor(self):
        self.memo_map = {}
        self.graph_stack = deque([None])

    def visit(self, graph_object, ctx, node_name=None, **kwargs):
        """Apply the visitor to an expression."""
        if graph_object in self.memo_map:
            return self.memo_map[graph_object]

        if isinstance(graph_object, Graph):
            res = self.visit_graph(graph_object, ctx, **kwargs)
        elif isinstance(graph_object, Node):
            if node_name:
                visitor_name = f"visit_{node_name}"
                assert visitor_name in dir(self)
                res = getattr(self, visitor_name)(graph_object, ctx, **kwargs)
            else:
                res = self.visit_node(graph_object, ctx, **kwargs)
        elif isinstance(graph_object, tuple):
            res = tuple([self.visit(o, ctx, **kwargs) for o in graph_object])
        else:
            res = self.visit_value(graph_object, ctx, **kwargs)
        self.memo_map[graph_object] = res
        return res

    def visit_node(self, _, ctx, **kwargs):
        raise NotImplementedError

    def visit_graph(self, _, ctx, **kwargs):
        raise NotImplementedError

    def visit_value(self, _, ctx, **kwargs):
        raise NotImplementedError

    @classmethod
    def init_from_cls(cls):
        assert cls in (NodeVisitor, NodeTransformer)
        return cls()



class NodeVisitor(Visitor):

    def visit(self, graph_object, ctx, node_name=None, pass_fn=None):
        """Apply the visitor to an expression."""
        if graph_object in self.memo_map:
            return self.memo_map[graph_object]

        if isinstance(graph_object, Graph):
            res = self.visit_graph(graph_object, ctx, pass_fn=pass_fn)
        elif isinstance(graph_object, Node):
            if node_name:
                visitor_name = f"visit_{node_name}"
                assert visitor_name in dir(self)
                res = getattr(self, visitor_name)(graph_object, ctx, pass_fn=pass_fn)
            else:
                pass_fn(graph_object, **ctx)
                res = self.visit_node(graph_object, ctx, pass_fn=pass_fn)
        elif isinstance(graph_object, tuple):
            res = tuple([self.visit(o, ctx, pass_fn) for o in graph_object])
        else:
            res = self.visit_value(graph_object, ctx, pass_fn=pass_fn)

        self.memo_map[graph_object] = res

        return res

    def visit_node(self, node, ctx, **kwargs):
        self.visit(node.name, ctx, **kwargs)
        self.visit(node.op_name, ctx, **kwargs)
        self.visit(node.shape, ctx, **kwargs)
        for dep in node.dependencies:
            self.visit(dep, ctx, **kwargs)
        for arg in node.args:
            self.visit(arg, ctx, **kwargs)
        for _, val in node.kwargs.items():
            self.visit(val, ctx, **kwargs)
        self.visit(node.nodes, ctx, **kwargs)

        return node

    def visit_graph(self, graph, ctx, **kwargs):
        for _, n in graph.items():
            self.visit(n, ctx, **kwargs)
        return graph

    def visit_value(self, val, ctx, **kwargs):
        return val

class NodeTransformer(Visitor):

    def visit(self, graph_object, ctx, node_name=None, pass_fn=None):
        """Apply the visitor to an expression."""
        if graph_object in self.memo_map:
            return self.memo_map[graph_object]

        if isinstance(graph_object, Graph):
            res = self.visit_graph(pickle.loads(pickle.dumps(graph_object)), ctx, pass_fn=pass_fn)
        elif isinstance(graph_object, Node):
            if node_name:
                visitor_name = f"visit_{node_name}"
                assert visitor_name in dir(self)
                res = getattr(self, visitor_name)(graph_object, ctx, pass_fn=pass_fn)
            else:
                res = pass_fn(pickle.loads(pickle.dumps(graph_object)), ctx)
                if isinstance(res, Node):
                    res = self.visit_node(res, ctx, pass_fn=pass_fn)
        elif isinstance(graph_object, tuple):
            res = tuple([self.visit(o, ctx, pass_fn=pass_fn) for o in graph_object])
        else:
            res = self.visit_value(graph_object, ctx, pass_fn=pass_fn)

        self.memo_map[graph_object] = res
        return res

    def visit_node(self, node, ctx, **kwargs):
        node_kwargs = {}

        node_kwargs["name"] = self.visit(node.name, ctx, **kwargs)
        node_kwargs["op_name"] = self.visit(node.op_name, ctx, **kwargs)
        node_kwargs["shape"] = tuple([self.visit(shape_val, ctx, **kwargs) for shape_val in node.shape])
        args = tuple([self.visit(arg, ctx, **kwargs) for arg in node.args])
        node_kwargs["dependencies"] = [self.visit(dep, ctx, **kwargs) for dep in node.dependencies]
        assert all([a is not None and not isinstance(a, (tuple, list)) for a in _flatten_iterable(args)])
        for k, v in node.kwargs.items():
            if k in KEY_ARGS:
                continue
            node_kwargs[k] = self.visit(v, ctx, **kwargs)
        if "target" in node_kwargs:
            node_kwargs.pop("target")

        if node.__class__.__name__ in ["func_op", "slice_op"]:
            args = (node.target,) + args
        new_node = node.__class__(*args,
                                  graph=self.graph_stack[-1],
                                  value=node.value,
                                  **node_kwargs
                                  )

        self.graph_stack.append(new_node)
        _ = self.visit(node.nodes, ctx, **kwargs)
        return self.graph_stack.pop()

    def visit_graph(self, graph, ctx, **kwargs):
        graph_iter = graph.copy()
        for name, n in graph_iter.items():
            _ = self.visit(graph[name], ctx, **kwargs)
        return self.graph_stack[-1].nodes

    def visit_value(self, val, ctx, **kwargs):
        return val

class NodePass(Visitor):
    def visit(self, graph_object, ctx, node_name=None, pass_fn=None):
        """Apply the visitor to an expression."""
        # hash_val = hash(graph_object) if not isinstance(graph_object, Node) else graph_object.func_hash()
        hash_val = hash(graph_object)
        if hash_val in self.memo_map:
            return self.memo_map[hash_val]

        if isinstance(graph_object, Graph):
            self.visit_graph(graph_object, ctx, pass_fn=pass_fn)
        elif isinstance(graph_object, Node):
            if self.debug and graph_object.graph:
                self.pbar.set_description(f"{self.cls_name}: Applying {pass_fn.__name__} pass to node {graph_object.name} - {graph_object.op_name}")
                self.pbar.update(n=1)
            pass_fn(graph_object, ctx)
            self.visit_node(graph_object, ctx, pass_fn=pass_fn)
        else:
            self.visit_value(graph_object, ctx, pass_fn=pass_fn)
        # new_hash_val = hash(graph_object) if not isinstance(graph_object, Node) else graph_object.func_hash()
        new_hash_val = hash(graph_object)
        self.memo_map[new_hash_val] = graph_object

        return graph_object

    def visit_node(self, node, ctx, **kwargs):
        self.visit(node.nodes, ctx, **kwargs)
        return node

    def visit_graph(self, graph, ctx, **kwargs):

        keys = list(graph.keys())
        _ = [self.visit(graph[name], ctx, **kwargs) for name in keys if name in graph]
        return graph

    def visit_value(self, val, ctx, **kwargs):
        return val


class Pass(object):
    analyzer = None
    def __init__(self, ctx=None, debug=False):
        self.ctx = ctx
        self.debug = debug

    def __call__(self, node, ctx=None, debug=False):
        self.pbar = tqdm.tqdm(desc=f"Applying {self.__class__.__name__} to nodes", file=sys.stdout, dynamic_ncols=True,
                              disable=not self.debug)

        visitor = NodePass(debug=self.debug, pbar=self.pbar, cls_name=self.__class__.__name__)
        self.ctx = self.ctx if self.ctx else ctx

        gcpy = pickle.loads(pickle.dumps(node))

        initialized_node = self.initialize_pass(gcpy, self.ctx)
        if self.debug:
            ncount = self.total_nodes(initialized_node, 0)

            self.pbar.reset(total=ncount)

        transformed_node = visitor.visit(initialized_node, self.ctx, pass_fn=self.apply_pass)
        if self.debug:
            ncount = self.total_nodes(transformed_node, 0)
            self.pbar.clear()
            self.pbar.reset(total=ncount)
        visitor.reset_visitor()
        tcpy = pickle.loads(pickle.dumps(transformed_node))
        final_node = visitor.visit(tcpy, self.ctx, pass_fn=self.finalize_pass)
        packaged_node = self.package_pass(final_node, self.ctx)
        self.pbar.close()
        return packaged_node

    def evaluate_node(self, node, context):
        res = Node.evaluate_node(node, context)
        node.graph.nodes[node.name].value = res
        return res

    def total_nodes(self, graph, count):
        count += len(graph.nodes)
        for _, n in graph.nodes.items():
            count = self.total_nodes(n, count)
        return count

    @classmethod
    def init_from_cls(cls):
        return cls()

    def initialize_pass(self, node, ctx):
        return node

    def apply_pass(self, node, ctx):
        return node

    def finalize_pass(self, node, ctx):
        return node

    def package_pass(self, node, ctx):
        return node

def _wrap_node_pass(apply_func):
    class _NodePass(Pass):
        def __init__(self):
            super(_NodePass, self).__init__()
    _NodePass.__name__ = apply_func.__name__
    apply_func.__name__ = _NodePass.apply_pass.__name__
    return _NodePass()

class PassRegistry(object):
    _registry = OrderedDict()

    def register(self, pss_func=None, init_pass=None, fin_pass=None, analysis=False, name=None):

        def create_pass(pass_fn):
            pass_name = name if name else pass_fn.__name__
            if inspect.isfunction(pass_fn):
                pss = _wrap_node_pass(pass_fn)
                pss.apply_pass = pass_fn
                pss.initialize_pass = init_pass or pss.initialize_pass
                pss.finalize_func = fin_pass or pss.finalize_pass
                pss_info = (pss, analysis)
            else:
                assert issubclass(pass_fn, Pass)
                pss = pass_fn

                pss_info = (pass_fn, analysis)
            self._registry[pass_name] = pss_info
            return pss

        if pss_func:
            return create_pass(pss_func)
        return create_pass

    def is_registered(self, clss):
        return clss in self._registry.keys()

    def run_passes(self, node):
        info = {}
        for pss_name, pss in self._registry.items():
            node, res = pss[0]()(node)
            info.update(res)

        return node, info


pass_registry = PassRegistry()
del PassRegistry
register_pass = pass_registry.register
