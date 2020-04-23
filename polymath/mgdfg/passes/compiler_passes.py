from polymath.mgdfg.passes import register_pass, Pass
from polymath.mgdfg.util import _flatten_iterable, is_iterable, compute_sum_indices
import polymath as pm
from numbers import Integral
from collections import defaultdict
from itertools import product
import numpy as np
import time
from math import ceil

@register_pass(analysis=True)
class CountNodes(Pass):
    def __init__(self):
        init_info = {"count": 0, "global": 0}
        super(CountNodes, self).__init__(init_info)

    def apply_pass(self, node, counts):
        if node.graph:
            if node.graph.name in counts.keys():
                counts[node.graph.name] += 1
            else:
                counts[node.graph.name] = 1
        else:
            counts["global"] += 1
        counts["count"] += 1
        return node

@register_pass
class DeadNodeElimination(Pass):


    def apply_pass(self, node, ctx):
        if node.op_name == "index":
            node.graph.pop(node.name)
            return node
        elif node.op_name == "var_index":

            if node.args[0].op_name == "placeholder":
                if node.args[0].name in node.args[0].graph.nodes:
                    node.args[0].graph.pop(node.args[0].name)
                new_node = pm.placeholder(name=node.name, type_modifier=node.args[0].kwargs["type_modifier"],
                                          shape=(1,))
                node.graph.nodes[node.name] = new_node
                new_node.graph = node.graph
                return new_node
            else:
                return node
        elif isinstance(node, (pm.func_op, pm.slice_op, pm.NonLinear)):
            new_args = []
            for i, a in enumerate(node.args):
                if isinstance(a, pm.Node):
                    assert a.name in node.graph.nodes
                    new_args.append(node.graph.nodes[a.name])
                elif isinstance(a, tuple):
                    for ii, aa in a:

                        if isinstance(aa, pm.Node):
                            continue
                else:
                    new_args.append(a)

            node.args = tuple(new_args)
            return node
        else:
            new_args = []
            for i, a in enumerate(node.args):

                if isinstance(a, pm.Node):
                    assert a.name in node.graph.nodes
                    new_args.append(node.graph.nodes[a.name])
                elif isinstance(a, tuple):
                    for aa in a:
                        if isinstance(aa, pm.Node):
                            continue
                else:
                    new_args.append(a)
            node.args = tuple(new_args)

            return node

@register_pass
class NormalizeGraph(Pass):
    def __init__(self, stored_shapes):
        self.context = self._check_input_shapes(stored_shapes) if stored_shapes else {}
        if "populate" in stored_shapes:
            self.populate = stored_shapes["populate"]
        else:
            self.populate = True

        self.evaluated = {}
        self.stored_objects = {}
        self.var_indices = {}
        self.output_shapes = {}
        self.output_shape_deps = defaultdict(list)
        self.delayed_evals = {}
        super(NormalizeGraph, self).__init__({"shapes": {}})

    def update_args(self, args):
        new_args = []
        for a in args:
            if isinstance(a, pm.Node) and not isinstance(a, (pm.input, pm.state, pm.output)):
                if a.name in self.context:
                    new_args.append(self.context[a.name])
                elif a in self.evaluated:
                    new_args.append(self.evaluated[a])
                else:
                    new_args.append(a)
            elif isinstance(a, tuple):
                new_args.append(self.update_args(a))
            else:
                new_args.append(a)
        return tuple(new_args)

    def initialize_pass(self, graph, _):
        for k in list(self.context.keys()):
            if k in graph.nodes:
                nshape = graph.nodes[k].shape
                for i, s in enumerate(nshape):
                    if isinstance(s, pm.Node):
                        self.context[s.name] = self.context[k][i]
        return graph

    def apply_pass(self, node, info):
        new_shape = []
        if isinstance(node, pm.var_index):
            shape = node.args[1]
        elif isinstance(node, pm.NonLinear):
            shape = node.args[0].shape
        else:
            shape = node.shape

        if not isinstance(node, pm.output):

            for s in shape:
                if isinstance(s, pm.Node):
                    assert s.name in self.context or s in self.evaluated
                    if s.name in self.context:
                        new_shape.append(self.context[s.name])
                    elif isinstance(s, pm.index):
                        sval = self.evaluated[s][-1] - self.evaluated[s][0] + 1
                        new_shape.append(sval)
                    else:
                        new_shape.append(self.evaluated[s])
                else:
                    assert isinstance(s, Integral)
                    new_shape.append(s)

            node._shape = tuple(new_shape)
            node._args = self.update_args(node.args)


            if isinstance(node, pm.write) and isinstance(node.dest, pm.output):
                node.dest._shape = node._shape

            eval_ready = all([not isinstance(i, pm.Node) for i in _flatten_iterable(node.args)])
            if eval_ready and isinstance(node, (pm.func_op, pm.index)):
                res = node._evaluate(*node.args, **node.kwargs)
                node.value = res
                self.evaluated[node] = res
                node.graph.nodes.pop(node.name)
        return node

    def finalize_pass(self, node, info):

        if not node.is_shape_finalized() and node.shape != (0,):
            raise ValueError(f"Shape not finalized during first iteration for "
                             f"{node.op_name} - {node.name}:\n\t"
                             f"{node.shape}\n\t"
                             f"{node.args}")
        if not self.populate:
            return node
        if id(node) in self.stored_objects:
            return self.stored_objects[id(node)]
        if isinstance(node, pm.GroupNode):
            self.populate_group_op(node)
        elif isinstance(node, pm.NonLinear):
            self.populate_nonlinear(node)
        elif isinstance(node, pm.slice_op):
            self.populate_slice_op(node)
        elif isinstance(node, pm.input):
            self.populate_input(node)
        elif isinstance(node, pm.state):
            self.populate_state(node)
        elif isinstance(node, pm.output):
            self.populate_output(node)
        elif isinstance(node, pm.write):
            self.populate_write(node)
        elif isinstance(node, pm.placeholder):
            self.populate_placeholder(node)
        elif isinstance(node, pm.var_index):
            self.populate_var_index(node)
        elif isinstance(node, pm.func_op):
            self.populate_func_op(node)

        return node

    def populate_func_op(self, node):
        pass

    def populate_nonlinear(self, node):

        indices = list(product(*tuple([np.arange(i) for i in node.args[0].shape])))

        if len(indices) > 1:
            for i in indices:
                x = node.__class__.init_from_args(node.args[0][i], graph=node, name=f"{node.name}{i}", shape=(1,))
                self.stored_objects[id(x)] = x

    def populate_slice_op(self, node):

        op1_idx = node.domain.map_sub_domain(node.args[0].domain) if isinstance(node.args[0], pm.Node) and node.args[0].shape != (0,) else tuple([])
        op2_idx = node.domain.map_sub_domain(node.args[1].domain) if isinstance(node.args[1], pm.Node) and node.args[1].shape != (0,) else tuple([])


        dom_pairs = node.domain.compute_pairs()
        kwargs = {}
        kwargs["op_name"] = node.target.__name__
        kwargs["graph"] = node
        assert len(dom_pairs) == len(op1_idx) or len(op1_idx) == 0
        assert len(dom_pairs) == len(op2_idx) or len(op2_idx) == 0
        if len(op1_idx) > 0:
            ops1 = list(map(lambda x: node.args[0][op1_idx[x]], range(len(dom_pairs))))
        else:
            ops1 = list(map(lambda x: node.args[0], range(len(dom_pairs))))


        if len(op2_idx) > 0:
            ops2 = list(map(lambda x: node.args[1][op2_idx[x]], range(len(dom_pairs))))
        else:
            ops2 = list(map(lambda x: node.args[1], range(len(dom_pairs))))
        for p, v in enumerate(dom_pairs):
            kwargs["name"] = f"{node.name}{v}"
            x = pm.func_op.init_from_args(node.target, ops1[p], ops2[p], **kwargs)
            self.stored_objects[id(x)] = x

    def populate_var_index(self, node):
        indices = np.asarray([i.value if isinstance(i, pm.Node) else i for i in node.args[1]])
        indices = np.array(list(product(*indices)))
        indices = list(map(lambda x: tuple(x), indices))
        out_shape = node.domain.shape_from_indices(node.args[1])
        node._shape = out_shape
        dom_pairs = node.domain.compute_pairs()
        dom_pairs = list(map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, dom_pairs))
        node.domain.set_computed(out_shape, dom_pairs)

        for i, d in enumerate(dom_pairs):
            ph_node = node.var[indices[i]]
            name = f"{node.var.name}{d}"
            node.nodes[name] = ph_node

    def populate_placeholder(self, node):
        if node.shape != (0,):
            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                x = pm.placeholder(graph=node, name=f"{node.name}{i}", shape=(1,), type_modifier=node.type_modifier)
                self.stored_objects[id(x)] = x

    def populate_output(self, node):
        if node.shape != (0,):
            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                x = pm.output(graph=node, name=f"{node.name}{i}", shape=(1,))
                self.stored_objects[id(x)] = x

    def populate_input(self, node):
        if node.shape != (0,):
            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                x = pm.input(graph=node, name=f"{node.name}{i}", shape=(1,))
                self.stored_objects[id(x)] = x

    def populate_state(self, node):
        if node.shape != (0,):
            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                x = pm.state(graph=node, name=f"{node.name}{i}", shape=(1,))
                self.stored_objects[id(x)] = x

    def populate_write(self, node):

        if node.shape != (1,):
            src, dst_key, dst = node.args
            dst_indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            key_indices = node.domain.compute_pairs()

            if isinstance(src, pm.Node):
                src_indices = node.domain.map_sub_domain(node.args[0].domain)
                for i in dst_indices:
                    if i in key_indices:
                        idx = key_indices.index(i)
                        dst_node = node.graph.nodes[f"{dst.name}"][i]
                        state_var = node.graph.nodes[f"{dst.alias}"][i]
                        val = pm.write(src[src_indices[idx]], i, dst_node, graph=node, alias=state_var.name,
                                       name=f"{state_var.name}{state_var.write_count}")
                        state_var.write_count += 1
                        self.stored_objects[id(val)] = val
                    else:
                        val = dst[i]
                        node.nodes[val.name] = val
            else:
                assert not is_iterable(src)
                for i in dst_indices:
                    if i in key_indices:
                        dst_node = node.graph.nodes[f"{dst.name}"][i]
                        state_var = node.graph.nodes[f"{dst.alias}"][i]
                        val = pm.write(src, i, dst_node, graph=node, alias=state_var.name,
                                       name=f"{state_var.name}{state_var.write_count}")
                        state_var.write_count += 1
                        self.stored_objects[id(val)] = val
                    else:
                        val = dst[i]
                        node.nodes[val.name] = val


    def populate_group_op(self, node):
        if len(node.domain) == 0:
            input_domain = node.input_node.domain.compute_pairs()
            inputs = []
            for d in input_domain:
                if isinstance(node.input_node, (pm.var_index, pm.slice_op, pm.placeholder, pm.GroupNode, pm.NonLinear)):
                    i = node.input_node[d]
                else:
                    i = node.input_node
                node.nodes[i.name] = i
                inputs.append(i)
            if node.graph:
                div = ceil(len(inputs)/2)
                x = self._div_conquer_reduce(inputs[0:div], inputs[div:], node)
                node.output_nodes.append(x)
                self.stored_objects[id(x)] = x
            node._shape = (1,)
        else:
            # TODO: Need to make this faster and make sure object ids are added
            input_domain = node.input_node.domain.compute_pairs(tuples=False)
            sum_domain = node.domain.compute_pairs(tuples=False)
            axes_idx = np.array([node.input_node.domain.index(s) for s in node.domain])
            make_tuple = lambda x: x if isinstance(x, tuple) else (tuple(x) if isinstance(x, (list)) else (tuple(x.tolist()) if isinstance(x, np.ndarray) else x))
            sd_map = {str(sd): compute_sum_indices(axes_idx, input_domain, sd) for sd in sum_domain}
            assert node.graph
            for k, v in sd_map.items():
                inputs = []
                for d in v:
                    i = node.input_node[make_tuple(input_domain[d])]
                    node.nodes[i.name] = i
                    inputs.append(i)
                x = self._div_conquer_reduce(inputs[0:len(inputs) // 2], inputs[len(inputs) // 2:], node)
                x.set_name(f"{node.name}{k}")
                node.output_nodes.append(x)
                self.stored_objects[id(x)] = x
            node._shape = [node.input_node.shape[i] for i in sorted(axes_idx)]


    def _check_input_shapes(self, input_shapes):
        assert isinstance(input_shapes, dict)
        shapes = {}
        for k, v in input_shapes.items():
            if isinstance(v, np.ndarray):
                shapes[k] = v.shape
            elif not isinstance(v, tuple):
                assert isinstance(v, Integral)
                shapes[k] = v
            else:
                shapes[k] = v
        return shapes

    def _div_conquer_reduce(self, left, right, node):
        kwargs = {}
        kwargs["graph"] = node
        kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs else node.scalar_target.__name__
        if len(left) == 1 and len(right) == 1:
            kwargs["name"] = f"{node.name}{len(node.nodes)}"
            return pm.func_op.init_from_args(node.scalar_target, left[0], right[0], **kwargs)
        elif len(left) == 1:
            div = ceil(len(right)/2)
            right_op = self._div_conquer_reduce(right[0:div], right[div:], node)
            kwargs["name"] = f"{node.name}{len(node.nodes)}"
            return pm.func_op.init_from_args(node.scalar_target, left[0], right_op, **kwargs)
        elif len(right) == 1:
            div = ceil(len(left)/2)

            left_op = self._div_conquer_reduce(left[0:div], left[div:], node)
            kwargs["name"] = f"{node.name}{len(node.nodes)}"
            return pm.func_op.init_from_args(node.scalar_target, left_op, right[0], **kwargs)
        else:
            ldiv = ceil(len(left)/2)
            left_op = self._div_conquer_reduce(left[0:ldiv], left[ldiv:], node)
            kwargs["name"] = f"{node.name}{len(node.nodes)}"
            rdiv = ceil(len(right)/2)
            right_op = self._div_conquer_reduce(right[0:rdiv], right[rdiv:], node)
            kwargs["name"] = f"{node.name}{len(node.nodes)}"
            return pm.func_op.init_from_args(node.scalar_target, left_op, right_op, **kwargs)

@register_pass
class Lower(Pass):

    def __init__(self, supported_ops):
        self.supported_ops = supported_ops
        self.top = None
        self.object_ids = {}
        super(Lower, self).__init__({})

    def apply_pass(self, node, ctx):
        iter_copy = node.nodes.copy()
        if node.graph is None and not self.top:
            self.top = node
        if node.op_name in self.supported_ops or (
                isinstance(node, (pm.func_op, pm.placeholder, pm.NonLinear, pm.write)) and len(node.nodes) == 0):

            assert node.name != self.top.name

            self.update_args(node)
            for k, n in iter_copy.items():
                node.nodes.pop(k)
            scope_name = f"{node.graph.name}/{node.name}" if id(node.graph) != id(self.top) else node.name

            if node.name in node.graph.nodes:
                node.graph.nodes.pop(node.name)

            if len(node.graph.nodes) == 0:
                node.graph.graph.nodes.pop(node.graph.name)
            node.graph = self.top
            if scope_name in self.top.nodes:
                self.object_ids[self.top.nodes[scope_name]] = [self.top.nodes[scope_name], node]
            else:
                assert scope_name not in self.top.nodes
                node.name = scope_name
                self.top.nodes[scope_name] = node
        return node

    def finalize_pass(self, node, ctx):
        if len(node.nodes) > 0 and node.name != self.top.name:
            node.graph.nodes.pop(node.name)

    def update_args(self, node):
        new_args = []
        for a in node.args:
            if isinstance(a, (pm.GroupNode)):
                if len(a.domain) == 0:
                    new_args.append(a.output_nodes[-1])
                else:
                    new_args.append(a)
            else:
                new_args.append(a)
            assert new_args
        node.args = tuple(new_args)