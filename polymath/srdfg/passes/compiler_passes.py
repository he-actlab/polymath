from polymath.srdfg.passes import register_pass, Pass
from polymath.srdfg.util import _flatten_iterable, is_iterable, extend_indices, squeeze_indices, get_indices
from polymath import func_op, DEFAULT_SHAPES, UNSET_SHAPE, SCALAR_IDX
import polymath as pm
from numbers import Integral
from collections import defaultdict
from itertools import product
import numpy as np
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

@register_pass(analysis=True)
class CountOpTypes(Pass):
    def __init__(self, skip=None):
        self.op_types = defaultdict(int)
        if skip:
            self.skip = skip
        else:
            self.skip = []
        super(CountOpTypes, self).__init__({})

    def apply_pass(self, node, counts):
        if node.op_name not in self.skip:
            self.op_types[node.op_name] += 1
        return node

@register_pass
class NormalizeGraph(Pass):
    def __init__(self, stored_shapes, debug=False):
        self.context = self._check_input_shapes(stored_shapes) if stored_shapes else {}
        if "populate" in stored_shapes:
            self.populate = stored_shapes["populate"]
        else:
            self.populate = True
        if len(stored_shapes) <= 1:
            self.check_values = True
        else:
            self.check_values = False
        self.evaluated = {}
        self.scalar_translations = {}
        self.stored_objects = {}
        self.var_indices = {}
        self.output_shapes = {}
        self.output_shape_deps = defaultdict(list)
        self.delayed_evals = {}
        super(NormalizeGraph, self).__init__({"shapes": {}}, debug=debug)

    def update_args(self, args):
        new_args = []
        for a in args:
            if isinstance(a, (pm.var_index, pm.slice_op)) and a in self.scalar_translations:
                new_args.append(self.scalar_translations[a])
            elif isinstance(a, pm.Node) and not isinstance(a, (pm.input, pm.state, pm.output, pm.temp)):
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

    def update_kwargs(self, kwargs):
        new_kwargs = {}
        for name, arg in kwargs.items():
            if isinstance(arg, (pm.var_index, pm.slice_op)) and arg in self.scalar_translations:
                new_kwargs[name] = self.scalar_translations[arg]
            elif isinstance(arg, pm.Node) and not isinstance(arg, (pm.input, pm.state, pm.output, pm.temp)):
                if arg.name in self.context:
                    new_kwargs[name] = self.context[arg.name]
                elif arg in self.evaluated:
                    new_kwargs[name] = self.evaluated[arg]
                else:
                    new_kwargs[name] = arg
            elif isinstance(arg, tuple):
                new_kwargs[name] = arg
                # raise RuntimeError(f"Dont have a case for handling this yet: {name} - {arg}")
            else:
                new_kwargs[name] = arg
        return new_kwargs


    def initialize_pass(self, graph, _):
        for k in list(self.context.keys()):
            if k in graph.nodes:
                nshape = graph.nodes[k].shape
                for i, s in enumerate(nshape):
                    if isinstance(s, pm.Node):
                        self.context[s.name] = self.context[k][i]
        return graph

    def get_var_index_shape(self, node):
        if all([isinstance(i, int) for i in node.args[1]]):
            assert len(node.args[1]) == len(node.var.shape)
            return (1,)
        else:
            return node.args[1]

    def replace_scalar_var_index(self, node):
        node.graph.nodes.pop(node.name)
        if isinstance(node.var, pm.slice_op):
            if node.name in node.var.nodes:
                scalar_node = node.var.nodes[node.name]
            else:
                kwargs = {}
                kwargs["op_name"] = node.var.target.__name__
                kwargs["graph"] = node.var
                kwargs["name"] = f"{node.name}"
                # if isinstance(node.var.args[0], pm.var_index)
                if isinstance(node.var.args[0], pm.Node):
                    op1 = node.var.args[0][node.args[1]]
                    print(f"{node.var.args[0].op_name}, {node.var.args[0].name}")

                else:
                    op1 = node.var.args[0]

                if isinstance(node.var.args[1], pm.Node):
                    op2 = node.var.args[1][node.args[1]]
                else:
                    op2 = node.var.args[1]
                scalar_node = func_op.init_from_args(node.var.target, op1, op2, **kwargs)
                print(f"{op1.name} - {op1.op_name}")
            self.stored_objects[id(scalar_node)] = scalar_node
        elif isinstance(node.var, pm.NonLinear):
            if node.name in node.var.nodes:
                scalar_node = node.var.nodes[node.name]
            else:
                if 'init_extras' in node.kwargs:
                    in_args = node.kwargs.pop('init_extras')
                else:
                    in_args = tuple([])
                scalar_node = node.var.__class__.init_from_args(*(in_args + (node.var.args[0][node.args[1]],)), graph=node.var, name=f"{node.name}", shape=(1,))
            self.stored_objects[id(scalar_node)] = scalar_node
        else:
            raise RuntimeError
        self.scalar_translations[node] = scalar_node
        return node

    def apply_pass(self, node, info):
        new_shape = []
        if node.is_shape_finalized():
            shape = node.shape
        elif isinstance(node, pm.var_index):
            shape = self.get_var_index_shape(node)
        elif isinstance(node, pm.NonLinear):
            shape = node.args[0].shape
        elif isinstance(node, pm.Transformation):
            shape = node.compute_shape()
        elif isinstance(node, pm.func_op):
            non_scalar = list(filter(lambda x: isinstance(x, pm.Node) and x.shape not in [UNSET_SHAPE, DEFAULT_SHAPES[0]], node.args))
            print(f"{node.name}, {node.op_name}")
            assert len(non_scalar) <= 1
            shape = DEFAULT_SHAPES[0] if len(non_scalar) == 0 else non_scalar[0].shape
        elif isinstance(node, pm.GroupNode):
            shape = node.domain.computed_shape
        else:
            shape = node.shape


        if not isinstance(node, pm.output):
            for s in shape:
                if isinstance(s, pm.Node):
                    if s.name not in self.context and s not in self.evaluated:
                        raise RuntimeError(f"Unable to evaluate shape variable {s} for node {node}.\n"
                                           f"\tContext keys: {list(self.context.keys())}.\n"
                                           f"")
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
            node.kwargs = self.update_kwargs(node.kwargs)

            if isinstance(node, pm.slice_op) and node.shape == pm.DEFAULT_SHAPES[0]:

                idx = node.graph.nodes.item_index(node.name)
                node.graph.nodes.pop(node.name)
                node.kwargs.pop('target')
                target = node.target
                scalar_node = pm.func_op.init_from_args(target, *node.args, graph=None,
                                                        shape=pm.DEFAULT_SHAPES[0],
                                                        name=node.name, **node.kwargs)
                self.scalar_translations[node] = scalar_node
                node.graph.insert_node(scalar_node, idx)
                scalar_node.graph = node.graph

            elif isinstance(node, pm.var_index) and isinstance(node.var, (pm.GroupNode, pm.placeholder, pm.NonLinear, pm.func_op, pm.Transformation)) and node.var.shape == pm.DEFAULT_SHAPES[0]:
                self.scalar_translations[node] = node.var
                node.graph.nodes.pop(node.name)
            elif isinstance(node, pm.var_index) and isinstance(node.var, (pm.slice_op, pm.NonLinear, pm.func_op, pm.Transformation)) and node.shape == pm.DEFAULT_SHAPES[0]:

                assert node.name not in node.nodes
                node = self.replace_scalar_var_index(node)
            else:
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

        if not node.is_shape_finalized() and node.shape != pm.UNSET_SHAPE:
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
        elif isinstance(node, pm.temp):
            self.populate_temp(node)
        elif isinstance(node, pm.placeholder):
            self.populate_placeholder(node)
        elif isinstance(node, pm.var_index):
            self.populate_var_index(node)
        elif isinstance(node, pm.func_op):
            self.populate_func_op(node)
        self.use_updated_scalars(node)
        return node

    def use_updated_scalars(self, node):
        args = []
        for a in node.args:

            if isinstance(a, pm.Node) and a in self.scalar_translations:
                args.append(self.scalar_translations[a])
            else:
                args.append(a)
        node._args = tuple(args)

        kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, pm.Node) and v in self.scalar_translations:
                kwargs[k] = self.scalar_translations[v]
            else:
                kwargs[k] = v
        node.kwargs = kwargs

    def check_scalar_map(self, old_node, new_node):

        self.scalar_translations[old_node] = new_node

    def populate_func_op(self, node):
        pass

    def populate_nonlinear(self, node):
        indices = list(product(*tuple([np.arange(i) for i in node.args[0].shape])))

        if len(indices) > 1:
            if 'init_extras' in node.kwargs:
                in_args = node.kwargs.pop('init_extras')
            else:
                in_args = tuple([])
            for i in indices:
                name = f"{node.name}{i}"
                if name in node.nodes:
                    old = node.nodes.pop(name)
                    x = node.__class__.init_from_args(*(in_args + (node.args[0][i],)), graph=node,
                                                      name=f"{node.name}{i}", shape=(1,))
                    self.check_scalar_map(old, x)
                else:
                    x = node.__class__.init_from_args(*(in_args + (node.args[0][i],)), graph=node, name=f"{node.name}{i}", shape=(1,))

                if name in node.graph.nodes:
                    node.graph.nodes.pop(name)
                self.stored_objects[id(x)] = x

        elif isinstance(node.args[0], pm.GroupNode):
            # TODO: Remove this conditional somehow, group nodes should not require special handling
            new_args = list(node.args)
            new_args[0] = node.args[0].output_nodes[-1]
            node.args = tuple(new_args)


    def populate_slice_op(self, node):

        arg0 = node.args[0]
        arg1 = node.args[1]
        node_dom = node.domain
        op1_idx = node_dom.map_sub_domain(arg0.domain) if isinstance(arg0, pm.Node) and arg0.shape != DEFAULT_SHAPES[0] else SCALAR_IDX
        op2_idx = node_dom.map_sub_domain(arg1.domain) if isinstance(arg1, pm.Node) and arg1.shape != DEFAULT_SHAPES[0] else SCALAR_IDX
        if len(node_dom.set_names) > len(node.shape):
            dom_pairs = node_dom.compute_pairs(squeeze=True)
        else:
            dom_pairs = node_dom.compute_pairs()

        dom_pair_len = len(dom_pairs)
        kwargs = {}

        kwargs["op_name"] = node.target.__name__
        kwargs["graph"] = node
        assert dom_pair_len == len(op1_idx) or op1_idx == SCALAR_IDX
        assert dom_pair_len == len(op2_idx) or op2_idx == SCALAR_IDX

        if len(op1_idx) > 1:
            ops1 = list(map(lambda x: arg0[op1_idx[x]], range(dom_pair_len)))
        elif isinstance(arg0, pm.var_index):
            ops1 = list(map(lambda x: arg0[op1_idx[0]], range(dom_pair_len)))
        else:
            ops1 = list(map(lambda x: arg0, range(len(dom_pairs))))

        if len(op2_idx) > 1:
            ops2 = list(map(lambda x: arg1[op2_idx[x]], range(dom_pair_len)))
        elif isinstance(arg1, pm.var_index):
            ops2 = list(map(lambda x: arg1[op2_idx[0]], range(dom_pair_len)))
        else:
            ops2 = list(map(lambda x: arg1, range(dom_pair_len)))
        assert len(ops1) == len(ops2)

        for p, v in enumerate(dom_pairs):
            kwargs["name"] = f"{node.name}{v}"
            if kwargs["name"] in node.nodes:
                old = node.nodes.pop(kwargs["name"])
                x = func_op.init_from_args(node.target, ops1[p], ops2[p], **kwargs)
                self.check_scalar_map(old, x)
            else:
                x = func_op.init_from_args(node.target, ops1[p], ops2[p], **kwargs)

            if kwargs["name"] in node.graph.nodes:
                node.graph.nodes.pop(kwargs["name"])
            self.stored_objects[id(x)] = x


    def populate_var_index(self, node):
        if node.var.shape != pm.DEFAULT_SHAPES[0]:

            indices = get_indices(node.args[1])

            indices = np.array(list(product(*indices)))
            indices = list(map(lambda x: tuple(x), indices))
            out_shape = node.domain.shape_from_indices(node.args[1])
            out_indices = tuple([np.arange(out_shape[i]) for i in range(len(out_shape))])
            out_indices = np.array(list(product(*out_indices)))
            dom_pairs = node.domain.compute_pairs()
            dom_pairs = list(map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, dom_pairs))
        elif isinstance(node.var, (pm.placeholder)):
            indices = [(0,)]
            out_shape = (1,)
            out_indices = ['']
            dom_pairs = [(0,)]
        else:
            indices = [pm.SCALAR_IDX]
            out_shape = (1,)
            out_indices = [pm.SCALAR_IDX]
            dom_pairs = [pm.SCALAR_IDX]
            idx_name = f"{node.var.name}{pm.SCALAR_IDX}"

            if idx_name not in node.var.nodes:
                if isinstance(node.var, pm.NonLinear):
                    # TODO: Add check or fix so that the input variable is checked to have nodes or not
                    x = node.var.init_from_args(node.var.args[0][pm.SCALAR_IDX], graph=node.var, name=idx_name, shape=(1,))
                    self.stored_objects[id(x)] = x

                elif isinstance(node.var, pm.GroupNode):
                    x = node.var[pm.SCALAR_IDX]
                    self.stored_objects[id(x)] = x
                else:
                    raise RuntimeError(f"Invalid variable for indexing into: {node.var}")
                    # TODO: Add initializers for other types of node
        node._shape = out_shape
        node.domain.set_computed(out_shape, dom_pairs)

        for i, d in enumerate(dom_pairs):
            ph_node = node.var[indices[i]]
            name = f"{node.var.name}{out_indices[i]}"
            node.nodes[name] = ph_node

    def populate_temp(self, node):
        if node.shape != pm.DEFAULT_SHAPES[0]:

            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                x = pm.temp(graph=node, name=f"{node.name}{i}", root_name=node.name, shape=(1,))
                self.stored_objects[id(x)] = x

    def populate_placeholder(self, node):
        if node.shape != pm.DEFAULT_SHAPES[0]:
            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                x = pm.placeholder(graph=node, name=f"{node.name}{i}", root_name=node.name, shape=(1,), type_modifier=node.type_modifier)
                self.stored_objects[id(x)] = x

    def populate_output(self, node):
        if node.shape != pm.DEFAULT_SHAPES[0]:
            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                x = pm.output(graph=node, name=f"{node.name}{i}", root_name=node.name, shape=(1,))
                self.stored_objects[id(x)] = x

    def populate_input(self, node):

        if node.shape != pm.DEFAULT_SHAPES[0]:
            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                x = pm.input(graph=node, name=f"{node.name}{i}", root_name=node.name, shape=(1,))
                self.stored_objects[id(x)] = x


    def populate_state(self, node):
        if node.shape != pm.DEFAULT_SHAPES[0]:
            indices = list(product(*tuple([np.arange(i) for i in node.shape])))
            for i in indices:
                if node.init_value is not None:
                    init_val = node.init_value[i]
                else:
                    init_val = None
                x = pm.state(graph=node, init_value=init_val, name=f"{node.name}{i}", root_name=node.name, shape=(1,))
                self.stored_objects[id(x)] = x

    def populate_write(self, node):
        expanded_dest = False

        if node.shape != pm.DEFAULT_SHAPES[0]:

            src, dst_key, dst = node.args
            key_indices = node.domain.compute_pairs()
            if isinstance(src, pm.Node) and not node.constant_value_write():

                if isinstance(node.args[0], pm.index):
                    src_dom = node.args[0].domain
                else:
                    src_dom = node.args[0].domain
                print(f"{src.name}, {src_dom}")
                src_indices = node.domain.map_sub_domain(src_dom, tuples=False, do_print=True)
                if len(src_indices) > 0 and src_indices.shape[1] < len(src.shape):
                    N = src_indices.shape[0]
                    for i in range(len(src.shape)):
                        if src.shape[i] <= 1:
                            src_indices = np.c_[src_indices[:, :i], np.zeros(N), src_indices[:, i:]]

                dst_indices = list(product(*tuple([np.arange(i) for i in node.shape])))
                src_indices = list(map(lambda x: tuple(x), src_indices.astype(np.int)))

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
                dst_indices = list(product(*tuple([np.arange(i) for i in node.shape])))

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
        if len(node.domain) == 0 or node.shape == pm.DEFAULT_SHAPES[0]:

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
                x = self._iter_div_conquer_reduce(inputs, node)
                node.output_nodes.append(x)
                self.stored_objects[id(x)] = x
            node._shape = (1,)
        else:
            # TODO: Need to make this faster and make sure object ids are added
            axes_idx = np.array([node.input_node.domain.index(s) for s in node.domain])
            sd_map = node.domain.map_reduction_dom(node.input_node.domain, axes_idx)
            assert node.graph
            append_outputs = node.output_nodes.append
            for k,v in sd_map.items():

                inputs = []
                for d in v:
                    i = node.input_node[d]
                    node.nodes[i.name] = i
                    inputs.append(i)
                x = self._iter_div_conquer_reduce(inputs, node)

                if len(inputs) > 1:
                    x.set_name(f"{node.name}{k}")
                append_outputs(x)
                self.stored_objects[id(x)] = x
            node._shape = [node.input_node.domain.computed_set_shape[i] for i in sorted(axes_idx)]

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


    def _iter_div_conquer_reduce(self, input_nodes, node):
        kwargs = {}
        kwargs["graph"] = node
        kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs else node.scalar_target.__name__
        working_set = input_nodes
        while len(working_set) > 1:
            kwargs["name"] = f"{node.name}{(len(node.nodes),)}"
            lop = working_set.pop(0)
            rop = working_set.pop(0)
            x = func_op.init_from_args(node.scalar_target, lop, rop, **kwargs)

            self.stored_objects[id(x)] = x
            working_set.append(x)

        return working_set[0]

    def _div_conquer_reduce(self, left, right, node):
        kwargs = {}
        kwargs["graph"] = node
        kwargs["op_name"] = kwargs["op_name"] if "op_name" in kwargs else node.scalar_target.__name__
        llen = len(left)
        rlen = len(right)
        if llen == 1 and rlen == 1:
            kwargs["name"] = f"{node.name}{(len(node.nodes),)}"
            return func_op.init_from_args(node.scalar_target, left[0], right[0], **kwargs)
        elif llen == 1:
            div = ceil(rlen/2)
            right_op = self._div_conquer_reduce(right[0:div], right[div:], node)
            self.stored_objects[id(right_op)] = right_op

            kwargs["name"] = f"{node.name}{(len(node.nodes),)}"
            return func_op.init_from_args(node.scalar_target, left[0], right_op, **kwargs)
        elif rlen == 1:
            div = ceil(llen/2)
            left_op = self._div_conquer_reduce(left[0:div], left[div:], node)
            self.stored_objects[id(left_op)] = left_op

            kwargs["name"] = f"{node.name}{(len(node.nodes),)}"
            return func_op.init_from_args(node.scalar_target, left_op, right[0], **kwargs)

        else:
            ldiv = ceil(llen/2)
            left_op = self._div_conquer_reduce(left[0:ldiv], left[ldiv:], node)
            self.stored_objects[id(left_op)] = left_op
            rdiv = ceil(rlen/2)
            right_op = self._div_conquer_reduce(right[0:rdiv], right[rdiv:], node)
            self.stored_objects[id(right_op)] = right_op

            kwargs["name"] = f"{node.name}{(len(node.nodes),)}"
            return func_op.init_from_args(node.scalar_target, left_op, right_op, **kwargs)

@register_pass
class Lower(Pass):

    def __init__(self, supported_ops, debug=False):
        self.supported_ops = supported_ops
        self.top = None
        self.object_ids = {}
        self.tobject_ids = {}
        super(Lower, self).__init__({}, debug=debug)

    def print_nodes(self, node):
        if self.top is None:
            return
        node_names = ["lats", "longs", "max_dist", "sqrtz", "ndists", "tempz"]
        print(f"Current node: {node.name}-{node.op_name}")
        all_nodes = []
        for k, v in self.top.nodes.items():
            if any([n in k for n in node_names]):
                all_nodes.append(k)
        print(", ".join(all_nodes))

    def apply_pass(self, node, ctx):

        if id(node) not in self.tobject_ids:
            self.tobject_ids[id(node)] = node
        else:
            raise RuntimeError

        if node.graph is None and not self.top:
            self.top = node


        if node.op_name in self.supported_ops or (
                isinstance(node, (func_op, pm.placeholder, pm.NonLinear, pm.write)) and len(node.nodes) == 0):
            # self.print_nodes(node)

            assert node.name != self.top.name
            self.update_args(node)
            node.nodes = pm.Graph()
            scope_name = self.get_scope_name(node)
            if node.name in node.graph.nodes:
                node.graph.nodes.pop(node.name)

            if node.name != scope_name:
                node.graph.nodes[node.name] = node

            if len(node.graph.nodes) == 0:
                node.graph.graph.nodes.pop(node.graph.name)
            node.graph = self.top

            if scope_name in self.top.nodes:
                self.object_ids[self.top.nodes[scope_name]] = [self.top.nodes[scope_name], node]
            else:
                node.name = scope_name
                self.top.nodes[scope_name] = node

        return node

    def finalize_pass(self, node, ctx):
        if len(node.nodes) > 0 and node.name != self.top.name:
            node.graph.nodes.pop(node.name)

    def package_pass(self, node, ctx):
        # print(list(self.top.nodes.keys()))
        # print(list(node.nodes.keys()))

        # print(self.top.name)
        return node


    def get_scope_name(self, node):
        scope_names = [node.name]
        cgraph = node.graph
        while cgraph and id(cgraph) != id(self.top):
            scope_names.append(cgraph.name)
            cgraph = cgraph.graph
        return "/".join(list(reversed(scope_names)))

    def update_args(self, node):
        new_args = []

        for a in node.args:
            if isinstance(a, (pm.GroupNode)):
                if len(a.domain) == 0 and len(a.output_nodes) > 0:
                    new_args.append(a.output_nodes[-1])
                else:
                    new_args.append(a)
            else:
                new_args.append(a)
            assert new_args
        node.args = tuple(new_args)