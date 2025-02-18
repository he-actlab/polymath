from polymath.srdfg.passes import register_pass, Pass
from polymath.srdfg.templates import fused_dnn
from polymath.srdfg.templates.template_utils import get_pad_tuple
import polymath as pm
import inspect
import numpy as np
from collections import defaultdict, namedtuple
ALL_TEMPLATES = dir(pm.srdfg.templates.math) + dir(pm.srdfg.templates.dnn) + dir(pm.srdfg.templates.tensor_transformations)
NON_DNN_NODE_OPS = (pm.write, pm.placeholder, pm.index, pm.var_index,
                    pm.slice_op, pm.func_op, pm.GroupNode, pm.NonLinear)
BATCH_FUNCS = {}

FusionDescription = namedtuple('FusionDescription', ['output', 'layer'])

FUSION_NAME_MAPPING = {
    'conv': 'conv_bias',
    'relu': 'relu',
    'leakyrelu': 'leaky_relu',
    'add': 'elem_add',
    'sub': 'elem_sub',
    'reducemean': 'reduce_mean',
    'mul': 'elem_mul',
    'div': 'elem_div',
    'sqrt': 'elem_sqrt',
    'depthwiseconv': 'depthwise_conv',
    'depthwiseconvbias': 'depthwise_conv_bias',
    'biasadd': 'bias_add',
    'maxpool': 'max_pool',
    'globalaveragepool': 'global_avg_pool',
    'clip': 'elem_clip',
    'averagepool': 'avg_pool',
    'reciprocal': 'reciprocal',
    'matmul': 'matmul',
    'gemm': 'gemm',
    'softmax': 'softmax',
    'transpose': 'tensor_transpose',
    'pow' : 'elem_pow',
    'reshape': 'tensor_reshape',
    'tanh': 'elem_tanh',
    "gelu": "gelu"
}

@register_pass
class SplitOps(Pass):
    def __init__(self, op_splits):
        assert isinstance(op_splits, dict)
        self.split_layer_counter = 0
        self.validate_splits(op_splits)
        self.op_splits = op_splits
        self.all_split_nodes = {'layers': [],
                           'split_inputs': [],
                           'split_outputs': [],
                           'intermediate': []
                           }
        self.split_instances = defaultdict(int)

        super(SplitOps, self).__init__()


    def split_layer(self, node, split_def, out_node=None):
        op_name, out_idx, all_args = split_def
        instance_name = f"{op_name}{self.split_instances[op_name]}"
        self.split_instances[op_name] += 1


        if isinstance(all_args, tuple):
            assert len(all_args) == 2
            args, kwargs = all_args
        else:
            assert isinstance(all_args, list), f"Not a valid type for non-kwargs: {all_args}"
            args = all_args
            kwargs = {}
        assert isinstance(kwargs, dict), f"Keyword arguments must be mapped from target layer to split layer"
        op_kwargs = {}
        op_args = []
        for init_arg, map_name in kwargs.items():
            op_kwargs[map_name] = node.kwargs[init_arg]
        for a in args:
            if isinstance(a, int):
                op_args.append(node.args[a])
            elif isinstance(a, tuple):
                op_args.append(self.split_layer(node, a))

        with node.graph:
            if out_node is None:
                out_node = pm.output(name=f"{node.outputs[0].name}_split{self.split_layer_counter}",
                                     shape=node.args[out_idx].shape)
                self.split_layer_counter += 1
            op_args.append(out_node)
            new_node = getattr(pm, op_name)(*op_args, name=instance_name, **op_kwargs)
        self.all_split_nodes['layers'].append(new_node)
        self.all_split_nodes['split_inputs'].append(out_node)
        self.topological_insert(node.graph, new_node)
        return out_node

    def initialize_pass(self, graph, ctx):
        nidx = 0
        node_list = list(graph.nodes.values())
        while nidx < len(node_list):
            n = node_list[nidx]

            if not isinstance(n, pm.Template):
                nidx += 1
                continue
            elif n in self.all_split_nodes['layers']:
                nidx += 1
                continue
            elif any([o in self.all_split_nodes['split_inputs'] for o in n.outputs]):
                    nidx += 1
                    continue

            if n.op_name in self.op_splits:
                self.split_layer(n, self.op_splits[n.op_name], out_node=n.outputs[0])
                self.remove_fused_node(graph, n)

            nidx += 1
        return graph

    def remove_fused_node(self, graph, node):
        graph.nodes.pop(node.name)

    def validate_splits(self, splits):
        for fused_op, split in splits.items():
            missing_ops = self.validate_split(split, [])
            if len(missing_ops) > 0:
                raise RuntimeError(f"Missing operations found for split definitions:\n"
                                   f"{missing_ops}")

    def validate_split(self, op_info, missing_ops):
        assert isinstance(op_info, tuple) and len(op_info) == 3

        op_name, out_idx, all_args = op_info
        assert isinstance(out_idx, int)
        if op_name not in ALL_TEMPLATES:
            missing_ops.append(op_name)
            return missing_ops
        if isinstance(all_args, tuple):
            assert len(all_args) == 2
            args, kwargs = all_args
        else:
            assert isinstance(all_args, list), f"Not a valid type for non-kwargs: {all_args}"
            args = all_args
            kwargs = {}
        assert isinstance(kwargs, dict), f"Keyword arguments must be mapped from target layer to split layer"
        self.check_signature(op_name, args, kwargs)

        for a in args:
            if isinstance(a, tuple):
                missing_ops = self.validate_split(a, missing_ops)

        return missing_ops

    def check_signature(self, split_name, layer_inputs, layer_kwargs):
        signature = inspect.signature(getattr(pm, split_name).define_graph)

        args = []
        kwargs = []
        for k, v in signature.parameters.items():
            if v.default is v.empty:
                args.append(k)
            else:
                kwargs.append(k)
        for k in layer_kwargs.keys():
            if k not in kwargs:
                raise RuntimeError(f"Non-existent keyword argument to split: {k} in operation {split_name}")
        # TODO: Need to remove the hardcoded '2', as there could be more than 1 output argument
        # This assumes 1 argument is 'self', another argument is 'output'
        if len(args) - 2 != len(layer_inputs) or len(kwargs) != len(layer_kwargs):

            raise RuntimeError(f"Invalid arguments split in {split_name}:\n"
                               f"Fusion signature args: {args}\n"
                               f"Fusion signature kwargs: {kwargs}\n"
                               f"Layer args: {layer_inputs}\n"
                               f"Kwargs: {layer_kwargs}")

    def topological_insert(self, graph, node):
        assert isinstance(node, pm.Node) and hasattr(node, 'inputs')
        assert all([i.name in graph.nodes for i in node.inputs])
        graph.nodes.pop(node.name)
        min_idx = 0

        for k, n in graph.nodes.items():
            i = list(graph.nodes.keys()).index(k)
            if isinstance(n, pm.Template):
                for o in n.outputs:
                    if o in node.inputs and i > min_idx:
                        min_idx = i
            elif n in node.inputs and i > min_idx:
                min_idx = i

        out = graph.nodes.pop(node.outputs[0].name)
        graph.insert_node(out, min_idx + 1)
        graph.insert_node(node, min_idx + 1)

@register_pass
class FuseOps(Pass):
    def __init__(self, fusion_seqs, pad_conv_constraint=False, test_run=False):
        self.test_run = test_run
        fusion_ops = []
        for o in fusion_seqs:
            seq = []
            for s in o:
                if isinstance(s, list):
                    subseq = []
                    for sub in s:
                        sl = sub.lower()
                        if sl in FUSION_NAME_MAPPING:
                            subseq.append(FUSION_NAME_MAPPING[sl])
                        else:
                            subseq.append(sl)
                    seq.append(subseq)
                else:
                    sl = s.lower()
                    if sl in FUSION_NAME_MAPPING:
                        seq.append(FUSION_NAME_MAPPING[sl])
                    else:
                        seq.append(sl)
            fusion_ops.append(seq)
        assert isinstance(fusion_ops, list) and len(fusion_ops) > 0
        self.pad_conv_constraint = pad_conv_constraint
        self.check_valid_fusions(fusion_ops)
        self.fusion_sequences = fusion_ops
        self.fusion_sequences = sorted(self.fusion_sequences, key=lambda x: len(x), reverse=True)
        self.fusion_starts = [f[0] for f in fusion_ops]
        self.all_fused_nodes = {'layers': [],
                           'fusion_inputs': [],
                           'fusion_outputs': [],
                           'intermediate': []
                           }
        self.fusion_instances = defaultdict(int)
        super(FuseOps, self).__init__()

    def check_valid_fusions(self, fusion_ops):
        missing_ops = []
        for f in fusion_ops:
            name = self.get_fusion_name(f)
            if name not in dir(fused_dnn):
                missing_ops.append((name, f))
        if len(missing_ops) > 0:
            raise RuntimeError(f"Fusion templates do not exist for sequences:"
                               f"\n{missing_ops}")


    def is_conv_dw_conv(self, seq) -> bool:
        return "conv_bias" == seq[0] and ("depthwise_conv_bias" in seq or "depthwise_conv" in seq)

    def is_valid_conv_dw_conv(self, conv_node) -> bool:
        assert conv_node.op_name == "conv_bias"
        return conv_node.inputs[1].shape[2:] == (1, 1)

    def get_possible_fusions(self, n):
        # TODO: Might need to validate the first operation is not arg list
        possible_fusions = []
        if self.pad_conv_constraint:
            for s in self.fusion_sequences:
                if s[0] == n.op_name:
                    if not self.is_conv_dw_conv(s):
                        possible_fusions.append(s)
                    elif self.is_valid_conv_dw_conv(n):
                        possible_fusions.append(s)
        else:
            for s in self.fusion_sequences:
                if s[0] == n.op_name:
                    possible_fusions.append(s)
        return possible_fusions

    def initialize_pass(self, graph, ctx):
        nidx = 0
        node_list = list(graph.nodes.values())
        while nidx < len(node_list):
            n = node_list[nidx]

            if not isinstance(n, pm.Template):
                nidx += 1
                continue
            elif n in self.all_fused_nodes['layers']:
                nidx += 1
                continue
            elif any([o in self.all_fused_nodes['fusion_inputs'] for o in n.outputs]):
                    nidx += 1
                    continue

            if n.op_name in self.fusion_starts:
                possible_fusions = self.get_possible_fusions(n)
                for pf in possible_fusions:
                    fused_nodes = self.get_fused_nodes(graph, pf, n)
                    if fused_nodes is not None:
                        self.fuse_layers(graph, fused_nodes, pf)
                        break

            nidx += 1

        return graph

    def cleanup_writes(self, graph, layers, intermediate_nodes, result):
        layer_nodes = []
        for layer_list in layers:
            for l in layer_list:
                assert l.layer.name in graph.nodes
                layer_nodes.append(l.layer)
                graph.nodes.pop(l.layer.name)

        for i in intermediate_nodes:
            if i.op_name == "output":
                i.reset_writes()
            is_read = False
            for n in graph.nodes.values():
                if isinstance(n, pm.Template) and n not in layer_nodes and i in n.inputs:
                    is_read = True
                    break
            if not is_read:
                graph.nodes.pop(i.name)

        assert result.op_name == "output"
        result.reset_writes()

    def flatten_seq(self, list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return self.flatten_seq(list_of_lists[0]) + self.flatten_seq(list_of_lists[1:])
        return list_of_lists[:1] + self.flatten_seq(list_of_lists[1:])

    def get_fusion_name(self, fusion_ops):
        fusion_ops = self.flatten_seq(fusion_ops)
        fusion_name = "_".join(fusion_ops)
        fusion_name = fusion_name.replace("elem_", "").replace("reduce_", "").replace('tensor_', '')
        return fusion_name


    def fuse_layers(self, graph, layers, fusion_ops):
        fusion_name = self.get_fusion_name(fusion_ops)
        instance_name = f"{fusion_name}{self.fusion_instances[fusion_name]}"
        self.fusion_instances[fusion_name] += 1
        if self.test_run:
            return
        intermediate_nodes = []
        for layer_list in layers[:-1]:
            for l in layer_list:
                intermediate_nodes.append(l.output)

        fused_templates = []
        for layer_list in layers:
            for l in layer_list:
                fused_templates.append(l.layer)

        layer_inputs = []
        layer_kwargs = {}
        assert layers[0][0].layer.name in graph.nodes
        argname_counts = defaultdict(int)
        for layer_list in layers:
            for l in layer_list:
                for i in l.layer.inputs:
                    if i not in intermediate_nodes and i not in layer_inputs:
                        layer_inputs.append(i)

                for k, v in l.layer.kwargs.items():
                    if k in layer_kwargs:
                        layer_kwargs[f"{k}{argname_counts[k]}"] = v
                        argname_counts[k] += 1
                    else:
                        layer_kwargs[k] = v


        result = layers[-1][-1].output
        self.cleanup_writes(graph, layers, intermediate_nodes, result)
        layer_inputs.append(result)

        self.all_fused_nodes['intermediate'] += intermediate_nodes
        self.all_fused_nodes['layers'] += fused_templates
        self.all_fused_nodes['fusion_inputs'] += layer_inputs
        self.all_fused_nodes['fusion_outputs'].append(result)

        signature = inspect.signature(getattr(fused_dnn, fusion_name).define_graph)

        all_arg_len = len(signature.parameters.keys()) - 1
        args = []
        kwargs = []
        for k, v in signature.parameters.items():
            if v.default is v.empty:
                args.append(k)
            else:
                kwargs.append(k)
        if len(args) - 1 != len(layer_inputs) or len(kwargs) != len(layer_kwargs):
        # if all_arg_len != len(layer_inputs) + len(layer_kwargs):

            raise RuntimeError(f"Invalid arguments for layer fusion in {fusion_name}:\n"
                               f"Fusion signature args: {args}\n"
                               f"Fusion signature kwargs: {kwargs}\n"
                               f"Layer args: {[n.name for n in layer_inputs]}\n"
                               f"Kwargs: {layer_kwargs.keys()}")
        with graph:
            node = getattr(fused_dnn, fusion_name)(*layer_inputs, name=instance_name, **layer_kwargs)
        self.topological_insert(graph, node)


    def print_graph(self, graph):
        for name, node in graph.nodes.items():
            print(f"{node.op_name}")

    def topological_insert(self, graph, node):

        assert isinstance(node, pm.Node) and hasattr(node, 'inputs')
        assert all([i.name in graph.nodes for i in node.inputs])

        graph.nodes.pop(node.name)
        min_idx = 0
        for k, n in graph.nodes.items():
            i = list(graph.nodes.keys()).index(k)
            if isinstance(n, pm.Template):
                for o in n.outputs:
                    if o in node.inputs and i > min_idx:
                        min_idx = i
            elif n in node.inputs and i > min_idx:
                min_idx = i


        out = graph.nodes.pop(node.outputs[0].name)
        graph.insert_node(out, min_idx + 1)
        graph.insert_node(node, min_idx + 1)



    def get_fused_nodes(self, graph, sequence, initial_layer):
        # TODO: Make sure the output isnt used in multiple places
        assert hasattr(initial_layer, "outputs") and len(initial_layer.outputs) == 1
        tgt_input = initial_layer.outputs[0]
        fdescriptors = [
            [FusionDescription(output=tgt_input, layer=initial_layer)]]
        for i, l in enumerate(sequence[1:]):
            fl = self.get_fusable_layer(graph, l, tgt_input)
            if fl is None:
                return None
            else:
                assert isinstance(fl, list)
                tgt_input = fl[0].output
                fdescriptors.append(fl)

        return fdescriptors

    def get_fusable_layer(self, graph, layer_name, input_node):
        if isinstance(layer_name, list):
            out_layers = []
            outputs = []
            for l in layer_name:
                for name, n in graph.nodes.items():
                    if isinstance(n, pm.Template) and n.op_name == l and input_node in n.inputs and n.outputs[0] not in outputs:
                        assert hasattr(n, "outputs") and len(n.outputs) == 1
                        out_layers.append(FusionDescription(output=n.outputs[0], layer=n))
                        outputs.append(n.outputs[0])
            if len(out_layers) == len(layer_name):
                return out_layers

        else:
            for name, n in graph.nodes.items():
                if isinstance(n, pm.Template) and n.op_name == layer_name and input_node in n.inputs:
                    assert hasattr(n, "outputs") and len(n.outputs) == 1
                    return [FusionDescription(output=n.outputs[0], layer=n)]
        return None

    def num_fusions(self):
        return self.fusion_instances

@register_pass
class CollectDNNShapes(Pass):
    def __init__(self):
        self.op_counter = defaultdict(int)
        self.shape_tracker = {}
        super(CollectDNNShapes, self).__init__()

    def apply_pass(self, node, ctx):
        if node.op_name in pm.ONNX_OP_NAMES:
            shapes = []
            for i in node.inputs:
                if isinstance(i, pm.Node):
                    shapes.append(i.shape)
            for o in node.outputs:
                if isinstance(o, pm.Node):
                    shapes.append(o.shape)
            self.shape_tracker[f"{node.op_name}{self.op_counter[node.op_name]}"] = shapes
            self.op_counter[node.op_name] += 1

@register_pass
class UpdateBatchSize(Pass):
    def __init__(self, batch_size, graph_name):
        self.graph_name = graph_name
        self.batch_size = batch_size
        self.op_counter = defaultdict(int)
        self.shape_tracker = {}
        super(UpdateBatchSize, self).__init__()

    def apply_pass(self, node, ctx):
        assert node.op_name not in dir(fused_dnn), f"Batch sizes must be updated prior to fusion, but found fused operation: {node.op_name}"
        if not isinstance(node, NON_DNN_NODE_OPS) and node.op_name != self.graph_name and node.name != self.graph_name and isinstance(node, pm.Template):
            assert node.op_name in BATCH_FUNCS, f"{node.op_name}, {self.graph_name}, {node.name}"
            node, shape_list = BATCH_FUNCS[node.op_name](node, self.batch_size)
            self.shape_tracker[f"{node.op_name}{self.op_counter[node.op_name]}"] = shape_list
            self.op_counter[node.op_name] += 1
        return node

@register_pass
class ScaleLayers(Pass):
    ELEMWISE = ["relu", "leaky_relu", "elem_add", "elem_sub", "elem_pow", "elem_sqrt", "elem_div", "elem_mul",
                "elem_where", "elem_tanh", "elem_clip", "gelu", "softmax"]
    UNIQUE=  ["global_avg_pool", "conv_bias", "conv", "max_pool", "gemm", "reduce_mean", "tensor_reshape", "split", "matmul", "tensor_transpose",
              "depthwise_conv_bias", "avg_pool", "tensor_squeeze", "coarse_flatten", "resize", "concat", "depthwise_conv_bias", "bias_add"]
    def __init__(self, scale_factors, graph_name, divisor=32, fallback_split=2, debug=False):
        self.fallback_split = fallback_split
        self.divisor = divisor
        assert divisor % fallback_split == 0
        self.vision_model = False
        self.graph_name = graph_name
        self.oc_split = scale_factors['OC']
        self.oh_split = scale_factors['OH']
        self.ow_split = scale_factors['OW']
        self.scale_factors = scale_factors
        self.op_counter = defaultdict(int)
        self.debug = debug

        self.shape_tracker = {}
        super(ScaleLayers, self).__init__()

    def print_shapes(self, node):
        for i in node.inputs + node.outputs:
            print(f"{i.name}: {i.shape}")

    def update_elemwise_transformer(self, node):
        shapes_list = []
        orig_base = None
        new_base = None
        for i in node.inputs:
            if 'base_shape' in i.kwargs:
                orig_base = i.kwargs['base_shape']
                new_base = i.shape
                break
        assert orig_base is not None, f"Could not find existing shape for elemwise operation {node.op_name} in node {node.name}"
        for i in (node.inputs + node.outputs):
            if 'base_shape' in i.kwargs:
                if orig_base != i.kwargs['base_shape']:
                    for idx in range(len(i.kwargs['base_shape'])):
                        rev_idx_new = len(i.kwargs['base_shape']) - idx - 1
                        rev_idx_exp = len(orig_base) - idx - 1
                        d = i.kwargs['base_shape'][rev_idx_new]
                        if d != orig_base[rev_idx_exp]:
                            assert d == 1, f"Expected non-equal dimensiosn to be 1 in elementwise {node.op_name}, but got {i.shape}, dimension {d} for {i.name}, expected {orig_base[rev_idx_exp]}"
                else:
                    assert orig_base == i.kwargs['base_shape'], f"Expected original base to be equal in operation {node.name}:{node.op_name}\n" \
                                                                f"Expected: {orig_base}\n" \
                                                                f"{i.name}: {i.kwargs['base_shape']} "
                    assert i.shape == new_base, f"Expected new base to be equal in operation {node.name}:{node.op_name}\n" \
                                                                f"Expected: {new_base}\n" \
                                                                f"{i.name}: {i.shape}\n" \
                                                f"Input shapes: {[inp.shape for inp in node.inputs]}"
                shapes_list.append(i.shape)
                continue
            new_shape = []
            for idx in range(len(i.shape)):
                rev_idx_new = len(i.shape) - idx - 1
                rev_idx_exp = len(orig_base) - idx - 1
                if i.shape[rev_idx_new] != orig_base[rev_idx_exp]:
                    assert i.shape[rev_idx_new] == 1, f"Expected non-equal dimensiosn to be 1 in elementwise {node.op_name}, but got {i.shape} ({i.shape[rev_idx_new]}, idx={rev_idx_new}) for {i.name}, expected {orig_base[rev_idx_exp]}"
                    new_shape.insert(0, 1)
                else:
                    new_shape.insert(0, new_base[rev_idx_exp])
            i.add_attribute('base_shape', i.shape)
            i.set_shape(new_shape, override=True)
            shapes_list.append(i.shape)
        return node, shapes_list

    def update_reduce_mean(self, node):
        x = node.args[0]
        o = node.args[1]
        axis = node.kwargs['axes']
        assert 'base_shape' in x.kwargs
        new_shape = []
        for i, d in enumerate(x.shape):
            if i in axis:
                new_shape.append(1)
            else:
                new_shape.append(d)
        o.add_attribute('base_shape', o.shape)
        o.set_shape(new_shape, override=True)
        return node, [x.shape, o.shape]

    def update_bias_add(self, node):
        x = node.args[0]
        b = node.args[1]
        o = node.args[2]
        assert 'base_shape' in x.kwargs
        assert len(b.shape) == 1
        b.add_attribute('base_shape', b.shape)
        b.set_shape([x.shape[1]], override=True)

        o.add_attribute('base_shape', o.shape)
        o.set_shape(x.shape, override=True)
        return node, [x.shape, b.shape, o.shape]


    def update_elemwise(self, node):
        shapes_list = []
        orig_base = None
        new_base = None
        for i in node.inputs:
            if 'base_shape' in i.kwargs:
                orig_base = i.kwargs['base_shape']
                new_base = i.shape
                break
        assert orig_base is not None, f"Could not find existing shape for elemwise operation in node {node.name}"
        for i in (node.inputs + node.outputs):
            if 'base_shape' in i.kwargs:
                assert orig_base == i.kwargs['base_shape'], f"Expected original base to be equal in operation {node.name}:{node.op_name}\n" \
                                                            f"Expected: {orig_base}\n" \
                                                            f"{i.name}: {i.kwargs['base_shape']} "
                assert i.shape == new_base, f"Expected new base to be equal in operation {node.name}:{node.op_name}\n" \
                                                            f"Expected: {new_base}\n" \
                                                            f"{i.name}: {i.shape} "
                shapes_list.append(i.shape)
                continue
            assert i.shape == orig_base, f"Expected all elementwise shapes to be equal, but got {i.shape} and {orig_base}"
            i.add_attribute('base_shape', i.shape)
            i.set_shape(new_base, override=True)

            shapes_list.append(i.shape)
        return node, shapes_list

    @property
    def batch_split(self):
        return self.oh_split*self.ow_split

    def update_transformer_shapes(self, node):
        if node.op_name in self.ELEMWISE: return self.update_elemwise_transformer(node)
        elif node.op_name == "bias_add": return self.update_bias_add(node)
        elif node.op_name in ["conv", "depthwise_conv"]: return self.update_conv(node, dw_conv=node.op_name=="depthwise_conv")
        elif node.op_name in ["conv_bias", "depthwise_conv_bias"]: return self.update_conv_bias(node, dw_conv=node.op_name=="depthwise_conv_bias")
        elif node.op_name == "matmul": return self.update_matmul(node)
        elif node.op_name == "gemm": return self.update_gemm(node)
        elif node.op_name == "global_avg_pool": return self.update_global_avg_pool(node)
        elif node.op_name == "avg_pool": return self.update_pool(node)
        elif node.op_name == "max_pool": return self.update_pool(node)
        elif node.op_name == "elem_where": return self.update_elem_where(node)
        elif node.op_name == "concat": return self.update_concat(node)
        elif node.op_name == "resize": return self.update_resize(node)
        elif node.op_name == "split": return self.update_split(node)
        elif node.op_name == "coarse_flatten": return self.update_coarse_flatten(node)
        elif node.op_name == "tensor_reshape": return self.update_tensor_reshape(node)
        elif node.op_name == "tensor_squeeze": return self.update_tensor_squeeze(node)
        elif node.op_name == "tensor_transpose": return self.update_tensor_transpose(node)
        elif node.op_name == "reduce_mean": return self.update_reduce_mean(node)
        else:
            raise RuntimeError(f"Unsupported transformer operation {node.op_name}")

    def get_conv_attrs(self, node):
        num_filter, channel, kernel_h, kernel_w = node.args[1].shape
        dilation = node.kwargs['dilation']
        stride = node.kwargs['stride']
        pad = node.kwargs['pad']
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad

        info = {}
        info['stride'] = (stride_h, stride_h)
        info['dilation'] = (dilation_h, dilation_w)
        info['padding'] = (pad_top, pad_left, pad_down, pad_right)
        return info

    def update_conv_bias(self, node, dw_conv=False):
        x = node.args[0]
        w = node.args[1]
        b = node.args[2]
        o = node.args[3]
        _, _, kernel_h, kernel_w = w.shape
        _, oc_orig, oh_orig, ow_orig = o.shape
        info = self.get_conv_attrs(node)

        n = x.shape[0]
        oc_new = np.ceil(oc_orig / self.oc_split).astype('int32')

        assert 'base_shape' not in o.kwargs and 'base_shape' not in w.kwargs and 'base_shape' not in b.kwargs
        assert 'base_shape' in x.kwargs, f"Could not find shape in arg {x.name} for node {node.name}"
        ic_orig = x.kwargs['base_shape'][1]
        if x.kwargs.get('onnx_input', False):
            oh_new = np.ceil(oh_orig/self.oh_split).astype('int32')
            ow_new = np.ceil(ow_orig/self.ow_split).astype('int32')
            ic_new = x.shape[1]
            ih_new = ((oh_new - 1)*info['stride'][0]) + 1 + info['dilation'][0] * (kernel_h - 1) -  info['padding'][0] - info['padding'][2]
            iw_new = ((ow_new - 1)*info['stride'][1]) + 1 + info['dilation'][1] * (kernel_w - 1) -  info['padding'][1] - info['padding'][3]
            assert ih_new > 0 and iw_new > 0
            x.set_shape([n, x.shape[1], ih_new, iw_new], override=True)
        else:
            _, ic_new, ih_new, iw_new = x.shape
            oh_new = (ih_new + info['padding'][0] + info['padding'][2] - info['dilation'][0] * (kernel_h - 1) - 1) / info['stride'][0] + 1
            ow_new = (iw_new + info['padding'][1] + info['padding'][3] - info['dilation'][1] * (kernel_w - 1) - 1) / info['stride'][1] + 1

        if int(oh_new) == 0:
            new_padding = list(info['padding'])
            if info['padding'][0] > info['padding'][2]:
                new_padding[2] += 1
            else:
                new_padding[0] += 1
            info['padding'] = new_padding
            oh_new = (ih_new + info['padding'][0] + info['padding'][2] - info['dilation'][0] * (kernel_h - 1) - 1) / info['stride'][0] + 1

        if int(ow_new) == 0:
            new_padding = list(info['padding'])
            if info['padding'][1] > info['padding'][3]:
                new_padding[3] += 1
            else:
                new_padding[1] += 1
            info['padding'] = new_padding
            ow_new = (iw_new + info['padding'][1] + info['padding'][3] - info['dilation'][1] * (kernel_w - 1) - 1) / info['stride'][1] + 1
        node.kwargs['pad'] = info['padding']
        oh_new = int(oh_new)
        ow_new = int(ow_new)
        assert oh_new > 0, f"Expected non-zero output height for {node.op_name}:\n" \
                           f"Input name: {x.name}\n" \
                           f"OH: {oh_new}\n" \
                           f"Previous OH: {oh_orig}\n" \
                           f"IH: {ih_new}\n" \
                           f"Previous IH: {x.kwargs['base_shape'][2]}\n" \
                           f"Pad: {info['padding'][0]}, {info['padding'][2]}\n" \
                           f"dilation: {info['dilation'][0]}\n" \
                           f"kh: {kernel_h}, stride: {info['stride'][0]}"
        assert ow_new > 0, f"Expected non-zero output width for {node.op_name}:\n" \
                           f"Input name: {x.name}\n" \
                           f"OH: {ow_new}\n" \
                           f"Previous OW: {ow_orig}\n" \
                           f"IW: {iw_new}\n" \
                           f"Previous IW: {x.kwargs['base_shape'][3]}\n" \
                           f"Pad: {info['padding'][1]}, {info['padding'][3]}\n" \
                           f"dilation: {info['dilation'][0]}\n" \
                           f"kw: {kernel_w}, stride: {info['stride'][0]}"
        o.set_shape([n, oc_new, oh_new, ow_new], override=True)
        o.add_attribute('base_shape', (n, oc_orig, oh_orig, ow_orig))
        w.add_attribute('base_shape', w.shape)
        w.set_shape([oc_new, ic_new, kernel_h, kernel_w], override=True)
        b.set_shape([oc_new], override=True)
        b.add_attribute('base_shape', oc_orig)

        if dw_conv:
            w.set_shape([w.shape[0], 1, w.shape[2], w.shape[3]], override=True)
        shapes_list = []
        shapes_list.append(x.shape)
        shapes_list.append(w.shape)
        shapes_list.append(b.shape)
        shapes_list.append(o.shape)
        return node, shapes_list


    def update_conv(self, node, dw_conv=False):
        x = node.args[0]
        w = node.args[1]
        o = node.args[2]
        _, _, kernel_h, kernel_w = w.shape
        _, oc_orig, oh_orig, ow_orig = o.shape
        info = self.get_conv_attrs(node)

        n = x.shape[0]
        oc_new = np.ceil(oc_orig / self.oc_split).astype('int32')

        assert 'base_shape' not in o.kwargs and 'base_shape' not in w.kwargs
        assert 'base_shape' in x.kwargs, f"Could not find shape in arg {x.name} for node {node.name}"
        ic_orig = x.kwargs['base_shape'][1]
        if x.kwargs.get('onnx_input', False):
            oh_new = np.ceil(oh_orig/self.oh_split).astype('int32')
            ow_new = np.ceil(ow_orig/self.ow_split).astype('int32')
            ic_new = x.shape[1]
            ih_new = ((oh_new - 1)*info['stride'][0]) + 1 + info['dilation'][0] * (kernel_h - 1) -  info['padding'][0] - info['padding'][2]
            iw_new = ((ow_new - 1)*info['stride'][1]) + 1 + info['dilation'][1] * (kernel_w - 1) -  info['padding'][1] - info['padding'][3]
            assert ih_new > 0 and iw_new > 0
            x.set_shape([n, x.shape[1], ih_new, iw_new], override=True)
        else:
            _, ic_new, ih_new, iw_new = x.shape
            oh_new = (ih_new + info['padding'][0] + info['padding'][2] - info['dilation'][0] * (kernel_h - 1) - 1) / info['stride'][0] + 1
            ow_new = (iw_new + info['padding'][1] + info['padding'][3] - info['dilation'][1] * (kernel_w - 1) - 1) / info['stride'][1] + 1
        if int(oh_new) == 0:
            new_padding = list(info['padding'])
            if info['padding'][0] > info['padding'][2]:
                new_padding[2] += 1
            else:
                new_padding[0] += 1
            info['padding'] = new_padding
            oh_new = (ih_new + info['padding'][0] + info['padding'][2] - info['dilation'][0] * (kernel_h - 1) - 1) / info['stride'][0] + 1

        if int(ow_new) == 0:
            new_padding = list(info['padding'])
            if info['padding'][1] > info['padding'][3]:
                new_padding[3] += 1
            else:
                new_padding[1] += 1
            info['padding'] = new_padding
            ow_new = (iw_new + info['padding'][1] + info['padding'][3] - info['dilation'][1] * (kernel_w - 1) - 1) / info['stride'][1] + 1
        node.kwargs['pad'] = info['padding']
        oh_new = int(oh_new)
        ow_new = int(ow_new)
        assert oh_new > 0, f"Expected non-zero output height for {node.op_name}:\n" \
                           f"Input name: {x.name}\n" \
                           f"OH: {oh_new}\n" \
                           f"Previous OH: {oh_orig}\n" \
                           f"IH: {ih_new}\n" \
                           f"Previous IH: {x.kwargs['base_shape'][2]}\n" \
                           f"Pad: {info['padding'][0]}, {info['padding'][2]}\n" \
                           f"dilation: {info['dilation'][0]}\n" \
                           f"kh: {kernel_h}, stride: {info['stride'][0]}"
        assert ow_new > 0, f"Expected non-zero output width for {node.op_name}:\n" \
                           f"Input name: {x.name}\n" \
                           f"OH: {ow_new}\n" \
                           f"Previous OW: {ow_orig}\n" \
                           f"IW: {iw_new}\n" \
                           f"Previous IW: {x.kwargs['base_shape'][3]}\n" \
                           f"Pad: {info['padding'][1]}, {info['padding'][3]}\n" \
                           f"dilation: {info['dilation'][0]}\n" \
                           f"kw: {kernel_w}, stride: {info['stride'][0]}"
        o.set_shape([n, oc_new, oh_new, ow_new], override=True)
        o.add_attribute('base_shape', (n, oc_orig, oh_orig, ow_orig))

        w.add_attribute('base_shape', w.shape)
        w.set_shape([oc_new, ic_new, kernel_h, kernel_w], override=True)
        # w.add_attribute('base_shape', (oc_orig, ic_orig, kernel_h, kernel_w))
        if dw_conv:
            w.set_shape([w.shape[0], 1, w.shape[2], w.shape[3]], override=True)
        shapes_list = []
        shapes_list.append(x.shape)
        shapes_list.append(w.shape)
        shapes_list.append(o.shape)
        return node, shapes_list

    def update_matmul(self, node):
        x = node.args[0]
        w = node.args[1]
        o = node.args[2]
        if len(x.shape) == 3:
            assert 'base_shape' in x.kwargs
            orig_p = w.shape[-1]
            new_p = max(np.ceil(orig_p / self.oc_split), self.divisor)
            new_m = x.shape[-2]
            new_n = x.shape[-1]

            w.add_attribute("base_shape", w.shape)
            assert len(w.shape) == 2
            w.set_shape([new_n, new_p], override=True)

            o.add_attribute("base_shape", o.shape)
            o.set_shape([x.shape[0], new_m, new_p], override=True)
        elif len(w.shape) == 4:
            assert len(x.shape) == 4
            assert len(w.shape) == 4
            assert 'base_shape' in x.kwargs
            assert 'base_shape' in w.kwargs
            new_m = x.shape[-2]
            new_p = w.shape[-1]
            o.add_attribute("base_shape", o.shape)
            o.set_shape([x.shape[0], x.shape[1], new_m, new_p], override=True)
        else:
            assert len(x.shape) == 4
            assert len(w.shape) == 2
            assert 'base_shape' in x.kwargs
            assert 'base_shape' not in w.kwargs
            new_m = x.shape[-2]
            new_p = max(np.ceil(w.shape[-1]/self.oc_split), self.divisor)
            w.add_attribute("base_shape", w.shape)
            w.set_shape([x.shape[-1], new_p], override=True)
            o.add_attribute("base_shape", o.shape)
            o.set_shape([x.shape[0], x.shape[1], new_m, new_p], override=True)
        return node, [x.shape, w.shape, o.shape]




        assert len(x.shape) == 2
        assert len(w.shape) == 2
        assert len(o.shape) == 2

        return node, [x.shape, w.shape, o.shape]

    def update_gemm(self, node):

        transA = node.kwargs['transA']
        transB = node.kwargs['transB']
        x = node.args[0]

        w = node.args[1]
        b = node.args[2]
        o = node.args[3]
        assert len(x.shape) == 2
        assert len(w.shape) == 2
        assert len(o.shape) == 2
        if transA:
            new_m = x.shape[1]
            new_n = x.shape[0]
        else:
            new_m = x.shape[0]
            new_n = x.shape[1]

        if transB:
            orig_p = w.shape[0]
            if self.vision_model:
                new_p = np.ceil(orig_p / self.oc_split)
            else:
                new_p = max(np.ceil(orig_p / self.oc_split), self.divisor)
            w.set_shape([new_p, new_n], override=True)
        else:
            orig_p = w.shape[1]
            if self.vision_model:
                new_p = np.ceil(orig_p / self.oc_split)
            else:
                new_p = max(np.ceil(orig_p / self.oc_split), self.divisor)
            w.add_attribute('base_shape', w.shape)
            w.set_shape([new_n, new_p], override=True)
        b.add_attribute('base_shape', b.shape)
        if len(b.shape) == 2:
            b.set_shape([new_m, new_p], override=True)
        else:
            b.set_shape([new_p], override=True)
        o.add_attribute('base_shape', o.shape)
        o.set_shape([new_m, new_p], override=True)
        return node, [x.shape, w.shape, b.shape, o.shape]

    def update_global_avg_pool(self, node):
        shapes_list = []
        x = node.args[0]
        assert 'base_shape' in x.kwargs
        o = node.args[1]
        new_out_shape = [o.shape[0], x.shape[1], o.shape[2], o.shape[3]]
        o.add_attribute('base_shape', o.shape)
        o.set_shape(new_out_shape, override=True)
        shapes_list.append(x.shape)
        shapes_list.append(o.shape)
        return node, shapes_list


    def update_pool(self, node):
        x = node.args[0]
        o = node.args[1]
        assert 'base_shape' in x.kwargs
        kernel_size = node.kwargs['kernel_size']
        stride = node.kwargs['stride']
        pad = node.kwargs['pad']

        if kernel_size is None:
            raise RuntimeError(f"Kernel size is a required parameter with no default value.\n"
                               f"Need to provide value.")
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            kh, kw = kernel_size
        elif isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 2
            kh, kw = kernel_size[0], kernel_size[1]
        else:
            raise RuntimeError(f"Invalid type for kernel size: {type(kernel_size)}, value: {kernel_size}")
        computed_oh = (x.shape[-2] + 2 * pad[0] - kh) // stride[0] + 1
        computed_ow = (x.shape[-1] + 2 * pad[1] - kw) // stride[1] + 1
        new_oh = max(computed_oh, 1)
        new_ow = max(computed_ow, 1)
        if computed_oh != new_oh:
            new_kh = (new_oh - 2*pad[0])
            new_kw = (new_ow - 2*pad[1])
            node.kwargs['kernel_size'] = (new_kh, new_kw)
        o.add_attribute('base_shape', o.shape)
        o.set_shape([o.shape[0], x.shape[1], new_oh, new_ow], override=True)
        return node, [x.shape, o.shape]

    def update_coarse_flatten(self, node):
        x = node.args[0]
        o = node.args[1]
        assert 'base_shape' in x.kwargs
        o.add_attribute('base_shape', o.shape)
        new_in_shape = x.shape
        new_out_shape = list(new_in_shape[:node.kwargs['axis']]) + [np.prod(new_in_shape[node.kwargs['axis']:])]
        o.set_shape(new_out_shape, override=True)
        return node, [x.shape, o.shape]

    def update_concat(self, node):
        axis = node.kwargs['axis']
        new_shape = list(node.inputs[0].shape)
        shapes = [node.inputs[0].shape]
        for i in node.inputs[1:]:
            assert 'base_shape' in i.kwargs
            shapes.append(i.shape)
            for idx, d in enumerate(i.shape):
                if idx == axis:
                    new_shape[idx] += d
                else:
                    assert d == new_shape[idx], f"All concatenation shapes must ahve the same value on the non-concat axis\n" \
                                                f"operation: {node.name}\n" \
                                                f"Operand: {i.name}\n" \
                                                f"Input shape: {i.shape}, Expected: {new_shape}, Axis: {axis}\n" \
                                                f"Invalid axis value ({idx}): {d}, Expected: {new_shape[idx]}"
        o = node.outputs[0]
        o.add_attribute('base_shape', o.shape)
        o.set_shape(new_shape, override=True)
        return node, shapes + [o.shape]


    def update_resize(self, node):
        x = node.args[0]
        assert 'base_shape' in x.kwargs
        scales = node.args[1]
        scales.add_attribute('base_shape', scales.shape)
        assert len(scales.init_value) == len(x.shape)
        o = node.args[2]
        new_shape = []
        o.add_attribute('base_shape', o.shape)
        splits = [1, self.oc_split, self.oh_split, self.ow_split]
        for i, d in enumerate(x.shape):
            if o.shape[i] % splits[i] == 0:
                new_shape.append(int(o.shape[i]/splits[i]))
            else:
                new_shape.append(d*scales.init_value[i])
        o.set_shape(new_shape, override=True)
        return node, [x.shape, scales.shape, o.shape]

    def update_split(self, node):
        x = node.args[0]
        assert 'base_shape' in x.kwargs
        split = node.kwargs['split'][0]
        assert all([s == split for s in node.kwargs['split']])
        total_size = x.kwargs['base_shape'][node.kwargs['axis']]

        split_factor = total_size / split
        scaled_split = max(x.shape[node.kwargs['axis']] / split_factor, self.divisor)
        shapes = [x.shape]
        for i, o in enumerate(node.outputs):
            o.add_attribute('base_shape', o.shape)
            new_shape = list(x.shape)
            new_shape[node.kwargs['axis']] = scaled_split
            o.set_shape(new_shape, override=True)
            shapes.append(o.shape)
        return node, shapes

    def update_tensor_reshape(self, node):
        x = node.args[0]
        o = node.args[1]
        assert 'base_shape' in x.kwargs
        o.add_attribute('base_shape', o.shape)
        prev_inp_shape = x.kwargs['base_shape']
        if isinstance(node.args[2], pm.state):
            prev_new_shape = node.args[2].init_value
        else:
            assert isinstance(node.args[2], tuple)
            prev_new_shape = node.args[2]
        new_shape = []
        # Hacky, special case for unsqueezing the first dimension
        if tuple([i for i in o.shape if i != 1]) == tuple([i for i in x.kwargs['base_shape'] if i != 1]):
            for i in range(len(prev_new_shape)):
                rev_idx = len(prev_new_shape) - i - 1
                if prev_new_shape[rev_idx] == 1:
                    new_shape.insert(0, 1)
                else:
                    rev_idx = len(x.shape) - i - 1
                    new_shape.insert(0, x.shape[rev_idx])
        elif tuple([i for i in x.kwargs['base_shape'] if i != 1]) == o.shape:
            new_shape = [d for d in x.shape if d != 1]
        elif len(prev_new_shape) > len(x.shape):
            for i, d in enumerate(prev_new_shape):
                if d == prev_inp_shape[i]:
                    new_shape.append(x.shape[i])
                elif d <= prev_inp_shape[i]:
                    assert prev_inp_shape[i] % d == 0
                    split_factor = prev_inp_shape[i]/d
                    if x.shape[i] % split_factor != 0:
                        if split_factor % (prev_inp_shape[i] / x.shape[i]) == 0:
                            assert split_factor % (prev_inp_shape[i] / x.shape[i]) == 0, f"Split factor {split_factor} not divisible by {prev_inp_shape[i]}"
                            split_factor /= (prev_inp_shape[i] / x.shape[i])
                        else:
                            split_factor = self.fallback_split
                    assert x.shape[i] % split_factor == 0
                    new_shape.append(split_factor)
                    new_shape.append(x.shape[i]/split_factor)
                    break
                else:
                    raise RuntimeError
        else:
            for i, d in enumerate(prev_new_shape):
                if d == prev_inp_shape[i]:
                    new_shape.append(x.shape[i])
                elif d >= prev_inp_shape[i]:
                    new_shape.append(np.prod(x.shape[i:]))
                    break
                else:
                    raise RuntimeError
        assert len(new_shape) == len(prev_new_shape), f"Expected new shape and prev shape to be the same length for {o.name}:\n" \
                                                      f"New shape: {new_shape}\n" \
                                                      f"Previous: {prev_new_shape}\n" \
                                                      f"{x.kwargs['base_shape']} --> {o.kwargs['base_shape']}"
        # if isinstance(node.args[2], pm.state):
        #     node.args[2].kwargs['init_value'] = new_shape
        # else:
        #     node._args[2] = new_shape
        assert np.prod(new_shape) == np.prod(x.shape)
        o.set_shape(new_shape, override=True)
        return node, [x.shape, o.shape]
    def update_tensor_squeeze(self, node):
        x = node.args[0]
        o = node.args[1]
        assert 'base_shape' in x.kwargs
        o.add_attribute('base_shape', o.shape)
        new_in_shape = x.shape
        new_out_shape = []
        for d in range(len(o.shape)):
            new_out_shape.append(new_in_shape[d])
        o.set_shape(new_out_shape, override=True)
        return node, [x.shape, o.shape]

    def update_tensor_transpose(self, node):
        x = node.args[0]
        o = node.args[1]
        axis = node.kwargs['perm']
        assert 'base_shape' in x.kwargs
        new_shape = [x.shape[axis[i]] for i in range(len(x.shape))]
        o.add_attribute('base_shape', o.shape)
        o.set_shape(new_shape, override=True)
        return node, [x.shape, o.shape]


    def update_vision_shapes(self, node):
        if node.op_name in self.ELEMWISE: return self.update_elemwise(node)
        elif node.op_name == "bias_add": return self.update_bias_add(node)
        elif node.op_name in ["depthwise_conv"]: return self.update_conv(node,dw_conv=True)
        elif node.op_name in ["depthwise_conv_bias"]: return self.update_conv_bias(node, dw_conv=True)
        elif node.op_name in ["conv"]: return self.update_conv(node)
        elif node.op_name in ["conv_bias"]: return self.update_conv_bias(node)
        elif node.op_name == "matmul": return self.update_matmul(node)
        elif node.op_name == "gemm": return self.update_gemm(node)
        elif node.op_name == "global_avg_pool": return self.update_global_avg_pool(node)
        elif node.op_name == "avg_pool": return self.update_pool(node)
        elif node.op_name == "max_pool": return self.update_pool(node)
        elif node.op_name == "elem_where": return self.update_elem_where(node)
        elif node.op_name == "concat": return self.update_concat(node)
        elif node.op_name == "resize": return self.update_resize(node)
        elif node.op_name == "split": return self.update_split(node)
        elif node.op_name == "coarse_flatten": return self.update_coarse_flatten(node)
        elif node.op_name == "tensor_reshape": return self.update_tensor_reshape(node)
        elif node.op_name == "tensor_squeeze": return self.update_tensor_squeeze(node)
        elif node.op_name == "tensor_transpose": return self.update_tensor_transpose(node)
        else:
            raise RuntimeError(f"Unsupported vision operation {node.op_name}")

    def update_nchw_shape(self, node):
        prev_shape = node.shape
        assert len(prev_shape) == 4
        new_shape = [prev_shape[0],
                     np.ceil(prev_shape[1]/self.oc_split).astype('int32'),
                     np.ceil(prev_shape[2]/self.oh_split).astype('int32'),
                     np.ceil(prev_shape[3]/self.ow_split).astype('int32'),
                     ]
        node.set_shape(new_shape, override=True)
        node.add_attribute('base_shape', prev_shape)
        return node

    def update_mnp_shape(self, node):
        prev_shape = node.shape
        if len(prev_shape) == 3 and node.kwargs.get("onnx_input", False):
            new_shape = [prev_shape[0],
                         max(np.ceil(prev_shape[1] / self.batch_split).astype('int32'), self.divisor),
                         max(np.ceil(prev_shape[2] / self.oc_split).astype('int32'), self.divisor)
                         ]
            node.set_shape(new_shape, override=True)
            node.add_attribute('base_shape', prev_shape)
        elif len(prev_shape) == 4 and node.kwargs.get("onnx_input", False):
            new_shape = [prev_shape[0],
                         prev_shape[1],
                         prev_shape[2],
                         max(np.ceil(prev_shape[3] / self.oc_split).astype('int32'), self.divisor),
                         ]
            node.set_shape(new_shape, override=True)
            node.add_attribute('base_shape', prev_shape)
        return node


    def initialize_pass(self, node, ctx):
        inputs = []
        for k in node.nodes.keys():
            v = node.nodes[k]
            if isinstance(v, pm.input) and v.kwargs.get('onnx_input', False):
                inputs.append(v)

        if len(inputs) == 1 and len(inputs[0].shape) == 4:
            self.vision_model = True
        else:
            self.vision_model = False

        return node

    def apply_pass(self, node, ctx):
        assert node.op_name not in dir(fused_dnn), f"Batch sizes must be updated prior to fusion, but found fused operation: {node.op_name}"
        if node.kwargs.get('onnx_input', False):
            if self.vision_model:
                return self.update_nchw_shape(node)
            else:
                return self.update_mnp_shape(node)
        elif node.op_name in ALL_TEMPLATES:
            if self.debug:
                self.print_shapes(node)

            if self.vision_model:
                node, shape_list = self.update_vision_shapes(node)
            else:
                node, shape_list = self.update_transformer_shapes(node)
            if any([np.prod(s) == 0 for s in shape_list]):
                raise RuntimeError(f"Incorrectly updated shape to 0 for {node.op_name}\n"
                                   f"Shapes: {shape_list}")

            if self.debug:
                print(f"Updated shapes for {node.op_name}({self.debug}):")
                for i in (node.inputs + node.outputs):
                    print(f"{i.name}: {i.kwargs['base_shape']} --> {i.shape}")
                print()

            if node.op_name not in self.op_counter and node.op_name not in self.ELEMWISE and node.op_name not in self.UNIQUE:
                print(f"{node.op_name}")
            self.shape_tracker[f"{node.op_name}{self.op_counter[node.op_name]}"] = shape_list

            self.op_counter[node.op_name] += 1

            if self.debug:
                print(f"Shapes after:")
                self.print_shapes(node)
        return node

@register_pass
class RenameMultiDimOps(Pass):
    MULTI_DIM_OP1_DEFAULTS = {
        'sgd': -1, 'elem_tanh': -1, 'elem_tanh_grad': -1, 'relu': 4, 'relu_grad': 4, "elem_ceil": -1, "elem_pow": -1,
        "reduce_mean": -1, "reduce_min": -1, "tensor_transpose": -1, "matmul": 2, 'softmax': 2, 'add_add': 3, "elem_add": 4,
        'elem_mul': 4, "elem_div": 4, "elem_sqrt": 4, "concat": 4, "elem_sigmoid": 4, "elem_sub": 4
    }
    MULTI_DIM_OP2_DEFAULTS = { 'elem_div': 4, 'elem_add': 4, 'elem_mul': 4, 'matmul': 4, 'elem_sub': 4}
    MULTI_DIM_OP3_DEFAULTS = { 'mul_add': 1}
    MULTI_OPERAND_OPS = ['tensor_reshape']
    def __init__(self):
        super(RenameMultiDimOps, self).__init__()

    def apply_pass(self, node, ctx):
        init_name = node.op_name
        if init_name in RenameMultiDimOps.MULTI_DIM_OP1_DEFAULTS.keys():
            node = self.rename_op1(node, init_name)
        if init_name in RenameMultiDimOps.MULTI_DIM_OP2_DEFAULTS.keys():
            node = self.rename_op2(node, init_name)
        if init_name in RenameMultiDimOps.MULTI_DIM_OP3_DEFAULTS.keys():
            node = self.rename_op3(node, init_name)
        if node.op_name in RenameMultiDimOps.MULTI_OPERAND_OPS:
            node = self.rename_multi_operand_op(node, init_name)

        return node

    def rename_multi_operand_op(self, node, init_name):
        assert len(node.inputs) == 1 and len(node.outputs) == 1
        node.op_name = f"{node.op_name}{str(len(node.inputs[0].shape))}d{str(len(node.outputs[0].shape))}d"

    def rename_op3(self, node, init_name):
        default_op3_size = RenameMultiDimOps.MULTI_DIM_OP3_DEFAULTS[init_name]
        if len(node.inputs[2].shape) != default_op3_size:
            node.op_name = f"{node.op_name}{str(len(node.inputs[2].shape))}d"
        return node

    def rename_op2(self, node, init_name):
        default_op2_size = RenameMultiDimOps.MULTI_DIM_OP2_DEFAULTS[init_name]

        if len(node.args[1].shape) != default_op2_size:
            if len(node.args[1].shape) == 1 and node.args[1].shape[0] == 1:
                node.op_name = f"{node.op_name}_const"
            else:
                node.op_name = f"{node.op_name}{str(len(node.inputs[1].shape))}d"
        return node

    def rename_op1(self, node, init_name):
        # first do op1
        # default_op1_size = RenameMultiDimOps.MULTI_DIM_OP1_DEFAULTS[init_name]
        # if len(node.inputs[0].shape) != default_op1_size:
        #     node.op_name = f"{node.op_name}{str(len(node.inputs[0].shape))}d"
        default_op1_size = RenameMultiDimOps.MULTI_DIM_OP1_DEFAULTS[init_name]

        if len(node.inputs[0].shape) != default_op1_size:
            if len(node.inputs[0].shape) == 1 and node.inputs[0].shape[0] == 1:
                node.op_name = f"{node.op_name}_const"
            else:
                node.op_name = f"{node.op_name}{str(len(node.inputs[0].shape))}d"

        return node

@register_pass
class UpdateLayout(Pass):
    UNIQUE_OPS = ['conv', 'conv_bias', 'global_average_pool_grad', 'max_pool_grad', 'avg_pool', 'average_pool_grad']
    def __init__(self, current_layout, new_layout):

        assert current_layout == 'nchw'
        assert new_layout == 'nhwc'
        self.layout_map = {}
        self.layout_map[0] = 0
        self.layout_map[1] = 3
        self.layout_map[2] = 2
        self.layout_map[3] = 1
        self.updated_shapes = {}

        super(UpdateLayout, self).__init__()

    def apply_pass(self, node, ctx):
        if isinstance(node, (pm.write, pm.placeholder, pm.temp)) and len(node.shape) == 4:
            node = self.update_shape(node)
        elif node.op_name in UpdateLayout.UNIQUE_OPS:
            node = self.handle_unique_op(node)
        return node


    def update_shape(self, node):
        new_shape = tuple([node.shape[self.layout_map[i]] for i in range(len(node.shape))])
        if node.name in self.updated_shapes:
            assert self.updated_shapes[node.name] == new_shape, f"Invalid shapes for {node.name}:\n" \
                                                                f"Previous shape: {self.updated_shapes[node.name]}\n" \
                                                                f"New shape: {node.shape}"

        self.updated_shapes[node.name] = new_shape
        node.shape = new_shape
        return node

    def handle_unique_op(self, node):
        if node.op_name in ['conv', 'conv_bias']:
            weight = node.inputs[1]
            if weight.name in self.updated_shapes:
                original_shape = self.get_original_shape(self.updated_shapes[weight.name])
            else:
                original_shape = weight.shape
            weight.shape = (original_shape[2], original_shape[3], original_shape[0], original_shape[1])

            activation = node.inputs[0]
            if activation.name not in self.updated_shapes:
                activation = self.update_shape(activation)
            output = node.outputs[0]

            if output.name not in self.updated_shapes:
                output = self.update_shape(output)

        elif node.op_name in ['global_average_pool_grad', 'max_pool_grad', 'average_pool_grad']:
            for i in node.inputs:
                if isinstance(i, pm.Node) and len(i.shape) == 4:
                    if i.name not in self.updated_shapes:
                        i = self.update_shape(i)

            for i in node.outputs:
                if isinstance(i, pm.Node) and len(i.shape) == 4:
                    if i.name not in self.updated_shapes:
                        i = self.update_shape(i)

        return node

    def get_original_shape(self, new_shape):
        rev_map = {v: k for k, v in self.layout_map.items()}
        orig_shape = tuple([new_shape[rev_map[i]] for i in range(len(new_shape))])
        return orig_shape


def unary_op_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.set_shape([batch_size] + list(act.shape[1:]), override=True)
    out.set_shape([batch_size] + list(out.shape[1:]), override=True)
    return node, [act.shape, out.shape]


def mean_var_batch(node, batch_size):
    act = node.inputs[0]
    act.set_shape(tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]]), override=True)
    return node, [act.shape]


def tensor_transpose_batch(node, batch_size):
    assert hasattr(node, "inputs")
    assert hasattr(node, "outputs")
    op1 = node.inputs[0]
    out = node.outputs[0]
    if op1.shape[0] == 1:
        op1.set_shape(tuple([batch_size] + list(op1.shape[1:])), override=True)
    else:
        assert out.shape[0] == 1
        out.set_shape(tuple([batch_size] + list(out.shape[1:])), override=True)
    return node, [op1.shape, out.shape]

def elem_gather_batch(node, batch_size):
    assert hasattr(node, "inputs")
    assert hasattr(node, "outputs")
    op1 = node.inputs[0]
    out = node.outputs[0]
    if len(op1.shape) > 1 and op1.shape[0] == out.shape[0]:
        op1.set_shape(tuple([batch_size] + list(op1.shape[1:])), override=True)
        out.set_shape(tuple([batch_size] + list(out.shape[1:])), override=True)
    return node, [op1.shape, out.shape]

def all_operands_batch(node, batch_size):
    new_shapes = []
    for i in node.inputs + node.outputs:
        if len(i.shape) > 1:
            i.set_shape(tuple([batch_size] + list(i.shape[1:])), override=True)
        new_shapes.append(i.shape)
    return node, new_shapes


BATCH_FUNCS['conv_bias'] = unary_op_batch
BATCH_FUNCS['conv'] = unary_op_batch
BATCH_FUNCS['relu'] = unary_op_batch
BATCH_FUNCS['leaky_relu'] = unary_op_batch
BATCH_FUNCS['elem_tanh'] = unary_op_batch
BATCH_FUNCS['elem_sqrt'] = unary_op_batch
BATCH_FUNCS['elem_sigmoid'] = unary_op_batch
BATCH_FUNCS['elem_pow'] = unary_op_batch
BATCH_FUNCS['softmax'] = unary_op_batch
BATCH_FUNCS['coarse_flatten'] = unary_op_batch
BATCH_FUNCS['batch_norm'] = unary_op_batch
BATCH_FUNCS['reduce_mean'] = unary_op_batch
BATCH_FUNCS['elem_clip'] = unary_op_batch
BATCH_FUNCS['tensor_squeeze'] = unary_op_batch
BATCH_FUNCS['gemm'] = unary_op_batch
BATCH_FUNCS['matmul'] = unary_op_batch
BATCH_FUNCS['tensor_reshape'] = unary_op_batch
BATCH_FUNCS['reshape'] = unary_op_batch
BATCH_FUNCS['resize'] = unary_op_batch
BATCH_FUNCS['global_avg_pool'] = unary_op_batch
BATCH_FUNCS['max_pool'] = unary_op_batch
BATCH_FUNCS['avg_pool'] = unary_op_batch
BATCH_FUNCS['depthwise_conv_bias'] = unary_op_batch
BATCH_FUNCS['gelu'] = unary_op_batch

BATCH_FUNCS['elem_sub'] = all_operands_batch
BATCH_FUNCS['elem_div'] = all_operands_batch
BATCH_FUNCS['elem_mul'] = all_operands_batch
BATCH_FUNCS['split'] = all_operands_batch
BATCH_FUNCS['elem_where'] = all_operands_batch
BATCH_FUNCS['concat'] = all_operands_batch
BATCH_FUNCS['elem_add'] = all_operands_batch


BATCH_FUNCS['elem_gather'] = elem_gather_batch
BATCH_FUNCS['mean_var'] = mean_var_batch
BATCH_FUNCS['tensor_transpose'] = tensor_transpose_batch
BATCH_FUNCS['transpose'] = tensor_transpose_batch

