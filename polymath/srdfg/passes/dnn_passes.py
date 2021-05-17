from polymath.srdfg.passes import register_pass, Pass
import polymath as pm
from collections import defaultdict

NON_DNN_NODE_OPS = (pm.write, pm.placeholder, pm.index, pm.var_index,
                    pm.slice_op, pm.func_op, pm.GroupNode, pm.NonLinear)
BATCH_FUNCS = {}

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
        if not isinstance(node, NON_DNN_NODE_OPS) and node.op_name != self.graph_name:
            assert node.op_name in BATCH_FUNCS, f"{node.op_name}"
            node, shape_list = BATCH_FUNCS[node.op_name](node, self.batch_size)
            self.shape_tracker[f"{node.op_name}{self.op_counter[node.op_name]}"] = shape_list
            self.op_counter[node.op_name] += 1
        return node


@register_pass
class RenameMultiDimOps(Pass):
    MULTI_DIM_OPS = ['sgd', 'elem_tanh', 'elem_tanh_grad', 'relu', 'relu_grad']
    def __init__(self):
        super(RenameMultiDimOps, self).__init__()

    def apply_pass(self, node, ctx):
        if node.op_name in RenameMultiDimOps.MULTI_DIM_OPS:
            node = self.rename_op(node)
        return node

    def rename_op(self, node):
        if node.op_name in ["elem_tanh", "elem_tanh_grad", "relu", "relu_grad"] and len(node.inputs[0].shape) == 4:
            op_name = node._op_name
        else:
            op_name = f"{node.op_name}{str(len(node.inputs[0].shape))}d"
        node._op_name = op_name
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


def conv_bias_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]])
    out.shape = tuple([batch_size, out.shape[1], out.shape[2], out.shape[3]])
    return node, [act.shape, out.shape]

def conv_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]])
    out.shape = tuple([batch_size, out.shape[1], out.shape[2], out.shape[3]])
    return node, [act.shape, out.shape]

def relu_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]])
    out.shape = tuple([batch_size, out.shape[1], out.shape[2], out.shape[3]])
    return node, [act.shape, out.shape]

def elem_tanh_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape[0] = batch_size
    out.shape[0] = batch_size
    return node, [act.shape, out.shape]

def batch_norm_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]])
    out.shape = tuple([batch_size, out.shape[1], out.shape[2], out.shape[3]])
    return node, [act.shape, out.shape]

def flatten_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]])
    out.shape = tuple([batch_size, out.shape[1]])
    return node, [act.shape, out.shape]

def elem_add_batch(node, batch_size):
    op1 = node.inputs[0]
    op2 = node.inputs[1]
    out = node.outputs[0]
    op1.shape = tuple([batch_size, op1.shape[1], op1.shape[2], op1.shape[3]])
    op2.shape = tuple([batch_size, op2.shape[1], op2.shape[2], op2.shape[3]])
    out.shape = tuple([batch_size, out.shape[1], out.shape[2], out.shape[3]])
    return node, [op1.shape, op2.shape, out.shape]

def global_avg_pool_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]])
    out.shape = tuple([batch_size, out.shape[1], out.shape[2], out.shape[3]])
    return node, [act.shape, out.shape]

def max_pool_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]])
    out.shape = tuple([batch_size, out.shape[1], out.shape[2], out.shape[3]])
    return node, [act.shape, out.shape]

def avg_pool_batch(node, batch_size):
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1], act.shape[2], act.shape[3]])
    out.shape = tuple([batch_size, out.shape[1], out.shape[2], out.shape[3]])
    return node, [act.shape, out.shape]

def gemm_batch(node, batch_size):
    # TODO: Check for transpose in kwargs
    act = node.inputs[0]
    out = node.outputs[0]
    act.shape = tuple([batch_size, act.shape[1]])
    out.shape = tuple([batch_size, out.shape[1]])
    return node, [act.shape, out.shape]

BATCH_FUNCS['conv_bias'] = conv_bias_batch
BATCH_FUNCS['conv'] = conv_batch
BATCH_FUNCS['relu'] = relu_batch
BATCH_FUNCS['elem_tanh'] = elem_tanh_batch
BATCH_FUNCS['coarse_flatten'] = flatten_batch
BATCH_FUNCS['elem_add'] = elem_add_batch
BATCH_FUNCS['global_avg_pool'] = global_avg_pool_batch
BATCH_FUNCS['max_pool'] = max_pool_batch
BATCH_FUNCS['avg_pool'] = avg_pool_batch
BATCH_FUNCS['batch_norm'] = batch_norm_batch
BATCH_FUNCS['gemm'] = gemm_batch

