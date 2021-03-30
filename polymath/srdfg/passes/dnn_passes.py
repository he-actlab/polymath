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
BATCH_FUNCS['coarse_flatten'] = flatten_batch
BATCH_FUNCS['elem_add'] = elem_add_batch
BATCH_FUNCS['global_avg_pool'] = global_avg_pool_batch
BATCH_FUNCS['max_pool'] = max_pool_batch
BATCH_FUNCS['batch_norm'] = batch_norm_batch
BATCH_FUNCS['gemm'] = gemm_batch

