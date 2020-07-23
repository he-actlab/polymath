import polymath as pm
from collections import OrderedDict
import numpy as np
import tvm
from tvm import relay



@pm.register_pass
class TVMPass(pm.Pass):

    def __init__(self):
        self.tvm_ir = {}
        self.last = None
        super(TVMPass, self).__init__()

    def apply_pass(self, node, ctx):

        if node.op_name in TVM_OPS:
            name, tvm_op = TVM_OPS[node.op_name](node, self.tvm_ir)
            self.tvm_ir[name] = tvm_op
            self.last = name

        return node

    def finalize_pass(self, node, ctx):

        return node

    def package_pass(self, node, ctx):
        last_expr = self.tvm_ir[self.last]
        self.tvm_ir['tvm_code'] = relay.Function(relay.analysis.free_vars(last_expr), last_expr)
        return node

def tvm_avg_pool(node, ctx):
    data = ctx[node.args[0].name]
    pool_size = []
    if not isinstance(node.args[2], pm.Node):
        pool_size.append(node.args[2])
    else:
        pool_size.append(ctx[node.args[2].name])

    if not isinstance(node.args[3], pm.Node):
        pool_size.append(node.args[3])
    else:
        pool_size.append(ctx[node.args[3].name])
    pool_size = tuple(pool_size)

    if not isinstance(node.args[4], pm.Node):
        stride = (node.args[4], node.args[4])
    else:
        stride = (ctx[node.args[4].name], ctx[node.args[4].name])

    if not isinstance(node.args[5], pm.Node):
        pad = (node.args[5], node.args[5])
    else:
        pad = (ctx[node.args[5].name], ctx[node.args[5].name])
    p = relay.nn.avg_pool2d(data, pool_size=pool_size, strides=stride, padding=pad)

    return node.args[1].name, p

def tvm_max_pool(node, ctx):
    return None

def tvm_conv2d(node, ctx):

    data = ctx[node.args[0].name]
    weights = ctx[node.args[1].name]
    if not isinstance(node.args[4], pm.Node):
        stride = (node.args[4], node.args[4])
    else:
        stride = (ctx[node.args[4].name], ctx[node.args[4].name])

    if not isinstance(node.args[5], pm.Node):
        pad = (node.args[5], node.args[5])
    else:
        pad = (ctx[node.args[5].name], ctx[node.args[5].name])
    c = relay.nn.conv2d(data, weights, strides=stride, padding=pad)
    return node.args[3].name, c


def tvm_conv2d_bias(node, ctx):

    data = ctx[node.args[0].name]
    weights = ctx[node.args[1].name]
    bias = ctx[node.args[2].name]
    if not isinstance(node.args[4], pm.Node):
        stride = (node.args[4], node.args[4])
    else:
        stride = (ctx[node.args[4].name], ctx[node.args[4].name])

    if not isinstance(node.args[5], pm.Node):
        pad = (node.args[5], node.args[5])
    else:
        pad = (ctx[node.args[5].name], ctx[node.args[5].name])
    c = relay.nn.conv2d(data, weights, strides=stride, padding=pad)
    cb = relay.nn.bias_add(c, bias)
    return node.args[4].name, cb

def tvm_var(node, ctx):
    var = relay.var(node.name, shape=node.shape, dtype="float32")
    return node.name, var

def tvm_relu(node, ctx):
    inp = ctx[node.args[0].name]
    return node.args[1].name, relay.nn.relu(inp)

def tvm_dense(node, ctx):
    inp = ctx[node.args[0].name]
    weights = ctx[node.args[1].name]
    units = node.args[1].shape[0]
    d = relay.nn.dense(inp, weights, units=units)
    return node.args[2].name, d

def tvm_softmax(node, ctx):
    inp = ctx[node.args[0].name]
    sm = relay.nn.softmax(inp)
    return node.args[1].name, sm

def tvm_batch_flatten(node, ctx):
    inp = ctx[node.args[0].name]
    bf = relay.nn.batch_flatten(inp)
    return node.args[1].name, bf

TVM_OPS = {"avg_pool2d": tvm_avg_pool,
           "avg_pool": tvm_avg_pool,
           "max_pool2d": tvm_max_pool,
           "max_pool": tvm_max_pool,
           "conv_bias": tvm_conv2d,
           "conv": tvm_conv2d,
           "input": tvm_var,
           "state": tvm_var,
           "relu": tvm_relu,
           "relu1d": tvm_relu,
           "softmax": tvm_softmax,
           "batch_flatten": tvm_batch_flatten,
           "dense": tvm_dense,
           "matmul": tvm_dense}

def _normalize_name(name):
    return name.rsplit("/", 1)[-1]
