import polymath as pm
from collections import OrderedDict
import numpy as np
# cmstack.codegen.dnnweavergen.dnnweaver2.tensorOps
from polymath.codegen.dnnweavergen.dnnweaver2.tensorOps.cnn import conv2D
from polymath.codegen.dnnweavergen.dnnweaver2.compiler import *
from polymath.codegen.dnnweavergen.dnnweaver2.simulator.accelerator import Accelerator
from polymath.codegen.dnnweavergen.dnnweaver2.scalar.dtypes import FQDtype, FixedPoint


@pm.register_pass
class DNNWeaverPass(pm.Pass):

    def __init__(self):
        self.dnnw_ir = {}
        self.last = None
        super(DNNWeaverPass, self).__init__()

    def apply_pass(self, node, ctx):

        if node.op_name in DNNWEAVER_OPS:
            name, dnnweaver_op = DNNWEAVER_OPS[node.op_name](node, self.dnnw_ir)
            self.dnnw_ir[name] = dnnweaver_op
            self.last = name

        return node

    def finalize_pass(self, node, ctx):
        return node

    def package_pass(self, node, ctx):
        last_expr = self.dnnw_ir[self.last]
        pass
        return node
#
# def dnnweaver_avg_pool(node, ctx):
#     data = ctx[node.args[0].name]
#     pool_size = []
#     if not isinstance(node.args[2], pm.Node):
#         pool_size.append(node.args[2])
#     else:
#         pool_size.append(ctx[node.args[2].name])
#
#     if not isinstance(node.args[3], pm.Node):
#         pool_size.append(node.args[3])
#     else:
#         pool_size.append(ctx[node.args[3].name])
#     pool_size = tuple(pool_size)
#
#     if not isinstance(node.args[4], pm.Node):
#         stride = (node.args[4], node.args[4])
#     else:
#         stride = (ctx[node.args[4].name], ctx[node.args[4].name])
#
#     if not isinstance(node.args[5], pm.Node):
#         pad = (node.args[5], node.args[5])
#     else:
#         pad = (ctx[node.args[5].name], ctx[node.args[5].name])
#     p = relay.nn.avg_pool2d(data, pool_size=pool_size, strides=stride, padding=pad)
#
#     return node.args[1].name, p
#
# def dnnweaver_max_pool(node, ctx):
#     return None
#
# def dnnweaver_conv2d(node, ctx):
#     fname = "cmstack.codegen.dnnweavergen.dnnweaver2.tensorOps.cnn.conv2D"
#     with g.as_default():
#         with g.name_scope(scope):
#             dtype_counters['cout'] += 1
#             return conv2D(*args, dtype=FixedPoint(16, dtype_map['cout'][dtype_counters['cout'] - 1]), **kwargs)
#
#     return node.args[3].name, c
#
#
# def dnnweaver_conv2d_bias(node, ctx):
#
#     data = ctx[node.args[0].name]
#     weights = ctx[node.args[1].name]
#     bias = ctx[node.args[2].name]
#     if not isinstance(node.args[4], pm.Node):
#         stride = (node.args[4], node.args[4])
#     else:
#         stride = (ctx[node.args[4].name], ctx[node.args[4].name])
#
#     if not isinstance(node.args[5], pm.Node):
#         pad = (node.args[5], node.args[5])
#     else:
#         pad = (ctx[node.args[5].name], ctx[node.args[5].name])
#     c = relay.nn.conv2d(data, weights, strides=stride, padding=pad)
#     cb = relay.nn.bias_add(c, bias)
#     return node.args[4].name, cb
#
# def dnnweaver_var(node, ctx):
#     var = relay.var(node.name, shape=node.shape, dtype="float32")
#     return node.name, var
#
# def dnnweaver_relu(node, ctx):
#     inp = ctx[node.args[0].name]
#     return node.args[1].name, relay.nn.relu(inp)
#
# def dnnweaver_dense(node, ctx):
#     inp = ctx[node.args[0].name]
#     weights = ctx[node.args[1].name]
#     units = node.args[1].shape[0]
#     d = relay.nn.dense(inp, weights, units=units)
#     return node.args[2].name, d
#
# def dnnweaver_softmax(node, ctx):
#     inp = ctx[node.args[0].name]
#     sm = relay.nn.softmax(inp)
#     return node.args[1].name, sm
#
# def dnnweaver_batch_flatten(node, ctx):
#     inp = ctx[node.args[0].name]
#     bf = relay.nn.batch_flatten(inp)
#     return node.args[1].name, bf
#
DNNWEAVER_OPS = {}
# DNNWEAVER_OPS = {"avg_pool2d": dnnweaver_avg_pool,
#            "max_pool2d": dnnweaver_max_pool,
#            "conv_bias": dnnweaver_conv2d,
#            "input": dnnweaver_var,
#            "state": dnnweaver_var,
#            "relu": dnnweaver_relu,
#            "relu1d": dnnweaver_relu,
#            "softmax": dnnweaver_softmax,
#            "batch_flatten": dnnweaver_batch_flatten,
#            "dense": dnnweaver_dense}

def _normalize_name(name):
    return name.rsplit("/", 1)[-1]