import polymath as pm
from dnnweaver2.graph import Graph
from dnnweaver2 import get_tensor
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint
from dnnweaver2.tensor_ops.cnn import conv2D, add, mul_scalar
import logging
from collections import OrderedDict
import numpy as np



@pm.register_pass
class DNNWeaverPass(pm.Pass):

    def __init__(self, debug=False):
        self.dnnw_ir = {'graph': None, 'dnnweaver_code': []}
        self.last = None
        super(DNNWeaverPass, self).__init__(debug=debug)

    def initialize_pass(self, node, ctx):
        self.dnnw_ir['graph'] = Graph(node.name, None, log_level=logging.INFO)
        return node

    def apply_pass(self, node, ctx):
        if node.op_name in DNNWEAVER_OPS:
            name, dnnweaver_op = DNNWEAVER_OPS[node.op_name](node, self.dnnw_ir)
            self.dnnw_ir[name] = dnnweaver_op
        return node

    def finalize_pass(self, node, ctx):
        return node

    # def package_pass(self, node, ctx):
    #     pass

def dnnweaver_conv2d(node, ctx):
    # Need to add implicit bias here?
    w_dtype = FixedPoint(16,14)
    w_shape = node.args[1].shape
    inp_dtype = get_dnnweaver_var(ctx, node.args[0]).dtype
    c_type = FixedPoint(16, 10)
    biases = get_tensor(shape=(node.args[1].shape[0]),
                        name='biases',
                        dtype=FixedPoint(32, w_dtype.frac_bits + inp_dtype.frac_bits))

    pad = (1, node.args[4], node.args[4],  1)
    strides = (1, node.args[3], node.args[3], 1)
    with ctx['graph'].as_default():
        with ctx['graph'].name_scope(node.name):
            inputs = get_dnnweaver_var(ctx, node.args[0])
            weights = get_dnnweaver_var(ctx, node.args[1])
            weights.shape = convert_conv_shape(weights.shape)
            if weights.shape[-1] != inputs.shape[-1]:
                inputs.shape = convert_conv_shape(inputs.shape)
            return node.name, conv2D(inputs, weights, biases, name=node.name, pad=pad, stride=strides, dtype=c_type)



def dnnweaver_conv2d_bias(node, ctx):
    w_dtype = FixedPoint(16,14)
    inp_dtype = ctx[node.args[0].name].dtype
    c_type = FixedPoint(16, 10)
    pad = (1, node.args[5], node.args[5], 1)
    strides = (1, node.args[4], node.args[4], 1)
    with ctx['graph'].as_default():
        with ctx['graph'].name_scope(node.name):
            inputs = get_dnnweaver_var(ctx, node.args[0])
            weights = get_dnnweaver_var(ctx, node.args[1])
            weights.shape = convert_conv_shape(weights.shape)
            if weights.shape[-1] != inputs.shape[-1]:
                inputs.shape = convert_conv_shape(inputs.shape)


            biases = get_dnnweaver_var(ctx, node.args[2])
            return node.name, conv2D(inputs, weights, biases, name=node.name, pad=pad, stride=strides, dtype=c_type)

def dnnweaver_add(node, ctx):

    with ctx['graph'].as_default():
        with ctx['graph'].name_scope(node.name):
            a = get_dnnweaver_var(ctx, node.args[0])
            b = get_dnnweaver_var(ctx, node.args[1])
            return node.name, add([a, b], name=node.name, out_shape=node.shape, dtype=a.dtype)

def dnnweaver_mul(node, ctx):

    with ctx['graph'].as_default():
        with ctx['graph'].name_scope(node.name):

            a = get_dnnweaver_var(ctx, node.args[0])
            b = get_dnnweaver_var(ctx, node.args[1])
            return node.name, mul_scalar(a, b, name=node.name, dtype=a.dtype)

def dnnweaver_var(node, ctx):
    if isinstance(node, pm.var_index):
        # TODO: Fix var index shape resolution during onnx translation
        assert node.var.name in ctx
        new_tensor = ctx[node.var.name]
    else:
        new_tensor = get_tensor(shape=node.shape, name=node.name, dtype=FixedPoint(16, 10))
    return node.name, new_tensor

def get_dnnweaver_var(ctx, node):
    if node.name not in ctx:
        raise KeyError(f"Unable to find node with in context:\n"
                       f"\tName: {node.name}\n"
                       f"\tOp: {node.op_name}")
    else:
        return ctx[node.name]

def convert_conv_shape(shape):
    lshape = list(shape)
    if len(shape) == 3:
        return tuple(lshape[1:] + [lshape[0]])
    else:
        return tuple([lshape[0]] + lshape[2:4] + [lshape[1]])


DNNWEAVER_OPS = {
            # "avg_pool2d": dnnweaver_avg_pool,
           # "max_pool": dnnweaver_max_pool,
           # "batch_norm": dnnweaver_batch_norm,
           "slice_add": dnnweaver_add,
           "slice_mul": dnnweaver_mul,
           "conv_bias": dnnweaver_conv2d_bias,
           "conv": dnnweaver_conv2d,
           "input": dnnweaver_var,
           "state": dnnweaver_var,
           "var_index": dnnweaver_var,
           # "leaky_relu": dnnweaver_leaky_relu,
           # "softmax": dnnweaver_softmax,
           # "batch_flatten": dnnweaver_batch_flatten,
}

def _normalize_name(name):
    return name.rsplit("/", 1)[-1]