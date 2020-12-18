import polymath as pm
from tests.util import linear, op_counts, logistic, svm, reco, dense, conv,\
    two_layer_dense, np_lenetv2
from pathlib import Path
# import tvm
import pytest
import pprint
import numpy as np
import copy
import onnxruntime as rt
import tvm

import pickle
from onnx import numpy_helper, helper, defs


def lenet(lenet_type="lenet5", coarse=True, debug=False):

    with pm.Node(name="lenet") as graph:
        n = pm.parameter(name="n")
        c = pm.parameter(name="ic")
        ih = pm.parameter(name="ih")
        iw = pm.parameter(name="iw")
        nf1 = pm.parameter(name="nf1")
        kh1 = pm.parameter(name="kh1")
        kw1 = pm.parameter(name="kw1")
        data = pm.input(name="data", shape=(n, c, ih, iw))
        w1 = pm.state(name="w1", shape=(nf1, c, kh1, kw1))
        b1 = pm.state(name="b1", shape=(nf1))

        s1 = pm.parameter(name="s1")
        p1 = pm.parameter(name="p1")
        c1 = pm.output(name="c1", shape=(n, nf1, 28, 28))
        a1 = pm.output(name="a1", shape=(n, nf1, 28, 28))
        l1 = pm.output(name="l1", shape=(n, nf1, 14, 14))

        pm.conv_bias(data, w1, b1, c1, s1, p1)
        pm.elem_tanh(c1, a1, shape=a1.shape)
        pm.avg_pool2d(a1, l1, 2, 2, 2, 0)

        nf2 = pm.parameter(name="nf2")
        kh2 = pm.parameter(name="kh2")
        kw2 = pm.parameter(name="kw2")
        w2 = pm.state(name="w2", shape=(nf2, nf1, kh2, kw2))

        b2 = pm.state(name="b2", shape=(nf2))
        s2 = pm.parameter(name="s2")
        p2 = pm.parameter(name="p2")
        c2 = pm.output(name="c2", shape=(n, nf2, 10, 10))
        a2 = pm.output(name="a2", shape=(n, nf2, 10, 10))
        l2 = pm.output(name="l2", shape=(n, nf2, 5, 5))

        pm.conv_bias(l1, w2, b2, c2, s2, p2)
        pm.elem_tanh(c2, a2, shape=a2.shape)
        pm.avg_pool2d(a2, l2, 2, 2, 2, 0)

        nf3 = pm.parameter(name="nf3")
        kh3 = pm.parameter(name="kh3")
        kw3 = pm.parameter(name="kw3")
        w3 = pm.state(name="w3", shape=(nf3, nf2, kh3, kw3))
        b3 = pm.state(name="b3", shape=(nf3))
        s3 = pm.parameter(name="s3")
        p3 = pm.parameter(name="p3")
        c3 = pm.output(name="c3", shape=(n, nf3, 1, 1))
        a3 = pm.output(name="a3", shape=(n, nf3, 1, 1))

        pm.conv_bias(l2, w3, b3, c3, s3, p3)
        pm.elem_tanh(c3, a3, shape=a3.shape)

        f4 = pm.output(name="f4", shape=(n, nf3))
        pm.coarse_flatten(a3, f4, axis=1, shape=f4.shape)

        m5 = pm.parameter(name="m5")
        n5 = pm.parameter(name="n5")
        f5 = pm.output(name="f5", shape=(n, m5))
        w5 = pm.state(name="w5", shape=(m5, n5))
        # w5 = pm.state(name="w5", shape=(n5, m5))
        a6 = pm.output(name="a5", shape=(n, m5))
        b5 = pm.state(name="b5", shape=(n5,))
        pm.gemm(f4, w5, b5, f5, shape=f5.shape,  alpha=1.0, beta=0.0, transA=False, transB=True)
        pm.elem_tanh(f5, a6, shape=a6.shape)

        m7 = pm.parameter(name="m7")
        n7 = pm.parameter(name="n7")
        f7 = pm.output(name="f7", shape=(n, n7))
        w7 = pm.state(name="w7", shape=(m7, n7))
        # w7 = pm.state(name="w7", shape=(n7, m7))
        b7 = pm.state(name="b7", shape=(n7,))

        pm.gemm(a6, w7, b7, f7, shape=f7.shape, alpha=1.0, beta=0.0, transA=False, transB=False)
        out = pm.output(name="sm")
        pm.softmax(f7, out, axis=1)

    if coarse:
        in_info, keys, out_info = np_lenetv2()
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": 1, "ic": 1, "ih": 32, "iw": 32,
                      "nf1": 6, "kh1": 5, "kw1": 5, "s1": 1, "p1": 0,
                      "nf2": 16, "kh2": 5, "kw2": 5, "s2": 1, "p2": 0,
                      "nf3": 120, "kh3": 5, "kw3": 5, "s3": 1, "p3": 0,
                      "m5": 120, "n5": 84, "m7": 84, "n7": 10
                      }
        shape_val_pass = pm.NormalizeGraph(shape_dict, debug=debug)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = np_lenetv2(lowered=True)
        return new_graph, in_info, out_info, keys

def get_onnx_lenet(inp_info):
    BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")
    filename = f"lenet.onnx"
    filepath = f"{BENCH_DIR}/full_dnns/{filename}"
    assert Path(filepath).exists()
    graph = pm.from_onnx(filepath)
    tvm_code = pm.generate_tvm(graph, inp_info, "")
    return tvm_code

def tvm_lenet(num_classes=10, data_shape=(1, 1, 32, 32),
               dtype='float32', alpha=1.0, is_shallow=False):
    from tvm import relay
    from tvm.relay.testing import layers

    """Function to construct a Lenet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    conv1 = layers.conv2d(data=data,channels=6, kernel_size=(5,5), name='conv1')
    conv1 = relay.tanh(conv1)
    pool2 = relay.nn.avg_pool2d(conv1, pool_size=(2,2), strides=(2,2))
    conv3 = layers.conv2d(data=pool2, channels=16, kernel_size=(5,5), name='conv3')
    conv3 = relay.tanh(conv3)
    pool4 = relay.nn.avg_pool2d(conv3, pool_size=(2,2), strides=(2,2))

    conv5 = layers.conv2d(data=pool4, channels=120, kernel_size=(5,5), name='conv5')
    conv5 = relay.tanh(conv5)
    # Temp
    flattened6 = relay.reshape(conv5, (1, 120))
    # flattened6 = relay.nn.batch_flatten(conv5)
    fcw7 = relay.var('fc7_weight', shape=(120, 84))
    fcw7 = relay.transpose(fcw7)
    fc7 = relay.nn.dense(data=flattened6, weight=fcw7, units=84)
    fc7 = relay.tanh(fc7)

    fcw8 = relay.var('fc6_weight', shape=(84, 10))
    fcw8 = relay.transpose(fcw8)

    fc8 = relay.nn.dense(data=fc7, weight=fcw8, units=10)

    softmax = relay.nn.softmax(data=fc8)
    fn = relay.Function(relay.analysis.free_vars(softmax), softmax)
    return fn




if __name__ == "__main__":

    # Get PolyMath Lenet Definition
    graph, inp_info, out_info, key = lenet(coarse=True)


    #
    # # Load Lenet-5 Definition from ONNX
    onnx_pm_mod = get_onnx_lenet(inp_info)
    onnx_pm_mod = tvm.IRModule.from_expr(onnx_pm_mod)
    onnx_pm_mod = tvm.relay.transform.InferType()(onnx_pm_mod)

    # # Load native TVM Lenet
    net = tvm_lenet()
    mod = tvm.IRModule.from_expr(net)
    mod = tvm.relay.transform.InferType()(mod)
    # print(f"--------------------------------------------------TVM-Compiled Relay IR--------------------------------------------------\n")
    # print(mod)
    # print(f"--------------------------------------------------PolyMath-Compiled Relay IR--------------------------------------------------\n")
    # print(pm_mod)
    # print(f"--------------------------------------------------ONNX-mg-DFG-Compiled Relay IR--------------------------------------------------\n")
    # print(onnx_pm_mod)
