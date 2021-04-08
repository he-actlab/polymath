from polymath.srdfg.passes import register_pass, Pass, pass_registry
from polymath import UpdateBatchSize, CollectDNNShapes
import polymath as pm
from polymath.srdfg.templates.template_utils import dilate, _get_conv_output_shape
import numpy as np
import torch
from torch.nn import functional as F
from itertools import product
from .util import get_pad_tuple, dilate_python, _grad_input_padding, \
    cross_entropy_loss, delta_cross_entropy, torch_ce_loss, log_softmax, nll_loss
import pytest
from pathlib import Path

BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")
CWD = Path(f"{__file__}").parent

ONNX_DNNS = f"{BENCH_DIR}/full_dnns/"
ONNX_LAYERS = f"{BENCH_DIR}/layers/"


def test_resnet18_batchsize():
    batch_size = 32
    resnet18_path = f"{ONNX_DNNS}/resnet18.onnx"
    resnet18_graph = pm.from_onnx(resnet18_path)

    batch_size_pass = UpdateBatchSize(batch_size, resnet18_graph.op_name)
    updated_resnet18 = batch_size_pass(resnet18_graph)
    test_op_shape_pass = CollectDNNShapes()
    _ = test_op_shape_pass(updated_resnet18)

    ref_resnet18_path = f"{ONNX_DNNS}/resnet18_batch{batch_size}.onnx"
    #
    ref_resnet18_graph = pm.from_onnx(ref_resnet18_path)

    ref_op_shape_pass = CollectDNNShapes()
    _ = ref_op_shape_pass(ref_resnet18_graph)
    ref_shapes = ref_op_shape_pass.shape_tracker
    test_shapes = test_op_shape_pass.shape_tracker

    assert len(list(ref_shapes.keys())) == len(list(test_shapes.keys())), f"Reference keys: {list(ref_shapes.keys())}\n" \
                                                                          f"Test keys: {list(test_shapes.keys())}"
    for op_name, shapes in ref_shapes.items():
        for idx, s in enumerate(shapes):
            assert isinstance(s, tuple) and s == test_shapes[op_name][idx]


@pytest.mark.parametrize('inp_shape, wgt_shape, stride, pad',[
    ((1, 3, 18, 18), (3, 10, 3, 3), 2, 1),
])
def test_conv2d_transpose_shapes(inp_shape, wgt_shape, stride, pad):
    groups = 1
    dilation = 1
    out_pad = 0
    inp = np.random.randint(-15, 15, np.prod(inp_shape)).reshape(inp_shape)
    wgt = np.random.randint(-15, 15, np.prod(wgt_shape)).reshape(wgt_shape)
    torch_res = F.conv_transpose2d(torch.from_numpy(inp), torch.from_numpy(wgt),
                                   stride=stride, padding=pad)

    info = {
        'data': inp,
        'w': wgt,
    }
    N, C, H, W = inp.shape

    x = pm.input(name="data", shape=inp_shape)
    w = pm.state(name="w", shape=wgt_shape)
    out = pm.output(name="out")

    graph = pm.conv_transpose(x, w, out, stride, pad)
    #
    tres = graph("out", info)

    np.testing.assert_allclose(tres, torch_res.numpy())

@pytest.mark.parametrize('filename',[
    f"{ONNX_LAYERS}/resnet18_gemm.onnx",
])
def test_layer_autodiff(filename):

    graph = pm.from_onnx(filename, lower=False)
    autodiff_pass = pm.AutoDiffGraph("cross_entropy", "sgd", {"lr": 0.01})
    train_graph = autodiff_pass(graph)
    pm.pb_store(train_graph, f"{BENCH_DIR}")
    pm.pb_load(f"{BENCH_DIR}/{train_graph.name}.srdfg")

    # for name, node in train_graph.nodes.items():
    #     print(f"{node.op_name} - {name}")


@pytest.mark.parametrize('shape',[
    (3, 100,),
])
def test_log_softmax(shape):
    inp = np.random.uniform(-15, 15, np.prod(shape)).reshape(shape)
    torch_res = F.log_softmax(torch.from_numpy(inp))
    info = {
        'data': inp,
    }
    np_res = log_softmax(inp)
    np.testing.assert_allclose(np_res, torch_res.numpy())
    x = pm.input(name="data", shape=shape)
    lsmx = pm.output(name="lsmx")

    graph = pm.log_softmax(x, lsmx, axis=1)
    tres = graph("lsmx", info)

    np.testing.assert_allclose(tres, torch_res.numpy())

@pytest.mark.parametrize('shape',[
    (3, 100,),
])
def test_nll_loss(shape):
    inp = np.random.uniform(-15, 15, np.prod(shape)).reshape(shape)
    tgt = np.random.randint(0, 15, np.prod(shape[0]))

    torch_res = F.nll_loss(torch.from_numpy(inp), torch.from_numpy(tgt))
    info = {
        'data': inp,
        'tgt': tgt,
    }
    np_res = nll_loss(inp, tgt)
    np.testing.assert_allclose(np_res, torch_res.numpy())
    x = pm.input(name="data", shape=shape)
    tgt_ = pm.state(name="tgt", shape=(shape[0],))

    loss = pm.output(name="loss")
    #
    graph = pm.nll_loss(x, tgt_, loss)
    tres = graph("loss", info)
    #

    np.testing.assert_allclose(tres, np_res)

@pytest.mark.parametrize('shape',[
    (3, 100,),
])
def test_loss(shape):
    inp = np.random.uniform(-15, 15, np.prod(shape)).reshape(shape)
    tgt = np.random.randint(0, 15, np.prod(shape[0]))

    torch_res = F.cross_entropy(torch.from_numpy(inp), torch.from_numpy(tgt))
    info = {
        'data': inp,
        'tgt': tgt,
    }
    np_res = torch_ce_loss(inp, tgt)
    np.testing.assert_allclose(np_res, torch_res.numpy())
    x = pm.input(name="data", shape=shape)
    tgt_ = pm.state(name="tgt", shape=(shape[0],))

    loss = pm.output(name="loss")

    graph = pm.cross_entropy_loss(x, tgt_, loss)
    tres = graph("loss", info)


    np.testing.assert_allclose(tres, np_res)

def test_autodiff():
    resnet18_path = f"{ONNX_DNNS}/resnet18.onnx"
    resnet18_graph = pm.from_onnx(resnet18_path)
    #
    # autodiff_pass = pm.AutoDiffGraph()
    # autodiff_graph = autodiff_pass(resnet18_graph)

