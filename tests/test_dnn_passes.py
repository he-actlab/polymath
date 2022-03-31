from polymath.srdfg.passes import register_pass, Pass, pass_registry
from polymath import UpdateBatchSize, CollectDNNShapes
import polymath as pm
import numpy as np
import torch
from torch.nn import functional as F
from itertools import product
from .util import get_pad_tuple, dilate_python, _grad_input_padding, \
    cross_entropy_loss, delta_cross_entropy, torch_ce_loss, log_softmax, nll_loss, batchnorm2d_backward
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


@pytest.mark.parametrize('model, fusion_sequences, expected_instances', [
    ('resnet50', [['Conv', 'Relu'],
                  ['Conv', 'Relu', 'MaxPool'],
                  ['Conv', 'Add', 'Relu', 'GlobalAveragePool'],
                  ['Conv', 'Add', 'Relu']],
     {"conv_bias_relu": 32,
      "conv_bias_relu_max_pool": 1,
      "conv_bias_elem_add_relu_global_avg_pool": 1,
      "conv_bias_elem_add_relu": 15,
      }
     ),
    ('resnet18', [['Conv', 'Relu'],
                     ['Conv', 'Relu', 'MaxPool'],
                     ['Conv', 'Add', 'Relu', 'GlobalAveragePool'],
                     ['Conv', 'Add', 'Relu']],
     {"conv_bias_relu": 8,
      "conv_bias_relu_max_pool": 1,
      "conv_bias_elem_add_relu_global_avg_pool": 1,
      "conv_bias_elem_add_relu": 7,
      }
     ),
    ('efficientnet-lite4-11-opt', [['Conv', 'Add'],
                     ['Conv', 'Clip', 'AveragePool'],
                     ['Conv', 'Clip', 'DepthwiseConv',],
                     ['Conv', 'Clip', 'DepthwiseConv', 'Clip',],],
     {"conv_bias_elem_add": 23,
     "conv_bias_elem_clip_avg_pool": 1,
      "conv_bias_elem_clip_depthwise_conv_bias_elem_clip": 30,
      }
     )
])
def test_model_layer_fusion(model, fusion_sequences, expected_instances):

    fpath = f"{ONNX_DNNS}/{model}.onnx"
    graph = pm.from_onnx(fpath)
    unfused_nodes = "\n".join([f"Node: {name}, {n.op_name}" for name, n in graph.nodes.items()])

    fusion_pass = pm.FuseOps(fusion_sequences)
    fused_graph = fusion_pass(graph)
    assert len(expected_instances) == len(fusion_pass.fusion_instances)
    assert all([k in fusion_pass.fusion_instances for k in expected_instances.keys()])
    for k, v in expected_instances.items():
        assert v == fusion_pass.fusion_instances[k]

@pytest.mark.parametrize('model, fusion_sequence, testnum', [
    # ('resnet18', ['Conv', 'Relu'], 0),
    # ('resnet18', ['Conv', 'Add', 'Relu'], 0),
    ('resnet18', ['Conv', 'Relu', 'MaxPool'], 0),
    # ('resnet18', ['Conv', 'Add', 'Relu', 'GlobalAveragePool'], 0),
    # ('resnet50', ['Conv', 'Relu'], 0),
    # ('resnet50', ['Conv', 'Add', 'Relu'], 0),
    # ('resnet50', ['Conv', 'Relu', 'MaxPool'], 0),
    # ('resnet50', ['Conv', 'Add', 'Relu', 'GlobalAvgPool'], 0),
    # ('efficientnet-lite4-11-opt', ['Conv', 'Add',], 0),
    # ('efficientnet-lite4-11-opt', ['Conv', 'Clip', 'AveragePool'], 0),
    # ('efficientnet-lite4-11-opt', ['Conv', 'Clip', 'DepthwiseConv',], 0),
    # ('efficientnet-lite4-11-opt', ['Conv', 'Clip', 'DepthwiseConv', 'Clip'], 0),
])
def test_single_layer_fusion(model, fusion_sequence, testnum):
    fusion_name = '_'.join(fusion_sequence)
    fname = f"{model}_{fusion_name}{testnum}.onnx"
    fpath = f"{BENCH_DIR}/fusion_layers/{fname}"
    graph = pm.from_onnx(fpath)

    unfused_nodes = "\n".join([f"Node: {name}, {n.op_name}" for name, n in graph.nodes.items()])

    fusion_pass = pm.FuseOps([fusion_sequence])
    fused_graph = fusion_pass(graph)
@pytest.mark.parametrize('model_name', [
    # "resnet18",
    # "resnet50",
    # "efficientnet-lite4-11-opt-no-softmax",
    "mobilenet27-opt",
    # "yolov3-opt-static",
    # "bertsquad-12-opt1"
])
def test_conversion(model_name):
    all_fusions = [
        ['Conv', 'Relu'],
        ['Conv', 'LeakyRelu'],
        ['Conv', 'Add', 'Relu'],
        ['Conv', 'Add', 'LeakyRelu'],
        ['Conv', 'Clip', 'DepthwiseConv'],
        ['Conv', 'Clip', 'DepthwiseConv', 'Clip'],
    ]
    fpath = f"{BENCH_DIR}/full_dnns/{model_name}.onnx"
    graph = pm.from_onnx(fpath)
    fusion_pass = pm.FuseOps(all_fusions, pad_conv_constraint=True)
    fused_graph = fusion_pass(graph)




def conv2d_transpose(
    input, weight, stride=1, padding=0, out_pad=0
):
    b, c, h, w = input.shape
    dim_in, dim_out, kh, kw = weight.shape
    sh, sw = stride - 1, stride - 1
    y = input.reshape(b*c,h*w,1,1)
    y = F.pad(y, [0, sw, 0, sh])
    y = y.reshape(b*c,h,w,1+sh,1+sw)
    y = y.permute(0,1,3,2,4)
    y = y.reshape(b, c, h*(1+sh), w*(1+sw))
    ph, pw = kh - padding - 1, kw - padding - 1

    weight = weight.permute(1,0,2,3)
    weight = weight.flip(2, 3)
    # y = F.pad(y, [pw, pw-sw, ph, ph-sh])
    pad_vals = (pw, pw - sw + out_pad, ph, ph - sh + out_pad)
    y = F.pad(y, pad_vals)
    y = F.conv2d(y, weight, padding=0, stride=1)

    return y

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
    torch_res = conv2d_transpose(torch.from_numpy(inp), torch.from_numpy(wgt), stride, pad)
    # np.testing.assert_allclose(tres.numpy(), torch_res.numpy())
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
    # f"{ONNX_LAYERS}/resnet18_globalaveragepool.onnx",
    # f"{ONNX_LAYERS}/resnet18_gemm.onnx",
    # f"{ONNX_LAYERS}/resnet18_flatten.onnx",
    # f"{ONNX_LAYERS}/resnet18_conv.onnx",
    # f"{ONNX_LAYERS}/resnet18_conv_bias.onnx",
    # f"{ONNX_LAYERS}/resnet18_relu.onnx",
    f"{ONNX_DNNS}/resnet18_train.onnx",
])
def test_layer_autodiff(filename):
    batch_size = 4
    train_graph = pm.from_onnx(filename)
    batch_pass = pm.UpdateBatchSize(batch_size, train_graph.name)
    train_graph = batch_pass(train_graph)
    target_layer = "batchnorm_grad"
    target_layers = ["batchnorm_grad", "sgd"]


    train_graph = pm.create_training_graph(train_graph)
    grads = []
    for name, node in train_graph.nodes.items():
        if isinstance(node, pm.Template):
            if node.op_name == target_layer:
                print(f"Layer: {node.op_name}")
                for i in node.inputs:
                    print(f"Input {i.name} - {i.shape}")
                print()
                for i in node.outputs:
                    print(f"Output {i.name} - {i.shape}")
                    grads.append(i.name)
                print()
            elif any([i.name in grads for i in node.inputs]):
                print(f"Node: {node.op_name}\n"
                      f"")
                for i in node.inputs:
                    print(f"Input {i.name} - {i.shape}")
                print()


def test_load_maskrcnn():
    # mrcnn_path = f"{ONNX_DNNS}/mask_rcnn_vision_backbone.onnx"
    mrcnn_path = f"{ONNX_DNNS}/resnet18_train.onnx"
    graph = pm.from_onnx(mrcnn_path)


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
    }
    np_res = nll_loss(inp, tgt)
    np.testing.assert_allclose(np_res, torch_res.numpy())
    x = pm.input(name="data", shape=shape)

    tgt_ = pm.state(name="tgt", init_value=tgt, shape=(shape[0],))

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

def test_bnorm():
    shape = (1, 16, 32, 32)
    grad = torch.rand(shape)
    x = torch.rand(shape)
    scale = torch.rand((shape[1],))
    bias = torch.rand((shape[1],))
    mean = torch.rand((shape[1],))
    var = torch.rand((shape[1],))
    torch_res = batchnorm2d_backward(grad, x, scale, bias)

    grad = grad.numpy()
    x = x.numpy()
    scale = scale.numpy()
    bias = bias.numpy()
    mean = mean.numpy()
    var = var.numpy()
    optimizer = "sgd"
    optimizer_kwargs = {"lr": 0.01}
    pm_x = pm.input(name="x", shape=shape)
    pm_grad = pm.input(name="grad", shape=shape)
    pm_scale = pm.state(name="scale", shape=scale.shape)
    pm_bias = pm.state(name="bias", shape=scale.shape)
    pm_mean = pm.state(name="mean", shape=scale.shape)
    pm_var = pm.state(name="var", shape=scale.shape)
    pm_x_grad = pm.output(name="x_grad", shape=shape)
    pm_scale_grad = pm.output(name="scale_grad", shape=scale.shape)
    pm_b_grad = pm.output(name="bias_grad", shape=bias.shape)

    inp_map = {
        'x': x,
        'grad': grad,
        'scale': scale,
        'bias': bias,
        'mean': mean,
        'var': var,
    }
    graph = pm.batchnorm_grad(pm_x, pm_scale, pm_bias, pm_mean, pm_var, pm_grad, pm_x_grad, pm_scale_grad, pm_b_grad,
                      optimizer, optimizer_kwargs)
    rtol, atol = 1.3e-3, 1e-3
    gout = graph("bias_grad", inp_map)
    np.testing.assert_allclose(gout, torch_res.numpy().reshape(gout.shape), rtol=rtol, atol=atol)

