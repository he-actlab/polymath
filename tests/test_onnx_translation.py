import polymath as pm
from tests.util import linear, op_counts, logistic, svm, reco,\
    dense, conv, two_layer_dense, pooling
from pathlib import Path
import islpy as isl

import pytest
import pprint
import numpy as np
import copy
import onnxruntime as rt
from onnx import numpy_helper, helper, defs
BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")

ONNX_FILE_DIR = Path(f"{Path(__file__).parent}/onnx_examples")

def generate_test_inputs(n):
    n = int(n)
    x = np.random.randint(-3,3, n)
    w = np.random.randint(-3,3, n)
    y = np.random.randint(-3,3, 1)
    return x, w, y

def test_load_files():
    for f in ONNX_FILE_DIR.iterdir():
        _ = pm.from_onnx(str(f))

@pytest.mark.parametrize('benchmark_name, feature_size',[
    ("linear", ['54']),
    # ("backprop", ['8', '16', '3'])
])
def test_convert_benchmarks(benchmark_name, feature_size):
    filename = f"{benchmark_name}{'-'.join(feature_size)}.onnx"
    filepath = f"{BENCH_DIR}/{benchmark_name}/{filename}"
    assert Path(filepath).exists()
    graph = pm.from_onnx(filepath)
    x, w, y = generate_test_inputs(feature_size[0])
    # np_res = w - (x.dot(w) - y)*x
    # np_res = x.dot(w)
    np_res = x*w
    # pprint.pprint(graph.nodes.keys())
    pm_res = graph("Mul_1:0", {"y:0": y, "x:0": x, "W:0": w})
    # print(pm_res)
    # print(np_res)
    # np.testing.assert_allclose(np_res, pm_res)


@pytest.mark.parametrize('m_',[
    3
])
def test_load_linear_regressor(m_):
    shape_dict = {"m": m_}
    m = pm.parameter("m")
    mu = pm.parameter(name="mu", default=1.0)
    x = pm.input("x", shape=(m))
    y = pm.input("y")
    w = pm.state("w", shape=(m))

    graph = pm.linear_regressor_train(x, w, y, mu, m)
    test_graph, input_info, out_info, keys = linear(m=m_, coarse=True)
    assert len(test_graph.nodes.keys()) == len(graph.nodes.keys())
    pprint.pprint(op_counts(test_graph))
    pprint.pprint(op_counts(graph))
    assert op_counts(test_graph) == op_counts(graph)

    shape_val_pass = pm.NormalizeGraph(shape_dict)
    new_graph = shape_val_pass(graph)
    test_res = new_graph(keys, input_info)
    np.testing.assert_allclose(test_res, out_info["w"])

    test_graph_lowered, input_info, new_out_info, keys = linear(m=m_)
    flatten_pass = pm.Lower({})
    test_flatten_pass = pm.Lower({})
    flattened_g = flatten_pass(new_graph)
    ref_lowered = test_flatten_pass(test_graph_lowered, {})
    assert len(ref_lowered.nodes.keys()) == len(flattened_g.nodes.keys())
    assert op_counts(ref_lowered) == op_counts(flattened_g)

    all_vals = flattened_g(keys, input_info)
    np.testing.assert_allclose(new_out_info["w"], all_vals)

@pytest.mark.parametrize('m_',[
    3
])
def test_load_nested_linear_regressor(m_):
    shape_dict = {"m": m_}
    with pm.Node(name="nested_linear") as graph:
        m = pm.parameter(name="m")
        mu = pm.parameter(name="mu", default=1.0)
        x = pm.input("x", shape=(m))
        y = pm.input("y")
        w = pm.state("w", shape=(m))
        pm.linear_regressor_train(x, w, y, mu, m, name="linear_regressor")
        j = pm.index(0, m-1, name="j")
        tw = (w[j] - 4).set_name("tw")

    test_graph, input_info, out_info, keys = linear(m=m_, coarse=True)
    shape_val_pass = pm.NormalizeGraph(shape_dict)
    new_graph = shape_val_pass(graph)
    test_res = new_graph("tw", input_info)
    np.testing.assert_allclose(test_res, (out_info["w"] - 4))

    ref_graph, input_info, new_out_info, keys = linear(m=m_)
    flatten_pass = pm.Lower({})
    test_flatten_pass = pm.Lower({})
    keys = [f"tw/tw({i},)" for i in range(m_)]

    flattened_g = flatten_pass(new_graph)
    pprint.pprint(list(flattened_g.nodes.keys()))
    all_vals = flattened_g(keys, input_info)

@pytest.mark.parametrize('m',[
    55
])
def test_translate_linear_regressor(m):
    fpath = f"{ONNX_FILE_DIR}/linear_{m}.onnx"
    shape_dict = {"m": m}
    graph = pm.from_onnx(fpath)
    test_graph, input_info, out_info, keys = linear(m=m, coarse=True)
    tinput_info = copy.deepcopy(input_info)
    tkeys = copy.deepcopy(keys)
    test_res = test_graph(tkeys, tinput_info)
    np.testing.assert_allclose(test_res, (out_info["w"]))
    onx_input_info = copy.deepcopy(input_info)
    onnx_res = graph(keys, onx_input_info)
    np.testing.assert_allclose(onnx_res, (out_info["w"]))

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}{m}_tabla.json"
    tabla_ir = pm.generate_tabla(graph,
                                  shape_dict,
                                  tabla_path)


@pytest.mark.parametrize('m',[
    54
])
def test_translate_logistic_regression(m):
    fpath = f"{ONNX_FILE_DIR}/logreg_{m}.onnx"
    shape_dict = {"m": m}
    graph = pm.from_onnx(fpath, infer_shapes=False)
    test_graph, input_info, out_info, keys = logistic(m_=m, coarse=True)
    tinput_info = copy.deepcopy(input_info)
    tkeys = copy.deepcopy(keys)
    test_res = test_graph(tkeys, tinput_info)
    np.testing.assert_allclose(test_res, (out_info["w"]))
    onx_input_info = copy.deepcopy(input_info)
    onnx_res = graph(keys, onx_input_info)
    np.testing.assert_allclose(onnx_res, (out_info["w"]))

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}{m}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path)


@pytest.mark.parametrize('m',[
    54
])
def test_translate_svm(m):
    fpath = f"{ONNX_FILE_DIR}/svm_{m}.onnx"
    shape_dict = {"m": m}
    graph = pm.from_onnx(fpath)
    test_graph, input_info, out_info, keys = svm(m=m, coarse=True)
    tinput_info = copy.deepcopy(input_info)
    tkeys = copy.deepcopy(keys)
    test_res = test_graph(tkeys, tinput_info)
    np.testing.assert_allclose(test_res, (out_info["w"]))
    onx_input_info = copy.deepcopy(input_info)
    onnx_keys = copy.deepcopy(keys)

    onnx_res = graph(onnx_keys, onx_input_info)
    np.testing.assert_allclose(onnx_res, (out_info["w"]))

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}{m}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path)

@pytest.mark.parametrize('m, n, k', [
    (3, 3, 2),
])
def test_translate_reco(m, n, k):
    shape_dict = {"m": m, "n": n, "k": k}
    test_graph, input_info, out_info, keys = reco(m_=m, n_=n, k_=k, coarse=True)

@pytest.mark.parametrize('x_shape, w_shape', [
    ((4,), (5, 4)),
])
def test_translate_dense(x_shape, w_shape):

    graph, input_info, out_info, keys = dense(x_shape, w_shape, coarse=True, debug_matrix=True)
    tinput_info = copy.deepcopy(input_info)
    res0 = graph("y", tinput_info)

    np.testing.assert_allclose(res0, out_info["y"])

    graph, input_info, out_info, keys = dense(x_shape, w_shape, coarse=False, debug_matrix=True)

    lower_pass = pm.Lower({})
    lowered_graph = lower_pass(graph)
    res = lowered_graph(keys, input_info)
    np.testing.assert_allclose(np.asarray(res), out_info["y"])


@pytest.mark.parametrize('x1_shape, w1_shape, w2_shape', [
    ((4,), (5, 4), (3, 5)),
])
def test_translate_multi_dense(x1_shape, w1_shape, w2_shape):

    graph, input_info, out_info, keys = two_layer_dense(x1_shape, w1_shape, w2_shape, coarse=True, debug_matrix=True)

    tinput_info = copy.deepcopy(input_info)
    res0 = graph(keys, tinput_info)
    np.testing.assert_allclose(res0, out_info["y"])

    graph, input_info, out_info, keys = two_layer_dense(x1_shape, w1_shape, w2_shape, coarse=False, debug_matrix=True)

    lower_pass = pm.Lower({})
    lowered_graph = lower_pass(graph)
    res = lowered_graph(keys, input_info)
    np.testing.assert_allclose(np.asarray(res), out_info["y"])

@pytest.mark.parametrize('data_shape, kernel_shape, stride', [
    ((1, 6, 28, 28), (2, 2), 2),
])
def test_avg_pool(data_shape, kernel_shape, stride):
    data = np.random.randint(0, 5, data_shape)
    tout = pooling(data, kernel_shape[0], kernel_shape[1], stride=stride)

    out = pm.output(name="out")
    n = pm.parameter("ns")
    ic = pm.parameter("ic")
    ih = pm.parameter("ih")
    iw = pm.parameter("iw")
    kh = pm.parameter("kh")
    kw = pm.parameter("kw")
    x = pm.input(name="data", shape=(n, ic, ih, iw))

    g = pm.avg_pool2d(x, out, kh, kw, stride, 0)
    inp_info = {}
    inp_info["data"] = data
    inp_info["kh"] = kernel_shape[0]
    inp_info["kw"] = kernel_shape[1]
    test_out = g("out", inp_info)
    np.testing.assert_allclose(test_out, tout)

@pytest.mark.parametrize('x_shape, w_shape, params', [
    ((1, 1, 32, 32), (6, 1, 5, 5), {"stride": 1, "pad": 0}),
    ((1, 1, 4, 4), (2, 1, 2, 2), {"stride": 2, "pad": 1}),
    ((1, 1, 32, 32), (2, 1, 4, 4), {"stride": 2, "pad": 1}),
])
def test_translate_conv(x_shape, w_shape, params):
    shape_dict = {"n": x_shape[0], "c": x_shape[1], "ih": x_shape[2], "iw": x_shape[3],
                  "nf": w_shape[0], "kh": w_shape[2], "kw": w_shape[3],
                  "stride": params["stride"], "pad": params["pad"]}


    _, input_info, out_info, keys = conv(x_shape, w_shape, params, coarse=True, debug_matrix=True)

    n = pm.parameter(name="n")
    c = pm.parameter(name="ic")
    ih = pm.parameter(name="ih")
    iw = pm.parameter(name="iw")
    nf = pm.parameter(name="nf")
    kh = pm.parameter(name="kh")
    kw = pm.parameter(name="kw")
    x = pm.input(name="data", shape=(n, c, ih, iw))
    w = pm.state(name="w", shape=(nf, c, kh, kw))
    b = pm.state(name="bias", shape=(nf))
    stride = pm.parameter(name="stride")
    pad = pm.parameter(name="pad")
    out = pm.output(name="out")
    graph = pm.conv(x, w, b, out, stride, pad)
    tinput_info = copy.deepcopy(input_info)

    res0 = graph("out", tinput_info)
    np.testing.assert_allclose(res0, out_info["out"])
    input_info['populate'] = False
    normalize_pass = pm.NormalizeGraph(input_info)
    normalized = normalize_pass(graph)

@pytest.mark.parametrize('x_shape', [
    (5, 5, 8, 8),
])
def test_translate_flatten(x_shape):
    x = np.random.randint(0, 5, x_shape)
    data = pm.input("x", shape=x.shape)
    out = pm.output("out")

    g = pm.batch_flatten(data, out)

    res = g("out", {"x": x})
    np.testing.assert_allclose(res, x.reshape(-1))

@pytest.mark.parametrize('x_shape', [
    (10,),
])
def test_translate_reduce_sum(x_shape):
    data = np.random.randint(-3, 3, x_shape)
    np_res = np.sum(data)
    graph = pm.Node("reduce")
    pm_data = pm.input(name="a", shape=x_shape, graph=graph)
    axis = 0
    keepdims = 0

    with graph:
        pm_graph = pm.reduce_sum(pm_data, axis, keepdims, x_shape[0], name="out")
    pm_res = graph("out", {"a": data})
    np.testing.assert_allclose(pm_res, np_res)


@pytest.mark.parametrize('x_shape', [
    (10,),
])
def test_translate_elem_mul(x_shape):
    a = np.random.randint(-3, 3, x_shape)
    b = np.random.randint(-3, 3, x_shape)
    np_res = a * b
    graph = pm.Node("elem_mul")

    pm_a = pm.input(name="a", shape=x_shape, graph=graph)
    pm_b = pm.input(name="b", shape=x_shape, graph=graph)
    with graph:
        pm_output = pm.elem_mul(pm_a, pm_b, x_shape[0], name="out")
    pm_res = graph("out", {"a": a, "b": b})
    np.testing.assert_allclose(pm_res, np_res)


@pytest.mark.parametrize('x_shape', [
    (10,),
])
def test_translate_vmul(x_shape):
    a = np.random.randint(-3, 3, x_shape)
    b = np.random.randint(-3, 3, x_shape)
    np_res = a.dot(b)
    with pm.Node("vmul") as pm_graph:
        pm_a = pm.input(name="a", shape=x_shape)
        pm_b = pm.input(name="b", shape=x_shape)
        outp = pm.elem_mul(pm_a, pm_b, x_shape[0])
        _ = pm.reduce_sum(outp, 0, 0, x_shape[0], name="out")

    pm_res = pm_graph("out", {"a": a, "b": b})
    np.testing.assert_allclose(pm_res, np_res)

@pytest.mark.parametrize('x_shape', [
    (1024,),
])
def test_translate_softmax(x_shape):
    softmax = lambda i: np.exp(i) / np.sum(np.exp(i))
    x = np.random.randint(0, 5, x_shape)
    data = pm.input("x", shape=x.shape)
    out = pm.output("out")
    g = pm.softmax(data, out)
    res = g("out", {"x": x})
    np_res = softmax(x)
    np.testing.assert_allclose(np_res, res)

