from pathlib import Path
import polymath as pm
import numpy as np
import pytest
from .util import logistic, linear, reco, svm, compare_tabla_dfg, set_shape_and_lower,\
    unwound_fft, backprop, conv, lenet
import pickle

CWD = Path(f"{__file__}").parent
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"

def test_linear_serialize():

    with pm.Node(name="linear_reg") as graph:
        m = pm.placeholder("m")
        x_ = pm.placeholder("x", shape=(m))
        y_ = pm.placeholder("y")
        w_ = pm.placeholder("w", shape=(m))
        mu = pm.parameter(name="mu", default=1.0)
        i = pm.index(0, (m-1).set_name("m-1"), name="i")
        h = pm.sum([i], (x_[i] * w_[i]).set_name("x*w"), name="h")
        d = (h-y_).set_name("h-y")
        g = (d*x_[i]).set_name("d*x")
        w_ = ((w_[i]) - (mu*g[i])).set_name("w_out")
    x = np.random.randint(1, 5, 5)
    y = np.random.randint(1, 5, 1)[0]
    w = np.random.randint(1, 5, 5)
    graph_res = graph("w_out", {"x": x, "y": y, "w": w})
    actual_res = w - ((np.sum(x*w) - y)*x)*1.0
    np.testing.assert_allclose(graph_res, actual_res)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pm.pb_store(graph, full_path)

def test_linear_deserialize():

    graph_name = "linear_reg1"
    with pm.Node(name=graph_name) as graph:
        m = pm.placeholder("m")
        x_ = pm.placeholder("x", shape=(m))
        y_ = pm.placeholder("y")
        w_ = pm.placeholder("w", shape=(m))
        mu = pm.parameter(name="mu", default=1.0)
        i = pm.index(0, (m-1).set_name("m-1"), name="i")
        h = pm.sum([i], (x_[i] * w_[i]).set_name("x*w"), name="h")
        d = (h-y_).set_name("h-y")
        g = (d*x_[i]).set_name("d*x")
        mug = (mu*g[i]).set_name("mu*g[i]")
        w_ = ((w_[i]) - mug).set_name("w_out")
    x = np.random.randint(0, 10, 10)
    y = np.random.randint(0, 10, 1)[0]
    w = np.random.randint(0, 10, 10)

    graph_res = graph("w_out", {"x": x, "y": y, "w": w})
    actual_res = w - ((np.sum(x*w) - y)*x)*1.0

    np.testing.assert_allclose(graph_res, actual_res)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/{graph_name}.srdfg"
    pm.pb_store(graph, full_path)
    node = pm.pb_load(pb_path)
    new_graph_res = node("w_out", {"x": x, "y": y, "w": w})
    np.testing.assert_allclose(graph_res, new_graph_res)
    np.testing.assert_allclose(actual_res, new_graph_res)

    # TODO: Figure out why hashed nodes are broken
    # assert (node.func_hash()) == (graph.func_hash())

@pytest.mark.parametrize('m_',[
    55
])
def test_tabla_linear(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = linear(m=m_, coarse=True)
    lgraph, input_info, out_info, keys = linear(m=m_, coarse=False)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    graph_name = f"{graph.name}_{m_}"
    tabla_path = f"{full_path}/{graph_name}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/{graph.name}.srdfg"
    pm.pb_store(graph, full_path)
    node = pm.pb_load(pb_path)


@pytest.mark.parametrize('x_shape, w_shape, params', [
    ((1, 1, 8, 8), (3, 1, 3, 3), {"stride": 1, "pad": 0}),
    ((1, 1, 4, 4), (2, 1, 2, 2), {"stride": 2, "pad": 1}),
])
def test_conv_embedded_values(x_shape, w_shape, params):
    shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3],
                  "nf": w_shape[0], "kh": w_shape[2], "kw": w_shape[3],
                  "stride": params["stride"], "pad": params["pad"]}
    graph, input_info0, out_info, keys = conv(x_shape, w_shape, params, coarse=True, debug_matrix=True)

    ngraph, input_info1, out_info, keys = conv(x_shape, w_shape, params, coarse=False, debug_matrix=True)

    lower_pass = pm.Lower({})
    lowered = lower_pass(ngraph)


    pb_path = f"{OUTPATH}/{graph.name}.srdfg"
    pm.pb_store(lowered, OUTPATH)
    node = pm.pb_load(pb_path)
    assert len(node.nodes) == len(lowered.nodes)
    assert list(node.nodes.keys()) == list(lowered.nodes.keys())


