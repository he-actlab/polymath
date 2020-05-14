from pathlib import Path
import polymath as pm
import numpy as np
import pytest
from .util import logistic, linear, reco, svm, compare_tabla_dfg, set_shape_and_lower,\
    unwound_fft, backprop, conv, lenet
import pickle

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

    graph_name = "linear_reg"
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
        mug = (mu*g[i]).set_name("mu*g[i]")
        w_ = ((w_[i])- mug).set_name("w_out")
    x = np.random.randint(0, 10, 10)
    y = np.random.randint(0, 10, 1)[0]
    w = np.random.randint(0, 10, 10)

    graph_res = graph("w_out", {"x": x, "y": y, "w": w})
    actual_res = w - ((np.sum(x*w) - y)*x)*1.0

    np.testing.assert_allclose(graph_res, actual_res)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/{graph_name}.pb"
    pm.pb_store(graph, full_path)
    node = pm.pb_load(pb_path)
    new_graph_res = node("w_out", {"x": x, "y": y, "w": w})
    np.testing.assert_allclose(graph_res, new_graph_res)
    np.testing.assert_allclose(actual_res, new_graph_res)

    assert (node.func_hash()) == (graph.func_hash())

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
    pb_path = f"{full_path}/{graph.name}.pb"
    pm.pb_store(graph, full_path)
    node = pm.pb_load(pb_path)


@pytest.mark.parametrize('x_shape, w_shape, params', [
    ((1, 1, 8, 8), (3, 1, 3, 3), {"stride": 1, "pad": 0}),
    # ((1, 1, 4, 4), (2, 1, 2, 2), {"stride": 2, "pad": 1}),
])
def test_conv_embedded_values(x_shape, w_shape, params):
    shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3],
                  "nf": w_shape[0], "kh": w_shape[2], "kw": w_shape[3],
                  "stride": params["stride"], "pad": params["pad"]}
    graph, input_info0, out_info, keys = conv(x_shape, w_shape, params, coarse=True, debug_matrix=True)

    ngraph, input_info1, out_info, keys = conv(x_shape, w_shape, params, coarse=False, debug_matrix=True)

    lower_pass = pm.Lower({})
    lowered = lower_pass(ngraph)

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"

    pb_path = f"{full_path}/{graph.name}.pb"
    pm.pb_store(lowered, full_path)
    node = pm.pb_load(pb_path)
    assert len(node.nodes) == len(lowered.nodes)
    assert list(node.nodes.keys()) == list(lowered.nodes.keys())


def test_lenet_embedded():
    shape_dict = {"n": 1, "ic": 1, "ih": 32, "iw": 32,
                  "nf1": 6, "kh1": 5, "kw1": 5, "s1": 1, "p1": 0,
                  "nf2": 16, "kh2": 5, "kw2": 5, "s2": 1, "p2": 0,
                  "m6": 400, "n6": 120, "m7": 120, "n7": 84, "m8": 84, "n8": 10
                  }
    graph, inp_info, out_info, keys = lenet(coarse=False, debug=True)
    print(f"Finished applying shape pass")
    lower_pass = pm.Lower({}, debug=True)
    graph = lower_pass(graph)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/lenet.pb"
    pm.pb_store(graph, full_path)
    node = pm.pb_load(pb_path)
    assert len(node.nodes) == len(graph.nodes)
    assert list(node.nodes.keys()) == list(graph.nodes.keys())


@pytest.mark.parametrize('x_shape, w_shape, params', [
    ((1, 1, 32, 32), (6, 1, 3, 3), {"stride": 1, "pad": 0}),
])
def test_mini_lenet(x_shape, w_shape, params):
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
        c1 = pm.output(name="c1")
        a1 = pm.output(name="a1")
        l1 = pm.output(name="l1")

        pm.conv(data, w1, b1, c1, s1, p1)

    shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3],
                  "nf1": w_shape[0], "kh1": w_shape[2], "kw1": w_shape[3],
                  "s1": params["stride"], "p1": params["pad"]}
    shape_val_pass = pm.NormalizeGraph(shape_dict, debug=True)
    graph = shape_val_pass(graph)
    lower_pass = pm.Lower({}, debug=True)
    graph = lower_pass(graph)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/{graph.name}.pb"
    pm.pb_store(graph, full_path)
    node = pm.pb_load(pb_path)
    assert len(node.nodes) == len(graph.nodes)
    assert list(node.nodes.keys()) == list(graph.nodes.keys())


