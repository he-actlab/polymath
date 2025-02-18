from polymath.srdfg.passes.compiler_passes import NormalizeGraph, Lower
import polymath as pm
import pprint
import numpy as np
from pathlib import Path
from tests.util import reco, sigmoid, svm, logistic, linear, set_shape_and_lower, conv
import pytest

CWD = Path(f"{__file__}").parent
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"


def test_single_dim():
    with pm.Node(name="elem5") as graph:
        m = pm.parameter(name="m")
        x = pm.input("x", shape=m)
        w = pm.state("w", shape=m)
        i = pm.index(0, m-1, name="i")
        w[i] = (w[i]*x[i])
    x_ = np.random.randint(0, 10, 3)
    w_ = np.random.randint(0, 10, 3)
    coarse_eval = graph("w", x=x_, w=w_)

    np_result = x_*w_
    np.testing.assert_allclose(coarse_eval, np_result)
    shape_pass = NormalizeGraph({"m": 3})
    graph_shapes = shape_pass(graph)

    shape_res = graph_shapes("w", x=x_, w=w_)
    np.testing.assert_allclose(shape_res, np_result)
    lower_pass = Lower({})
    lowered_graph = lower_pass(graph_shapes)
    input_info = {f"w/w({i},)": w_[i] for i in range(len(w_))}
    input_info.update({f"x/x({i},)": x_[i] for i in range(len(x_))})
    fine_grained_eval = lowered_graph("w/w(1,)", input_info)

    assert fine_grained_eval == np_result[1]


    pb_path = f"{OUTPATH}/{graph.name}.srdfg"
    pm.pb_store(lowered_graph, OUTPATH)
    loaded_node = pm.pb_load(pb_path)
    input_info = {f"w/w({i},)": w_[i] for i in range(len(w_))}
    input_info.update({f"x/x({i},)": x_[i] for i in range(len(x_))})
    fine_grained_eval = loaded_node("w/w(1,)", input_info)
    assert fine_grained_eval == np_result[1]


@pm.register_pass
def get_children(node, ctx):
    for a in node.args:
        if a.name in ctx:
            ctx[a.name]["children"].append(node.name)

@pm.register_pass
def non_class_pass(node, ctx):
    ctx['dtype'] = node.type_modifier
    return node

def test_multi_dim():
    with pm.Node(name="elem4") as graph:
        m = pm.parameter(name="m")
        n = pm.parameter(name="n")
        x = pm.input("x", shape=(m,n))
        w = pm.state("w", shape=(m,n))
        i = pm.index(0, m-1, name="i")
        j = pm.index(0, n-1, name="j")
        w[i,j] = (w[i,j]*x[i,j])
    m_ = 3
    n_ = 4
    x_ = np.random.randint(0, 10, m_*n_).reshape((m_,n_))
    w_ = np.random.randint(0, 10, m_*n_).reshape((m_,n_))
    coarse_eval = graph("w", x=x_, w=w_)
    np_result = x_*w_
    np.testing.assert_allclose(coarse_eval, np_result)
    shape_pass = NormalizeGraph({"m": m_, "n": n_})
    graph_shapes = shape_pass(graph)
    shape_res = graph_shapes("w", x=x_, w=w_)
    np.testing.assert_allclose(shape_res, np_result)
    lower_pass = Lower({})
    lowered_graph = lower_pass(graph_shapes)
    input_info = {}
    for i in range(m_):
        for j in range(n_):
            input_info[f"w/w({i}, {j})"] = w_[i,j]
            input_info[f"x/x({i}, {j})"] = x_[i,j]

    fine_grained_eval = lowered_graph("w/w(2, 3)", input_info)
    assert fine_grained_eval == np_result[2,3]

def test_single_dim_op_slice():
    with pm.Node(name="elem3") as graph:
        m = pm.parameter(name="m")
        x = pm.input("x", shape=m)
        w = pm.state("w", shape=m)
        i = pm.index(0, m-1, name="i")
        out = (w[i]*x[i])
        w[i] = (out[i] - w[i])

    m_ = 3
    x_ = np.random.randint(0, 10, m_)
    w_ = np.random.randint(0, 10, m_)

    coarse_eval = graph("w", x=x_, w=w_)
    np_result = x_*w_ - w_
    np.testing.assert_allclose(coarse_eval, np_result)

    shape_pass = NormalizeGraph({"m": 3})
    graph_shapes = shape_pass(graph)
    shape_res = graph_shapes("w", x=x_, w=w_)

    np.testing.assert_allclose(shape_res, np_result)
    lower_pass = Lower({})
    lowered_graph = lower_pass(graph_shapes)
    input_info = {f"w/w({i},)": w_[i] for i in range(len(w_))}
    input_info.update({f"x/x({i},)": x_[i] for i in range(len(x_))})
    fine_grained_eval = lowered_graph("w/w(2,)", input_info)
    assert fine_grained_eval == np_result[2]

def test_multi_dim_op_slice():
    with pm.Node(name="elem2") as graph:
        m = pm.parameter(name="m")
        n = pm.parameter(name="n")
        mu = pm.parameter(name="mu", default=2.0)
        x = pm.input(name="x", shape=(m,n))
        w = pm.state(name="w", shape=(m,n))
        i = pm.index(0, m-1, name="i")
        j = pm.index(0, n-1, name="j")
        out = (x[i,j]*w[i,j]).set_name("w_out")
        w[i,j] = (mu * (out[i,j] - w[i,j]) )
    m_ = 3
    n_ = 2
    x_ = np.random.randint(0, 10, m_*n_).reshape((m_, n_))
    w_ = np.random.randint(0, 10, m_*n_).reshape((m_, n_))
    coarse_eval = graph("w", x=x_, w=w_)
    np_result = (x_*w_ - w_)*2.0
    np.testing.assert_allclose(coarse_eval, np_result)
    shape_pass = NormalizeGraph({"m": m_, "n": n_})
    graph_shapes = shape_pass(graph)
    shape_res = graph_shapes("w", x=x_, w=w_)
    np.testing.assert_allclose(shape_res, np_result)
    lower_pass = Lower({})
    lowered_graph = lower_pass(graph_shapes)
    input_info = {}
    for i in range(m_):
        for j in range(n_):
            input_info[f"w/w({i}, {j})"] = w_[i,j]
            input_info[f"x/x({i}, {j})"] = x_[i,j]
    fine_grained_eval = lowered_graph("w/w(2, 1)", input_info)
    assert fine_grained_eval == np_result[2, 1]

def test_lower_group_op():
    with pm.Node(name="linear_reg1") as graph:
        m = pm.parameter(name="m")
        x = pm.input("x", shape=(m))
        y = pm.input("y")
        w = pm.state("w", shape=(m))
        i = pm.index(0, m-1, name="i")
        h = pm.sum([i], w[i] * x[i], name="h")
    m_ = 3
    n_ = 3
    x_ = np.random.randint(0, 10, m_)
    w_ = np.random.randint(0, 10, (m_))
    np_result = np.sum(x_ * w_)
    np.testing.assert_allclose(graph("h", {"w": w_, "x": x_}), np_result)
    np.testing.assert_allclose(graph("h", w=w_, x=x_), np_result)
    shape_pass = NormalizeGraph({"m": m_, "n": n_})
    graph_shapes = shape_pass(graph)
    shape_res = graph_shapes("h", x=x_, w=w_)
    np.testing.assert_allclose(shape_res, np_result)

    lower_pass = Lower({})
    lowered_graph = lower_pass(graph_shapes)
    input_info = {f"w/w({i},)": w_[i] for i in range(len(w_))}
    input_info.update({f"x/x({i},)": x_[i] for i in range(len(x_))})
    #
    fine_grained_eval = lowered_graph("h/h(4,)", input_info)
    assert fine_grained_eval == np_result

    pb_path = f"{OUTPATH}/linear_reg1.srdfg"

    pm.pb_store(lowered_graph, OUTPATH)
    loaded_node = pm.pb_load(pb_path)    #
    input_info = {f"w/w({i},)": w_[i] for i in range(len(w_))}
    input_info.update({f"x/x({i},)": x_[i] for i in range(len(x_))})

    loaded_res = loaded_node("h/h(4,)", input_info)

    assert loaded_node.func_hash() == lowered_graph.func_hash()
    assert loaded_res == np_result
#
def test_single_dim_norm():
    with pm.Node(name="elem1") as graph:
        m = pm.parameter("m")
        x = pm.input("x", shape=m)
        w = pm.state("w", shape=m)
        i = pm.index(0, m-1, name="i")
        w[i] = (w[i]*x[i])
    x_ = np.random.randint(0, 10, 3)
    w_ = np.random.randint(0, 10, 3)
    coarse_eval = graph("w", x=x_, w=w_)

    np_result = x_*w_
    np.testing.assert_allclose(coarse_eval, np_result)
    shape_pass = NormalizeGraph({"m": 3})
    graph_shapes = shape_pass(graph)

    shape_res = graph_shapes("w", x=x_, w=w_)
    np.testing.assert_allclose(shape_res, np_result)
    lower_pass = Lower({})
    lowered_graph = lower_pass(graph_shapes)
    input_info = {f"w/w({i},)": w_[i] for i in range(len(w_))}
    input_info.update({f"x/x({i},)": x_[i] for i in range(len(x_))})
    fine_grained_eval = lowered_graph("w/w(1,)", input_info)

    assert fine_grained_eval == np_result[1]

    pb_path = f"{OUTPATH}/{graph.name}.srdfg"
    pm.pb_store(lowered_graph, OUTPATH)
    loaded_node = pm.pb_load(pb_path)
    input_info = {f"w/w({i},)": w_[i] for i in range(len(w_))}
    input_info.update({f"x/x({i},)": x_[i] for i in range(len(x_))})
    fine_grained_eval = loaded_node("w/w(1,)", input_info)
    assert fine_grained_eval == np_result[1]

def test_multi_dim_norm():
    with pm.Node(name="elem") as graph:
        m = pm.parameter(name="m")
        n = pm.parameter(name="n")
        x = pm.input("x", shape=(m,n))
        w = pm.state("w", shape=(m,n))
        i = pm.index(0, m-1, name="i")
        j = pm.index(0, n-1, name="j")
        w[i,j] = (w[i,j]*x[i,j])
    m_ = 3
    n_ = 4
    x_ = np.random.randint(0, 10, m_*n_).reshape((m_,n_))
    w_ = np.random.randint(0, 10, m_*n_).reshape((m_,n_))
    coarse_eval = graph("w", x=x_, w=w_)
    np_result = x_*w_
    np.testing.assert_allclose(coarse_eval, np_result)
    shape_pass = NormalizeGraph({"m": m_, "n": n_})
    graph_shapes = shape_pass(graph)
    shape_res = graph_shapes("w", x=x_, w=w_)
    np.testing.assert_allclose(shape_res, np_result)
    lower_pass = pm.Lower({})
    lowered_graph = lower_pass(graph_shapes, {})
    input_info = {}
    for i in range(m_):
        for j in range(n_):
            input_info[f"w/w({i}, {j})"] = w_[i,j]
            input_info[f"x/x({i}, {j})"] = x_[i,j]
    fine_grained_eval = lowered_graph("w/w(2, 3)", input_info)
    assert fine_grained_eval == np_result[2,3]

def test_reco():
    m_ = 3
    n_ = 3
    k_ = 2
    graph, input_info, out_info, keys = reco(m=m_, n=n_, k=k_, coarse=True)
    shape_val_pass = pm.NormalizeGraph({"m": m_, "n": n_, "k": k_})
    new_graph = shape_val_pass(graph)

    test_res = new_graph(keys, input_info)
    np.testing.assert_allclose(test_res[0], out_info["w1"])
    np.testing.assert_allclose(test_res[1], out_info["w2"])

    graph, input_info, new_out_info, keys = reco(m=m_, n=n_, k=k_)
    flatten_pass = pm.Lower({})
    flattened_g = flatten_pass(new_graph)

    all_vals = flattened_g(keys, input_info)
    out1 = np.asarray(list(all_vals[0:6])).reshape(new_out_info["w2"].shape)
    out2 = np.asarray(list(all_vals[6:])).reshape(new_out_info["w2"].shape)
    np.testing.assert_allclose(new_out_info["w1"], out1)
    np.testing.assert_allclose(new_out_info["w2"], out2)

@pytest.mark.parametrize('m_',[
    3, 54
])
def test_svm(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = svm(**shape_dict, coarse=True)
    shape_val_pass = pm.NormalizeGraph(shape_dict)
    new_graph = shape_val_pass(graph)
    test_res = new_graph(keys, input_info)
    np.testing.assert_allclose(test_res, out_info["w"])

    graph, input_info, new_out_info, keys = svm(**shape_dict)
    flatten_pass = pm.Lower({})
    flattened_g = flatten_pass(new_graph)

    all_vals = flattened_g(keys, input_info)
    np.testing.assert_allclose(new_out_info["w"], all_vals)


@pytest.mark.parametrize('m_',[
    10
])
def test_linear(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = linear(**shape_dict, coarse=True)
    shape_val_pass = pm.NormalizeGraph(shape_dict)
    new_graph = shape_val_pass(graph)
    test_res = new_graph(keys, input_info)
    np.testing.assert_allclose(test_res, out_info["w"])
    graph, input_info, new_out_info, keys = linear(**shape_dict)
    flatten_pass = pm.Lower({})
    flattened_g = flatten_pass(new_graph)
    all_vals = flattened_g(keys, input_info)
    np.testing.assert_allclose(new_out_info["w"], all_vals)

@pytest.mark.parametrize('m_',[
    3
])
def test_sigmoid(m_):

    with pm.Node(name="logistic1") as graph:
        m = pm.parameter(name="m")
        n = pm.parameter(name="n")
        x = pm.input("x", shape=(m))
        w = pm.state("w", shape=(m))
        i = pm.index(0, m-1, name="i")
        o = pm.sigmoid(pm.sum([i], w[i]*x[i]), name="out")
    x_ = np.random.randint(0, 10, m_)
    w_ = np.random.randint(0, 10, m_)
    input_dict = {"x": x_, "w": w_}
    np_res = int(sigmoid(np.sum(x_*w_)))
    shape_dict = {"m": m_}

    coarse_eval = graph("out", x=x_, w=w_)
    np.testing.assert_allclose(np_res, coarse_eval)
    lowered = set_shape_and_lower(graph, shape_dict)


@pytest.mark.parametrize('m_',[
    3
])
def test_multidim_sigmoid(m_):

    with pm.Node(name="logistic") as graph:
        m = pm.parameter(name="m")
        n = pm.parameter(name="n")
        x = pm.input("x", shape=(m))
        w = pm.state("w", shape=(m))
        i = pm.index(0, m-1, name="i")
        o = pm.sigmoid(w[i]*x[i], name="out")
    x_ = np.random.randint(0, 10, m_).astype(np.float)
    w_ = np.random.randint(0, 10, m_).astype(np.float)
    shape_dict = {"m": m_}
    input_dict = {"x": x_, "w": w_}
    np_res = sigmoid((x_*w_))

    coarse_eval = graph("out", input_dict)
    np.testing.assert_allclose(np_res, coarse_eval)
    lowered = set_shape_and_lower(graph, shape_dict)
    keys = [f"out/out({i},)" for i in range(m_)]

    x_ = np.random.randint(0, 10, m_).astype(np.float)
    w_ = np.random.randint(0, 10, m_).astype(np.float)
    input_dict = {}
    for i in range(m_):
        input_dict[f"x/x({i},)"] = x_[i]
        input_dict[f"w/w({i},)"] = w_[i]
    np_res = sigmoid((x_*w_))

    lower_res = np.asarray(lowered(keys, input_dict)).reshape(np_res.shape)
    np.testing.assert_allclose(lower_res, np_res)



