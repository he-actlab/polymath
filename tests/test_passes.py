from polymath.srdfg.passes import register_pass, Pass, pass_registry
from polymath.srdfg.passes.compiler_passes import Lower, NormalizeGraph, CountNodes
import polymath as pm
import numpy as np
from itertools import product

@register_pass
class RemoveNode(Pass):
    def __init__(self):
        init_info = {"count": 0}
        super(RemoveNode, self).__init__(init_info)

    def apply_pass(self, node, count=None):
        if "t" in node.nodes:
            node.nodes.pop("t")
        return node


def linear_reg_graph_mg():
    graph_name = "linear_reg"
    with pm.Node(name="linear_reg") as graph:
        m = pm.placeholder("m")
        x_ = pm.placeholder("x", shape=(m), type_modifier="input")
        y_ = pm.placeholder("y", type_modifier="input")
        w_ = pm.placeholder("w", shape=(m), type_modifier="state")
        i = pm.index(0, m-1, name="i")
        h = pm.sum([i], (x_[i] * w_[i]).set_name("x*w"), name="h")
        d = (h-y_).set_name("h-y")
        g = (d*x_[i]).set_name("d*x")
        with pm.Node(name="grad_update") as graph2:
            mu = pm.parameter(name="mu", default=1.0)
            p1 = mu*g[i]
            p2 = w_[i]
            w_prime = (p2 - p1).set_name("res1")
        tout = (w_prime * 1.0).set_name("res")
    return graph

def linear_reg_graph():
    graph_name = "linear_reg"
    with pm.Node(name="linear_reg") as graph:
        m = pm.placeholder("m")
        mu = pm.parameter(name="mu", default=1.0)
        x_ = pm.placeholder("x", shape=(m), type_modifier="input")
        y_ = pm.placeholder("y", type_modifier="input")
        w_ = pm.placeholder("w", shape=(m), type_modifier="state")
        i = pm.index(0, m-1, name="i")
        h = pm.sum([i], (x_[i] * w_[i]).set_name("x*w"), name="h")
        d = (h-y_).set_name("h-y")
        g = (d*x_[i]).set_name("d*x")
        w_out = (w_[i])- mu*g[i]
        w_out.set_name("res")
    return graph


def test_visit():

    graph = linear_reg_graph()
    x = np.random.randint(0, 255, 10)
    y = np.random.randint(0, 255, 1)[0]
    w = np.random.randint(0, 255, 10)
    actual_res = w - ((np.sum(x*w) - y)*x)*1.0
    graph_res = graph("res", {"x": x, "y": y, "w": w})

    test_pass = CountNodes()
    new_graph = test_pass(graph)
    orig_ops = [(n.name, n.op_name) for _, n in graph.nodes.items()]
    new_ops = [(n.name, n.op_name) for _, n in new_graph.nodes.items()]
    assert graph.func_hash() == new_graph.func_hash()

    assert test_pass.ctx["count"] == 17
    assert test_pass.ctx["global"] == 1
    assert test_pass.ctx["linear_reg"] == 16
    visit_res = new_graph("res", {"x": x, "y": y, "w": w})
    np.testing.assert_allclose(graph_res, actual_res)
    np.testing.assert_allclose(visit_res, actual_res)

def test_transform():
    graph = linear_reg_graph()

    x = np.random.randint(0, 255, 10)
    y = np.random.randint(0, 255, 1)[0]
    w = np.random.randint(0, 255, 10)
    actual_res = w - ((np.sum(x * w) - y) * x) * 1.0
    graph_res = graph("res", {"x": x, "y": y, "w": w})
    test_pass = RemoveNode()
    test_count = CountNodes()
    new_graph = test_pass(graph)
    orig_graph = test_count(new_graph)
    assert test_count.ctx["count"] == 17
    assert test_count.ctx["global"] == 1
    assert test_count.ctx["linear_reg"] == 16
    visit_res = new_graph("res", {"x": x, "y": y, "w": w})
    np.testing.assert_allclose(graph_res, actual_res)
    np.testing.assert_allclose(visit_res, actual_res)

def test_fn_transform():
    graph = linear_reg_graph()

    x = np.random.randint(0, 255, 10)
    y = np.random.randint(0, 255, 1)[0]
    w = np.random.randint(0, 255, 10)
    actual_res = w - ((np.sum(x * w) - y) * x) * 1.0
    graph_res = graph("res", {"x": x, "y": y, "w": w})
    test_pass = RemoveNode()
    test_count = CountNodes()
    new_graph = test_pass(graph)

    orig_graph = test_count(new_graph)
    visit_res = new_graph("res", {"x": x, "y": y, "w": w})
    np.testing.assert_allclose(graph_res, actual_res)
    np.testing.assert_allclose(visit_res, actual_res)

def test_shape_eval():

    graph = linear_reg_graph_mg()

    shape_val_pass = NormalizeGraph({"m": 3})
    flatten_pass = Lower({})
    new_graph = shape_val_pass(graph)
    count_pass = CountNodes()
    orig_graph = count_pass(new_graph)
    assert new_graph["x"].shape == (3,)


def test_flatten_result_length():
    with pm.Node(name="linear_reg") as graph:
        m = pm.placeholder("m", type_modifier="param")
        x = pm.placeholder("x", shape=(m), type_modifier="input")
        y = pm.placeholder("y", type_modifier="input")
        w = pm.placeholder("w", shape=(m), type_modifier="state")
        mu = pm.placeholder("mu", default_val=1.0, type_modifier="param")
        i = pm.index(0, (m - 1).set_name("m-1")).set_name("i")
        h = pm.sum([i], (x[i] * w[i]).set_name("x*w"), name="h")
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w_ = (w[i] - (mu * g[i]).set_name("mu*g")).set_name(("w_out"))

    shape_val_pass = NormalizeGraph({"m": 3})
    count_pass = CountNodes()
    flatten_pass = Lower({})

    new_graph = shape_val_pass(graph)

    flattened_g = flatten_pass(new_graph)
    x = np.random.randint(0, 10, 10)
    y = np.random.randint(0, 10, 1)[0]
    w = np.random.randint(0, 10, 10)

    orig_graph = count_pass(flattened_g)

def numpy_reco(input_dict):

    out_info = {}
    w1_x2 = np.zeros(input_dict["w1"].shape)
    w2_x1 = np.zeros(input_dict["w2"].shape)
    h1 = np.zeros(shape=(input_dict["m"]))
    h1_sum = np.zeros(shape=(input_dict["m"]))
    h2 = np.zeros(shape=(input_dict["n"]))
    h2_sum = np.zeros(shape=(input_dict["n"]))

    for i in range(input_dict["m"]):
        for l in range(input_dict["k"]):
            w1_x2[i][l] = input_dict["w1"][i][l] * input_dict["x2"][l]
            h1_sum[i] += w1_x2[i][l]
        h1[i] = h1_sum[i] * input_dict["r1"][i]
    out_info["h1"] = h1
    out_info["h1_sum"] = h1_sum
    out_info["w1_x2"] = w1_x2

    for j in range(input_dict["n"]):
        for l in range(input_dict["k"]):
            w2_x1[j][l] =input_dict["w2"][j][l] * input_dict["x1"][l]
            h2_sum[j] += w2_x1[j][l]
        h2[j] = h2_sum[j] * input_dict["r2"][j]
    out_info["h2"] = h2
    out_info["h2_sum"] = h2_sum
    out_info["w2_x1"] = w2_x1
    d1 = h1 - input_dict["y1"]
    out_info["d1"] = d1
    d2 = h2 - input_dict["y2"]
    out_info["d2"] = d2
    g1 = np.zeros(shape=(input_dict["m"], input_dict["k"]))
    g2 = np.zeros(shape=(input_dict["n"], input_dict["k"]))

    for i in range(input_dict["m"]):
        for l in range(input_dict["k"]):
            g1[i][l] = d1[i] * input_dict["x2"][l]
    out_info["g1"] = g1
    for j in range(input_dict["n"]):
        for l in range(input_dict["k"]):
            g2[j][l] = d2[j] * input_dict["x1"][l]
    out_info["g2"] = g2
    w1_out = input_dict["w1"] - g1
    w2_out = input_dict["w2"] - g2
    out_info["w1"] = w1_out
    out_info["w2"] = w2_out
    return out_info

def test_flatten_reco():
    with pm.Node(name="recommender") as graph:
        m = pm.parameter("m")
        n = pm.parameter("n")
        k = pm.parameter("k")
        x1 = pm.input("x1", shape=(k,))
        x2 = pm.input("x2", shape=(k,))

        r1 = pm.input("r1", shape=(m,))
        y1 = pm.input("y1", shape=(m,))

        r2 = pm.input("r2", shape=(n,))
        y2 = pm.input("y2", shape=(n,))

        w1 = pm.state("w1", shape=(m, k))
        w2 = pm.state("w2", shape=(n, k))
        i = pm.index(0, m-1, name="i")
        j = pm.index(0, n-1, name="j")
        l = pm.index(0, k-1, name="l")
        h1_sum = pm.sum([l], (w1[i, l] * x2[l]).set_name("w1*x2")).set_name("h1_sum")
        h1 = (h1_sum[i] * r1[i]).set_name("h1")
        h2_sum = pm.sum([l], (w2[j, l]* x1[l]).set_name("w2*x1")).set_name("h2_sum")
        h2 = (h2_sum[j] * r2[j]).set_name("h2")

        d1 = (h1[i] - y1[i]).set_name("d1")
        d2 = (h2[j] - y2[j]).set_name("d2")
        g1 = (d1[i] * x2[l]).set_name("g1")
        g2 = (d2[j] * x1[l]).set_name("g2")
        w1[i,l] = (w1[i,l] - g1[i,l])
        w2[j,l] = (w2[j,l] - g2[j,l])
    m_ = 3
    n_ = 3
    k_ = 2
    input_info = {}
    input_info["m"] = m_
    input_info["n"] = n_
    input_info["k"] = k_
    input_info["w1"] = np.random.randint(1, 6, m_*k_).reshape(m_, k_)
    input_info["w2"] = np.random.randint(1, 6, n_*k_).reshape(n_, k_)
    input_info["x1"] = np.random.randint(1, 6, k_)
    input_info["x2"] = np.random.randint(1, 6, k_)

    input_info["r1"] = np.random.randint(0, 2, m_)
    input_info["y1"] = np.random.randint(0, 6, m_)
    input_info["r2"] = np.random.randint(0, 2, n_)
    input_info["y2"] = np.random.randint(0, 6, n_)
    out_info = numpy_reco(input_info)
    shape_val_pass = NormalizeGraph({"m": m_, "n": n_, "k": k_})
    flatten_pass = Lower({})

    new_graph = shape_val_pass(graph)
    test_res = new_graph(["w1", "w2"], input_info)
    np.testing.assert_allclose(test_res[0], out_info["w1"])
    np.testing.assert_allclose(test_res[1], out_info["w2"])
    flattened_g = flatten_pass(new_graph)
    input_info = {}
    input_info["m"] = m_
    input_info["n"] = n_
    input_info["k"] = k_
    input_info["w1"] = np.random.randint(1, 6, m_*k_).reshape(m_, k_)
    input_info["w2"] = np.random.randint(1, 6, n_*k_).reshape(n_, k_)
    input_info["x1"] = np.random.randint(1, 6, k_)
    input_info["x2"] = np.random.randint(1, 6, k_)

    input_info["r1"] = np.random.randint(0, 2, m_)
    input_info["y1"] = np.random.randint(0, 6, m_)
    input_info["r2"] = np.random.randint(0, 2, n_)
    input_info["y2"] = np.random.randint(0, 6, n_)
    new_out_info = numpy_reco(input_info)

    pairs_w1 = list(product(*tuple([np.arange(i) for i in input_info["w1"].shape])))
    pairs_w2 = list(product(*tuple([np.arange(i) for i in input_info["w2"].shape])))
    w1_init = input_info["w1"]
    for p in pairs_w1:
        input_info[f"w1/w1({p[0]}, {p[1]})"] = input_info["w1"][p]
    input_info.pop("w1")
    w2_init = input_info["w2"]

    for p in pairs_w2:
        input_info[f"w2/w2({p[0]}, {p[1]})"] = input_info["w2"][p]
    input_info.pop("w2")

    for p in range(k_):
        input_info[f"x1/x1({p},)"] = input_info["x1"][p]
        input_info[f"x2/x2({p},)"] = input_info["x2"][p]
    input_info.pop("x1")
    input_info.pop("x2")

    for p in range(m_):
        input_info[f"r1/r1({p},)"] = input_info["r1"][p]
        input_info[f"y1/y1({p},)"] = input_info["y1"][p]
    input_info.pop("r1")
    input_info.pop("y1")

    for p in range(n_):
        input_info[f"r2/r2({p},)"] = input_info["r2"][p]
        input_info[f"y2/y2({p},)"] = input_info["y2"][p]
    input_info.pop("r2")
    input_info.pop("y2")

    w1_keys = [f"w1/w1({p[0]}, {p[1]})" for p in pairs_w1]
    w2_keys = [f"w2/w2({p[0]}, {p[1]})" for p in pairs_w2]

    all_vals = flattened_g(w1_keys + w2_keys, input_info)
    out1 = np.asarray(list(all_vals[0:6])).reshape(new_out_info["w2"].shape)
    out2 = np.asarray(list(all_vals[6:])).reshape(new_out_info["w2"].shape)
    np.testing.assert_allclose(new_out_info["w1"], np.asarray(list(all_vals[0:6])).reshape(new_out_info["w2"].shape))
    np.testing.assert_allclose(new_out_info["w2"], np.asarray(list(all_vals[6:])).reshape(new_out_info["w2"].shape))
