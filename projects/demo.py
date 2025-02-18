import numpy as np
import polymath as pm
from pathlib import Path
import tvm
import pickle
from tests.util import np_lenet
BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")
CWD = Path(f"{__file__}").parent
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"
ONNX_FILE_DIR = Path(f"{Path(__file__).parent}/onnx_examples")

def create_graph():

    with pm.Node(name="linear_train") as graph:
        m = pm.parameter("m")
        mu = pm.parameter(name="mu", default=1.0)

        x = pm.input("x", shape=(m))
        y = pm.input("y")
        w = pm.state("w", shape=(m))
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("g")
        w[i] = (w[i] - (mu * g[i]).set_name("mu*g")).set_name("w_update")

    return graph

def run_graph(graph):
    x_inp = np.random.randint(-3, 3, 10)
    w_inp = np.random.randint(-3, 3, 10)
    y_inp = np.random.randint(-3, 3, 1)[0]
    print(f"Weights before: {w_inp}")
    result = graph("w", {"x": x_inp, "w": w_inp, "y": y_inp})
    print(f"Weights after: {result}")

def show_graph_nodes(graph, indents):
    for name, node in graph.nodes.items():
        if isinstance(node, pm.var_index):
            continue
        print(f"{indents}{name} - {node.op_name}")
        show_graph_nodes(node, indents + "\t")

def store_graph(graph):
    filepath = Path(f"{__file__}").parent
    pm.pb_store(graph, filepath)
    loaded_graph = pm.pb_load(f"{filepath}/linear.pb")

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
        c1 = pm.output(name="c1")
        a1 = pm.output(name="a1")
        l1 = pm.output(name="l1")

        pm.conv_bias(data, w1, b1, c1, s1, p1)
        pm.relu(c1, a1)
        pm.avg_pool2d(a1, l1, 2, 2, 2, 0)

        nf2 = pm.parameter(name="nf2")
        kh2 = pm.parameter(name="kh2")
        kw2 = pm.parameter(name="kw2")
        w2 = pm.state(name="w2", shape=(nf2, nf1, kh2, kw2))
        b2 = pm.state(name="b2", shape=(nf2))
        s2 = pm.parameter(name="s2")
        p2 = pm.parameter(name="p2")
        c2 = pm.output(name="c2")
        a2 = pm.output(name="a2")
        l2 = pm.output(name="l2")

        pm.conv_bias(l1, w2, b2, c2, s2, p2)
        pm.relu(c2, a2)
        pm.avg_pool2d(a2, l2, 2, 2, 2, 0)

        f5 = pm.output(name="f5")
        pm.batch_flatten(l2, f5)

        f6 = pm.output(name="f6")
        m6 = pm.parameter(name="m6")
        n6 = pm.parameter(name="n6")
        w6 = pm.state(name="w6", shape=(n6, m6))
        a6 = pm.output(name="a6")
        pm.dense(f5, w6, f6)
        pm.relu1d(f6, a6)

        f7 = pm.output(name="f7")
        m7 = pm.parameter(name="m7")
        n7 = pm.parameter(name="n7")
        w7 = pm.state(name="w7", shape=(n7, m7))
        a7 = pm.output(name="a7")
        pm.dense(a6, w7, f7)
        pm.relu1d(f7, a7)

        f8 = pm.output(name="f8")
        m8 = pm.parameter(name="m8")
        n8 = pm.parameter(name="n8")
        w8 = pm.state(name="w8", shape=(n8, m8))
        a8 = pm.output(name="a8")
        pm.dense(a7, w8, f8)
        pm.relu1d(f8, a8)

        out = pm.output(name="sm")
        pm.softmax(a8, out)
    if coarse:
        in_info, keys, out_info = np_lenet()
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": 1, "ic": 1, "ih": 32, "iw": 32,
                      "nf1": 6, "kh1": 5, "kw1": 5, "s1": 1, "p1": 0,
                      "nf2": 16, "kh2": 5, "kw2": 5, "s2": 1, "p2": 0,
                      "m6": 400, "n6": 120, "m7": 120, "n7": 84, "m8": 84, "n8": 10
                      }
        shape_val_pass = pm.NormalizeGraph(shape_dict, debug=debug)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = np_lenet(lowered=True)
        return new_graph, in_info, out_info, keys

def tvm_lenet(num_classes=10, data_shape=(1, 1, 32, 32),
               dtype='float32', alpha=1.0, is_shallow=False):
    from tvm import relay
    from tvm.relay.testing import layers

    """Function to construct arg Lenet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    conv1 = layers.conv2d(data=data,channels=6, kernel_size=(5,5), name='conv1')
    conv1 = relay.nn.relu(conv1)
    pool2 = relay.nn.avg_pool2d(conv1, pool_size=(2,2), strides=(2,2))
    conv3 = layers.conv2d(data=pool2, channels=16, kernel_size=(5,5), name='conv3')
    conv3 = relay.nn.relu(conv3)
    pool4 = relay.nn.avg_pool2d(conv3, pool_size=(2,2), strides=(2,2))
    flattened5 = relay.nn.batch_flatten(pool4)

    fcw5 = relay.var('fc5_weight')
    fc5 = relay.nn.dense(data=flattened5, weight=fcw5, units=120)
    fc5 = relay.nn.relu(fc5)

    fcw6 = relay.var('fc6_weight')
    fc6 = relay.nn.dense(data=fc5, weight=fcw6, units=84)
    fc6 = relay.nn.relu(fc6)

    fcw7 = relay.var('fc7_weight')
    fc7= relay.nn.dense(data=fc6, weight=fcw7, units=num_classes)
    fc7 = relay.nn.relu(fc7)

    softmax = relay.nn.softmax(data=fc7)
    fn = relay.Function(relay.analysis.free_vars(softmax), softmax)
    return fn

def compare_lenets():
    graph, inp_info, out_info, key = lenet(coarse=True)
    coarse_cpy = pickle.loads(pickle.dumps(inp_info))

    tvm_code = pm.generate_tvm(graph, inp_info, "")
    pm_mod = tvm.IRModule.from_expr(tvm_code)
    pm_mod = tvm.relay.transform.InferType()(pm_mod)

    net = tvm_lenet()
    mod = tvm.IRModule.from_expr(net)
    mod = tvm.relay.transform.InferType()(mod)
    # res = graph(key, coarse_cpy)
    # np.testing.assert_allclose(res, out_info[key])
    return pm_mod, mod, graph

def lenet_from_onnx():
    filename = f"mnist-lenet.onnx"
    filepath = f"{BENCH_DIR}/full_dnns/{filename}"
    assert Path(filepath).exists()
    graph = pm.from_onnx(filepath)
    return graph

if __name__ == "__main__":
    # Load onnx
    # from_onnx_mgdfg = lenet_from_onnx()
    # show_graph_nodes(from_onnx_mgdfg, "\t")

    pm_mod, mod, lenet_mgdfg = compare_lenets()
    print(f"Graph: {lenet_mgdfg.name}")
    show_graph_nodes(lenet_mgdfg, "\t")
    # #
    print(f"------------------------TVM Lenet----------------------------------")
    print(mod)
    print(f"------------------------PolyMath->TVM Lenet------------------------")
    print(pm_mod)







