import polymath as pm
from pathlib import Path
BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")
CWD = Path(f"{__file__}").parent
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"
ONNX_FILE_DIR = Path(f"{Path(__file__).parent}/onnx_examples")


def reco(m_=3, n_=3, k_=2):
    with pm.Node(name="recommender") as graph:
        mu = pm.placeholder("mu")
        m = pm.placeholder("m")
        n = pm.placeholder("n")
        k = pm.placeholder("k")
        x1 = pm.placeholder("x1", shape=k)
        x2 = pm.placeholder("x2", shape=k)

        r1 = pm.placeholder("r1", shape=m)
        y1 = pm.placeholder("y1", shape=m)

        r2 = pm.placeholder("r2", shape=n)
        y2 = pm.placeholder("y2", shape=n)

        w1 = pm.placeholder("w1", shape=(m, k))
        w2 = pm.placeholder("w2", shape=(n, k))
        i = pm.index(0, m - 1, name="i")
        j = pm.index(0, n - 1, name="j")
        l = pm.index(0, k - 1, name="l")
        h1_sum = pm.sum([l], (w1[i, l] * x2[l]).set_name("w1*x2")).set_name("h1_sum")
        h1 = (h1_sum[i] * r1[i]).set_name("h1")
        h2_sum = pm.sum([l], (x1[l] * w2[j, l]).set_name("x1*w2")).set_name("h2_sum")
        h2 = (h2_sum[j] * r2[j]).set_name("h2")
        #
        d1 = (h1[i] - y1[i]).set_name("d1")
        d2 = (h2[j] - y2[j]).set_name("d2")
        g1 = (d1[i] * x2[l]).set_name("g1")
        g2 = (d2[j] * x1[l]).set_name("g2")
        w1_ = (w1[i, l] - g1[i, l]).set_name("w1_")
        w2_ = (w2[i, l] - g2[i, l]).set_name("w2_")

    shape_val_pass = pm.NormalizeGraph({"m": m_, "n": n_, "k": k_})
    new_graph, res = shape_val_pass(graph)
    return new_graph

def linear_reg():
    with pm.Node(name="linear_reg") as graph:
        m = pm.placeholder("m")
        x = pm.placeholder("x", shape=(m), type_modifier="input")
        y = pm.placeholder("y", type_modifier="input")
        w = pm.placeholder("w", shape=(m), type_modifier="state")
        mu = pm.parameter(name="mu", default=1.0)
        i = pm.index(0, (graph["m"]-1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]).set_name("x*w"), name="h")
        d = (h-y).set_name("h-y")
        g = (d*x).set_name("d*x")
        w_ = (w - (mu*g).set_name("mu*g")).set_name("w-mu*g")

def lenet_from_onnx():
    filename = f"mnist-lenet.onnx"
    filepath = f"{BENCH_DIR}/full_dnns/{filename}"
    assert Path(filepath).exists()
    graph = pm.from_onnx(filepath)
    print_skip_nodes = ['write', 'output', 'var_index', 'input', 'index']
    for k,v in graph.nodes.items():
        if v.op_name not in print_skip_nodes:
            print(f"{k} - {v.op_name}")