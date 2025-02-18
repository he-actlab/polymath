import sys
sys.path.insert(0, "..")
import numpy as np
from pathlib import Path
import polymath as pm
from tests.util import logistic, linear, reco, svm, backprop, svm_wifi_datagen
import argparse
BENCH_DIR = Path(f"{Path(__file__).parent}/onnx_files")

def create_linear(m):
    shape_dict = {"m": m}
    graph, input_info, out_info, keys = linear(m=m, coarse=True)
    _, input_info, out_info, keys = linear(m=m, coarse=False)
    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)
def create_logistic(m, onnx_graph=None):
    shape_dict = {"m": m}
    graph, input_info, out_info, keys = logistic(m=m, coarse=True)
    _, input_info, out_info, keys = logistic(m=m, coarse=False)
    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m}_tabla.json"
    if onnx_graph is not None:
        graph = onnx_graph
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

def create_reco(m, n, k, onnx_graph=None):
    shape_dict = {"m": m, "n": n, "k": k}
    graph, input_info, out_info, keys = reco(m_=m, n_=n, k_=k, coarse=True)
    _, input_info, out_info, keys = reco(m_=m, n_=n, k_=k, coarse=False)
    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m}_{n}_{k}_tabla.json"
    if onnx_graph is not None:
        graph = onnx_graph
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

def create_svm(m, onnx_graph=None):

    shape_dict = {"m": m}
    graph, input_info, out_info, keys = svm(m=m, coarse=True)
    _, input_info, out_info, keys = svm(m=m, coarse=False)
    if onnx_graph is not None:
        graph = onnx_graph
    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

def create_svm_wifi(features, locations, lr=0.0001, deltav=1, train_size=7703):
    with pm.Node(name="svm_wifi") as graph:
        learning_rate = pm.parameter("learning_rate", default=lr)
        delta = pm.parameter("delta", default=deltav)
        n_features = pm.parameter("n_features", default=features)
        n_locations = pm.parameter("n_locations", default=locations)
        x_train = pm.input("x_train", shape=(n_features,))
        y_train = pm.input("y_train", shape=(n_locations,))
        y_train_inv = pm.input("y_train_inv", shape=(n_locations,))
        weights = pm.state("weights", shape=(n_features, n_locations))

        i = pm.index(0, n_features - 1, name="i")
        j = pm.index(0, n_locations - 1, name="j")

        scores = pm.sum([i], (weights[i, j] * x_train[i]), name="scores")
        correct_class_score = pm.sum([j], (scores[j] * y_train[j]), name="correct_class_score")

        h = ((scores[j] - correct_class_score + delta).set_name("h") > 0)

        # margin = (pm.cast(np.float32, h[j]) * y_train_inv[j]).set_name("margin")
        margin = (h[j] * y_train_inv[j]).set_name("margin")
        valid_margin_count = pm.sum([j], margin[j], name="valid_margin_count")
        partial = (y_train[j] * valid_margin_count).set_name("partial")
        updated_margin = (margin[j] - partial[j]).set_name("updated_margin")
        # # #
        dW = (x_train[i] * updated_margin[j]).set_name("dW")
        weights[i, j] = (weights[i, j] - learning_rate * dW[i, j]).set_name("weights_update")

    shape_dict = {"n_features": features, "n_locations": locations}
    input_info, keys, out_info = svm_wifi_datagen(features, locations, lr, deltav, lowered=True)

    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{locations}_{features}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)


def create_backprop(l1, l2, l3, onnx_graph=None):
    shape_dict = {"l1": l1, "l2": l2 , "l3": l3}
    graph, input_info, out_info, keys = backprop(l1, l2, l3, coarse=True)
    _, input_info, out_info, keys = backprop(l1, l2, l3, coarse=False, debug=True)

    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{l1}_{l2}_{l3}_tabla.json"
    if onnx_graph is not None:
        graph = onnx_graph
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)
def generate_test_inputs(n):
    n = int(n)
    x = np.random.randint(-3,3, n)
    w = np.random.randint(-3,3, n)
    y = np.random.randint(-3,3, 1)
    y = np.random.randint(-3,3, 1)
    return x, w, y

def run_onnx_benchmark(benchmark_name, feature_size):
    filename = f"{benchmark_name}{'-'.join(feature_size)}.onnx"
    filepath = f"{BENCH_DIR}/ml_algorithms/{filename}"

    if Path(filepath).exists():
        features = tuple([int(i) for i in feature_size])
        graph = pm.from_onnx(filepath)
        if benchmark_name == "svm":
            create_svm(*features, graph)
        elif benchmark_name == "reco":
            create_reco(*features, graph)
        elif benchmark_name == "logistic":
            create_logistic(*features)
        elif benchmark_name == "svm_wifi":
            create_logistic(*features)
    else:
        raise RuntimeError(f"Benchmark {filename} does not exist in {filepath}.")


if __name__ == "__main__":
    # run_onnx_benchmark("svm_wifi", ['20', '30'])
    create_svm_wifi(139, 325)

    # create_svm(*(54,))
    # argparser = argparse.ArgumentParser(description='Memory Interface Instructino Generator')
    # argparser.add_argument('-b', '--benchmark', required=True,
    #                        help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
    #                             'or "svm".')
    # argparser.add_argument('-fs', '--feature_size', nargs='+', required=True,
    #                        help='Feature size to use for creating the benchmark')
    # argparser.add_argument('-o', '--onnx_benchmark', required=False, default=False,
    #                        help='Determines whether or not to load the benchmark from an ONNX file or not')
    #
    # args = argparser.parse_args()
    # features = tuple([int(i) for i in args.feature_size])
    # if args.onnx_benchmark:
    #     run_onnx_benchmark(args.benchmark, args.feature_size)
    # elif args.benchmark == "linear":
    #     create_linear(*features)
    # elif args.benchmark == "logistic":
    #     create_logistic(*features)
    # elif args.benchmark == "reco":
    #     create_reco(*features)
    # elif args.benchmark == "svm":
    #     create_svm(*features)
    # elif args.benchmark == "backprop":
    #     create_backprop(*features)
    # else:
    #     raise RuntimeError(f"Invalid benchmark supplied. Options are one of:\n"
    #                        f"\"logistic\", \"linear\", \"reco\","
    #                             "or \"svm\".")

