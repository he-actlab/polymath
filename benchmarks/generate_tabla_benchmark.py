import sys
sys.path.insert(0, "..")
import numpy as np
from pathlib import Path
import polymath as pm
from tests.util import logistic, linear, reco, svm, backprop
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
def create_logistic(m):
    shape_dict = {"m": m}
    graph, input_info, out_info, keys = logistic(m_=m, coarse=True)
    _, input_info, out_info, keys = logistic(m_=m, coarse=False)
    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

def create_reco(m, n, k):
    shape_dict = {"m": m, "n": n, "k": k}
    graph, input_info, out_info, keys = reco(m_=m, n_=n, k_=k, coarse=True)
    _, input_info, out_info, keys = reco(m_=m, n_=n, k_=k, coarse=False)
    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m}_{n}_{k}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

def create_svm(m):
    shape_dict = {"m": m}
    graph, input_info, out_info, keys = svm(m=m, coarse=True)
    _, input_info, out_info, keys = svm(m=m, coarse=False)
    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

def create_backprop(l1, l2, l3):
    shape_dict = {"l1": l1, "l2": l2 , "l3": l3}
    graph, input_info, out_info, keys = backprop(l1, l2, l3, coarse=True)
    _, input_info, out_info, keys = backprop(l1, l2, l3, coarse=False)

    cwd = Path(f"{__file__}").parent
    full_path = f"{cwd}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{l1}_{l2}_{l3}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

def generate_test_inputs(n):
    n = int(n)
    x = np.random.randint(-3,3, n)
    w = np.random.randint(-3,3, n)
    y = np.random.randint(-3,3, 1)
    return x, w, y

def run_onnx_benchmark(benchmark_name, feature_size):
    filename = f"{benchmark_name}{'-'.join(feature_size)}.onnx"
    filepath = f"{BENCH_DIR}/{benchmark_name}/{filename}"

    if Path(filepath).exists():
        graph = pm.from_onnx(filepath)
        # x, w, y = generate_test_inputs(feature_size[0])
        # # for k1,v1 in graph.nodes.items():
        # #     if v1.op_name == "elem_sub":
        # #         for k, v in v1.nodes.items():
        # #             print(f"{k} - {v.op_name} - {v}")
        # np_res = w - (x.dot(w) - y)*x
        # pm_res = graph("Sub:0", {"y:0": y, "x:0": x, "W:0": w})
    else:
        raise RuntimeError(f"Benchmark {filename} does not exist in {filepath}.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Memory Interface Instructino Generator')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
                                'or "svm".')
    argparser.add_argument('-fs', '--feature_size', nargs='+', required=True,
                           help='Feature size to use for creating the benchmark')
    argparser.add_argument('-o', '--onnx_benchmark', required=True, default=False,
                           help='Determines whether or not to load the benchmark from an ONNX file or not')

    args = argparser.parse_args()
    features = tuple([int(i) for i in args.feature_size])
    if args.onnx_benchmark:
        run_onnx_benchmark(args.benchmark, args.feature_size)
    elif args.benchmark == "linear":
        create_linear(*features)
    elif args.benchmark == "logistic":
        create_logistic(*features)
    elif args.benchmark == "reco":
        create_reco(*features)
    elif args.benchmark == "svm":
        create_svm(*features)
    elif args.benchmark == "backprop":
        create_backprop(*features)
    else:
        raise RuntimeError(f"Invalid benchmark supplied. Options are one of:\n"
                           f"\"logistic\", \"linear\", \"reco\","
                                "or \"svm\".")

