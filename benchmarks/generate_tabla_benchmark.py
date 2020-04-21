import sys
sys.path.insert(0, "..")
# sys.path.insert(0, "../tests")

from pathlib import Path
import polymath as pm
from tests.util import logistic, linear, reco, svm
import argparse

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
    tabla_path = f"{full_path}/{graph.name}_{m}_tabla.json"

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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Memory Interface Instructino Generator')
    argparser.add_argument('-b', '--benchmark',
                           help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
                                'or "svm".')
    argparser.add_argument('-fs', '--feature_size',
                           help='Feature size to use for creating the benchmark')

    args = argparser.parse_args()

    if args.benchmark == "linear":
        create_linear(int(args.feature_size))
    elif args.benchmark == "logistic":
        create_logistic(int(args.feature_size))
    elif args.benchmark == "reco":
        create_reco(*args.feature_size)
    elif args.benchmark == "svm":
        create_svm(int(args.feature_size))
    else:
        raise RuntimeError(f"Invalid benchmark supplied. Options are one of:\n"
                           f"\"logistic\", \"linear\", \"reco\","
                                "or \"svm\".")

