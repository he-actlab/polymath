import sys
sys.path.insert(0, "..")
# sys.path.insert(0, "../tests")

from pathlib import Path
import polymath as pm
from tests.util import logistic, linear, reco, svm, backprop
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Memory Interface Instructino Generator')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "logistic", "linear", "reco",'
                                'or "svm".')
    argparser.add_argument('-fs', '--feature_size', nargs='+', required=True,
                           help='Feature size to use for creating the benchmark')

    args = argparser.parse_args()
    features = tuple([int(i) for i in args.feature_size])
    if args.benchmark == "linear":
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

