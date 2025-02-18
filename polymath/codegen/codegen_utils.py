import numpy as np
import importlib
import pydot
import logging
import sys

CMLANG_CAST_MAP = {
    "int" : np.int,
    "str" : np.str,
    "bool" :np.bool,
    "float": np.float,
    "complex": np.complex
}


def get_func(function_name):
    mod_id, func_id = function_name.rsplit('.', 1)
    try:
        mod = importlib.import_module(mod_id)
    except ModuleNotFoundError:
        logging.error(f"Unable to import {mod_id}. Exiting")
        exit(1)
    func = getattr(mod, func_id)
    return func

def visualize(input_proto,graph, graph_name, output_dir, output_file):
    rankdir = "TB"
    pydot_graph = pydot.Dot(name=input_proto, rankdir=rankdir)

    out_graph = GetPydotGraph(graph, name=graph_name, rankdir=rankdir)
    filename = output_dir + '/' + output_file[:-3] + '.dot'
    pydot_graph.add_subgraph(out_graph)

    pydot_graph.write(filename, format='raw')
    pdf_filename = filename[:-3] + 'png'
    try:
        pydot_graph.write_png(pdf_filename)

    except Exception:
        print(
            'Error when writing out the png file. Pydot requires graphviz '
            'to convert dot files to pdf, and you may not have installed '
            'graphviz. On ubuntu this can usually be installed with "sudo '
            'apt-get install graphviz". We have generated the .dot file '
            'but will not be able to generate png file for now.'
        )
