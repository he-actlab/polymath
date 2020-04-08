import polymath as pm
from polymath.codegen.tabla.tabla_pass import TablaPass
import json

def generate_tabla(graph, input_dict, filepath, context_dict=None, add_kwargs=False):
    assert len(input_dict) > 0
    shape_pass = pm.NormalizeGraph(input_dict)
    tabla_pass = TablaPass(context_dict, add_kwargs=add_kwargs)
    lower_pass = pm.Lower({})
    shaped = shape_pass(graph)

    lowered = lower_pass(shaped)
    res= tabla_pass(lowered)
    tabla_nodes = [node for _, node in tabla_pass.dfg.items()]

    with open(filepath, "w") as f:
        json.dump(tabla_nodes, f, indent=4)

    return tabla_nodes, res


