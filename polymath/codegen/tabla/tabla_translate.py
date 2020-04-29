import polymath as pm
from polymath.codegen.tabla.tabla_pass import TablaPass
import json

def generate_tabla(graph, input_dict, filepath, context_dict=None, add_kwargs=False, debug=True):
    assert len(input_dict) > 0
    shape_pass = pm.NormalizeGraph(input_dict)

    tabla_pass = TablaPass(context_dict, add_kwargs=add_kwargs, debug=debug)
    lower_pass = pm.Lower({})
    print(f"Starting graph normalization...")
    shaped = shape_pass(graph)
    print(f"Finished graph normalization. Executing lower pass.")
    lowered = lower_pass(shaped)
    print(f"Finished graph lowering, generating TABLA dfg.")
    res = tabla_pass(lowered)
    print(f"Finished generating TABLA dfg, now storing to JSON file at {filepath}.")

    tabla_nodes = [node for _, node in tabla_pass.dfg.items()]

    with open(filepath, "w") as f:
        json.dump(tabla_nodes, f, indent=4)

    return tabla_nodes, res


