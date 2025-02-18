import polymath as pm
from polymath.codegen.tabla.tabla_pass import TablaPass
import json

def generate_tabla(graph, input_dict, filepath, context_dict=None, add_kwargs=False, debug=True):
    assert len(input_dict) > 0
    shape_pass = pm.NormalizeGraph(input_dict, debug=debug)
    context_dict = context_dict or {}

    lower_pass = pm.Lower({}, debug=debug)
    print(f"Starting graph normalization...")

    shaped = shape_pass(graph)
    # for k, n in shaped.nodes.items():
    #     if "tempz" in k:
    #         print(f"{n.name} - {n.op_name} - {n.shape}")
    print(f"Finished graph normalization. Executing lower pass.")
    lowered = lower_pass(shaped)

    # print(list(lowered.nodes.keys()))
    print(f"Finished graph lowering, generating TABLA dfg.")
    for k in list(context_dict.keys()):
        if k not in lowered.nodes:
            context_dict.pop(k)
    tabla_pass = TablaPass(context_dict, add_kwargs=add_kwargs, debug=debug)
    res = tabla_pass(lowered)
    print(f"Finished generating TABLA dfg, now storing to JSON file at {filepath}.")

    tabla_nodes = [node for _, node in tabla_pass.dfg.items()]

    with open(filepath, "w") as f:
        json.dump(tabla_nodes, f, indent=4)

    return tabla_nodes, res


