import polymath as pm
SKIP_OP_TYPES = (pm.state, pm.output, pm.temp, pm.write, pm.placeholder)

def print_graph_ops(graph: pm.Node):
    for name, node in graph.nodes.items():
        if not isinstance(node, SKIP_OP_TYPES):
            print(f"{node.op_name}:{node.name}")
