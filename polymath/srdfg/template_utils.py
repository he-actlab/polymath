from graphviz import Digraph
import polymath.srdfg.base as poly

from typing import TYPE_CHECKING, Dict
from polymath.pmlang.antlr_generator.parser import InputStream, CommonTokenStream, PMLangParser
from polymath.pmlang.antlr_generator.lexer import PMLangLexer
import polymath.srdfg.serialization.mgdfgv2_pb2 as mgdfg
import polymath.srdfg.base as poly
if TYPE_CHECKING:
    from polymath.srdfg.template import Template
    from polymath.srdfg.graph_objects import Node, Edge

def visualize_component(component: 'Template', filepath, verbose=True):
    vis_graph = Digraph(component.name)
    edge_count = 0
    for node in component._nodes:
        vis_graph.node(str(node.node_id), label=get_node_label(node, verbose=verbose))
        edge_count += len(node.in_edges)
        for in_edge_id in node.in_edges:
            in_edge = component.get_edge(in_edge_id)
            if in_edge.dest_id != node.node_id:
                raise ValueError(f"Destination id for input edge "
                                 f"does not have the correct destination id: "
                                 f"\nEdge dest: {in_edge.edge_str}\n\nNode id: {node.node_str}")

            vis_graph.edge(str(in_edge.source_id), str(in_edge.dest_id), label=get_edge_label(in_edge, verbose=verbose))
    print(f"Edge from nodes: {edge_count}\nEdges from Edges: {len(component.edges)}")
    name = f"{filepath}/{component.name}"
    vis_graph.render(name, view=False)

def get_edge_label(edge: 'Edge', verbose=False) -> str:
    if verbose:
        return edge.edge_str
    else:
        return edge.edge_name

def get_node_label(node: 'Node', verbose=False) -> str:
    if verbose:
        label = f"{node.node_str}"
    else:
        label = f"{node.op_name}"
    return label

def parse_statement_str(statement: str):
    chars = InputStream(statement)
    lexer = PMLangLexer(chars)
    tokens = CommonTokenStream(lexer)
    parser = PMLangParser(tokens)
    parse_stmt = parser.statement()
    return parse_stmt.getChild(0)

def node_from_pb(node: mgdfg.Node) -> 'Node':
    pass

def serialize_graph(components: Dict[str, mgdfg.Template]):
    pass

