# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
#
# import json
# from collections import defaultdict
# from typing import Text, Any, Callable, Optional, Dict
#
# import pydot  # type: ignore
# from onnx import GraphProto, NodeProto
#
# OP_STYLE = {
#     'shape': 'box',
#     'color': '#0F9D58',
#     'style': 'filled',
#     'fontcolor': '#FFFFFF'
# }
#
# BLOB_STYLE = {'shape': 'octagon'}
#
# _NodeProducer = Callable[[NodeProto, int], pydot.Node]
#
#
# def _escape_label(name):  # type: (Text) -> Text
#     # json.dumps is poor man's escaping
#     return json.dumps(name)
#
#
# def _form_and_sanitize_docstring(s):  # type: (Text) -> Text
#     url = 'javascript:alert('
#     url += _escape_label(s).replace('"', '\'').replace('<', '').replace('>', '')
#     url += ')'
#     return url
#
#
# def GetOpNodeProducer(embed_docstring=False, **kwargs):  # type: (bool, **Any) -> _NodeProducer
#     def ReallyGetOpNode(op, op_id):  # type: (NodeProto, int) -> pydot.Node
#         if op.name:
#             node_name = '%s/%s (op#%d)' % (op.name, op.op_type, op_id)
#         else:
#             node_name = '%s (op#%d)' % (op.op_type, op_id)
#         for i, input in enumerate(op.input):
#             node_name += '\n input' + str(i) + ' ' + input
#         for i, output in enumerate(op.output):
#             node_name += '\n output' + str(i) + ' ' + output
#         node = pydot.Node(node_name, **kwargs)
#         if embed_docstring:
#             url = _form_and_sanitize_docstring(op.doc_string)
#             node.set_URL(url)
#         return node
#     return ReallyGetOpNode
#
#
# def GetPydotGraph(
#     graph,  # type: GraphProto
#     name=None,  # type: Optional[Text]
#     rankdir='LR',  # type: Text
#     node_producer=None,  # type: Optional[_NodeProducer]
#     embed_docstring=False,  # type: bool
# ):  # type: (...) -> pydot.Dot
#     if node_producer is None:
#         node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **OP_STYLE)
#     pydot_graph = pydot.Dot(name, rankdir=rankdir)
#     pydot_nodes = {}  # type: Dict[Text, pydot.Node]
#     pydot_node_counts = defaultdict(int)  # type: Dict[Text, int]
#     for op_id, op in enumerate(graph.node):
#         op_node = node_producer(op, op_id)
#         pydot_graph.add_node(op_node)
#         for input_name in op.input:
#             if input_name not in pydot_nodes:
#                 input_node = pydot.Node(
#                     _escape_label(
#                         input_name + str(pydot_node_counts[input_name])),
#                     label=_escape_label(input_name),
#                     **BLOB_STYLE
#                 )
#                 pydot_nodes[input_name] = input_node
#             else:
#                 input_node = pydot_nodes[input_name]
#             pydot_graph.add_node(input_node)
#             pydot_graph.add_edge(pydot.Edge(input_node, op_node))
#         for output_name in op.output:
#             if output_name in pydot_nodes:
#                 pydot_node_counts[output_name] += 1
#             output_node = pydot.Node(
#                 _escape_label(
#                     output_name + str(pydot_node_counts[output_name])),
#                 label=_escape_label(output_name),
#                 **BLOB_STYLE
#             )
#             pydot_nodes[output_name] = output_node
#             pydot_graph.add_node(output_node)
#             pydot_graph.add_edge(pydot.Edge(op_node, output_node))
#     return pydot_graph

