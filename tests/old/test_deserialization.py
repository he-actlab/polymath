# from polymath.mgdfg.serialization.pmlang_mgdfg import compile_to_pb, store_pb, load_pb, parse_file, mgdfg_from_pb, mgdfg_gen
# from polymath.mgdfg.graph_objects import Edge, Node
# from polymath.mgdfg.template import Template
# from polymath.mgdfg.expression import Expression
# from polymath.mgdfg.variables import Index, Variable
#
#
# from google.protobuf.json_format import MessageToDict
#
# def test_symbol_from_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#
#     orig_graph = parse_file(full_path)
#     orig_components = orig_graph.components
#
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     stored_program = load_pb(load_path)
#     lr_comp = stored_program.templates["rec_model"]
#     orig_lr_comp = orig_components["rec_model"]
#
#     for var in lr_comp.symbols:
#         var_name = var.name
#         if var.HasField("index"):
#             var_index = var.index
#             lbound_name = var_index.WhichOneof("lower")
#             ubound_name = var_index.WhichOneof("upper")
#
#             if lbound_name == "lbound_val":
#                 lower = var.index.lbound_val
#             else:
#                 lower = var.index.lbound_var
#
#             if ubound_name == "ubound_val":
#                 upper = var.index.ubound_val
#             else:
#                 upper = var.index.ubound_var
#             init_var = Index(var.name, lower, upper)
#         else:
#             init_var = Variable(var.name, var.dtype)
#
#         init_var.deserialize(var)
#         if init_var != orig_lr_comp.symbols[var_name]:
#             init_var.debug_eq(orig_lr_comp.symbols[var_name])
#         assert init_var == orig_lr_comp.symbols[var_name]
#
# def test_nodes_from_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#
#     orig_graph = parse_file(full_path)
#     orig_components = orig_graph.components
#
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     stored_program = load_pb(load_path)
#     lr_comp = stored_program.templates["rec_model"]
#     orig_lr_comp = orig_components["rec_model"]
#
#     for node_idx, node in enumerate(lr_comp.nodes):
#         init_node = Node(node.node_id, node.op_name)
#         node_dict = node.DESCRIPTOR.fields_by_name
#         init_node.deserialize(node)
#
#         if init_node != orig_lr_comp.nodes[node_idx]:
#             init_node.debug_eq(orig_lr_comp.nodes[node_idx])
#         assert init_node == orig_lr_comp.nodes[node_idx]
#
# def test_edge_from_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#
#     orig_graph = parse_file(full_path)
#     orig_components = orig_graph.components
#
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     stored_program = load_pb(load_path)
#     lr_comp = stored_program.templates["rec_model"]
#     orig_lr_comp = orig_components["rec_model"]
#
#     for edge_idx, edge in enumerate(lr_comp.edges):
#         init_edge = Edge(edge.edge_id)
#         init_edge.deserialize(edge)
#         if init_edge != orig_lr_comp.edges[edge_idx]:
#             init_edge.debug_eq(orig_lr_comp.edges[edge_idx])
#         assert init_edge == orig_lr_comp.edges[edge_idx]
#
# def test_expr_from_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#
#     orig_graph = parse_file(full_path)
#     orig_components = orig_graph.components
#
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     stored_program = load_pb(load_path)
#     lr_comp = stored_program.templates["rec_model"]
#     orig_lr_comp = orig_components["rec_model"]
#
#     for expr_idx, expr in enumerate(lr_comp.expressions):
#         init_expr = Expression(expr.expr_str)
#         init_expr.deserialize(expr)
#         if init_expr != orig_lr_comp.expressions[expr_idx]:
#             init_expr.debug_eq(orig_lr_comp.expressions[expr_idx])
#         assert init_expr == orig_lr_comp.expressions[expr_idx]
#
#
# def test_comp_from_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#
#     orig_graph = parse_file(full_path)
#     orig_components = orig_graph.components
#
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     stored_program = load_pb(load_path)
#     lr_comp = stored_program.templates["rec_model"]
#     orig_lr_comp = orig_components["rec_model"]
#     new_comp = Template("rec_model")
#     new_comp.deserialize(lr_comp)
#     if new_comp != orig_lr_comp:
#         new_comp.debug_eq(orig_lr_comp)
#     assert new_comp == orig_lr_comp
#
#
#
# def test_load_serialized_linear():
#     file = "linear.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#     orig_graph = parse_file(full_path)
#     comp_dict_ = orig_graph.components
#     main_comp_ = mgdfg_gen(comp_dict_)
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     comp_dict, main_comp = mgdfg_from_pb(load_path)
#     for comp_name, comp in comp_dict.items():
#         if comp != comp_dict_[comp_name]:
#             comp.debug_eq(comp_dict_[comp_name])
#         assert comp == comp_dict_[comp_name]
#
#     if main_comp != main_comp_:
#         main_comp.debug_eq(main_comp_)
#     assert main_comp == main_comp_
#
# #
# def test_load_serialized__backprop():
#     file = "backpropagation.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#     orig_graph = parse_file(full_path)
#     comp_dict_ = orig_graph.components
#     main_comp_ = mgdfg_gen(comp_dict_)
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     comp_dict, main_comp = mgdfg_from_pb(load_path)
#     for comp_name, comp in comp_dict.items():
#         if comp != comp_dict_[comp_name]:
#             comp.debug_eq(comp_dict_[comp_name])
#         assert comp == comp_dict_[comp_name]
#
#     if main_comp != main_comp_:
#         main_comp.debug_eq(main_comp_)
#     assert main_comp == main_comp_
#
# def test_load_serialized__logistic():
#     file = "logistic.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#     orig_graph = parse_file(full_path)
#     comp_dict_ = orig_graph.components
#     main_comp_ = mgdfg_gen(comp_dict_)
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     comp_dict, main_comp = mgdfg_from_pb(load_path)
#     for comp_name, comp in comp_dict.items():
#         if comp != comp_dict_[comp_name]:
#             comp.debug_eq(comp_dict_[comp_name])
#         assert comp == comp_dict_[comp_name]
#
#     if main_comp != main_comp_:
#         main_comp.debug_eq(main_comp_)
#     assert main_comp == main_comp_
#
# def test_load_serialized__recommender():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#     orig_graph = parse_file(full_path)
#     comp_dict_ = orig_graph.components
#     main_comp_ = mgdfg_gen(comp_dict_)
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     comp_dict, main_comp = mgdfg_from_pb(load_path)
#     for comp_name, comp in comp_dict.items():
#         if comp != comp_dict_[comp_name]:
#             comp.debug_eq(comp_dict_[comp_name])
#         assert comp == comp_dict_[comp_name]
#
#     if main_comp != main_comp_:
#         main_comp.debug_eq(main_comp_)
#     assert main_comp == main_comp_
#
# def test_load_serialized__lenet():
#     file = "lenet.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#     orig_graph = parse_file(full_path)
#     comp_dict_ = orig_graph.components
#     main_comp_ = mgdfg_gen(comp_dict_)
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     comp_dict, main_comp = mgdfg_from_pb(load_path)
#     for comp_name, comp in comp_dict.items():
#         if comp != comp_dict_[comp_name]:
#             comp.debug_eq(comp_dict_[comp_name])
#         assert comp == comp_dict_[comp_name]
#
#     if main_comp != main_comp_:
#         main_comp.debug_eq(main_comp_)
#     assert main_comp == main_comp_
#
# def test_load_serialized__yolo():
#     file = "yolodnn.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#     orig_graph = parse_file(full_path)
#     comp_dict_ = orig_graph.components
#     main_comp_ = mgdfg_gen(comp_dict_)
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     comp_dict, main_comp = mgdfg_from_pb(load_path)
#     for comp_name, comp in comp_dict.items():
#         if comp != comp_dict_[comp_name]:
#             comp.debug_eq(comp_dict_[comp_name])
#         assert comp == comp_dict_[comp_name]
#
#     if main_comp != main_comp_:
#         main_comp.debug_eq(main_comp_)
#     assert main_comp == main_comp_
#
# def test_load_serialized__resnet():
#     file = "resnet18.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     output_path = f"{base_path}/outputs/"
#     orig_graph = parse_file(full_path)
#     comp_dict_ = orig_graph.components
#     main_comp_ = mgdfg_gen(comp_dict_)
#     pb_object = compile_to_pb(full_path, orig_listener=orig_graph)
#     store_pb(output_path, pb_object)
#
#     load_path = f"{output_path}/{pb_object.name}.pb"
#     comp_dict, main_comp = mgdfg_from_pb(load_path)
#     for comp_name, comp in comp_dict.items():
#         if comp != comp_dict_[comp_name]:
#             comp.debug_eq(comp_dict_[comp_name])
#         assert comp == comp_dict_[comp_name]
#
#     if main_comp != main_comp_:
#         main_comp.debug_eq(main_comp_)
#     assert main_comp == main_comp_