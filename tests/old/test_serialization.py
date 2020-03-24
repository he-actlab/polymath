#
# from polymath.mgdfg.serialization.pmlang_mgdfg import mgdfg_gen, parse_file, compile_to_pb, store_pb, load_pb
#
# def test_symbols_to_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#
#     pmlang_graph = parse_file(full_path)
#
#     lr_comp = pmlang_graph.components["rec_model"]
#
#     for var_name, var in lr_comp.symbols.items():
#         var.serialize()
#
# def test_node_to_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#
#     lr_comp = pmlang_graph.components["rec_model"]
#
#     for node in lr_comp.nodes:
#         node.serialize()
#
# def test_edge_to_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#
#     pmlang_graph = parse_file(full_path)
#
#     lr_comp = pmlang_graph.components["rec_model"]
#
#     for edge in lr_comp.edges:
#         edge.serialize()
#
#
# def test_expr_to_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#
#     lr_comp = pmlang_graph.components["rec_model"]
#
#     for expr in lr_comp.expressions:
#         expr.serialize()
#
# def test_comp_to_mgdfg():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#     lr_comp = pmlang_graph.components["main"]
#     test_pb = lr_comp.serialize()
#
#
# def test_serialization_linear():
#     file = "linear.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pb_object = compile_to_pb(full_path)
#
# def test_serialization_backprop():
#     file = "backpropagation.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pb_object = compile_to_pb(full_path)
#
# def test_serialization_logistic():
#     file = "logistic.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pb_object = compile_to_pb(full_path)
#
#
# def test_serialization_recommender():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pb_object = compile_to_pb(full_path)
#
# def test_serialization_lenet():
#     file = "lenet.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pb_object = compile_to_pb(full_path)
#
#
# def test_serialization_yolo():
#     file = "yolodnn.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pb_object = compile_to_pb(full_path)
#
# def test_serialization_resnet():
#     file = "resnet18.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pb_object = compile_to_pb(full_path)
#
#
#
#
#
