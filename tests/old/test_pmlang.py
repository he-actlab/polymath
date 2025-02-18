# from polymath.pmlang.antlr_generator.parser import FileStream, CommonTokenStream, PMLangParser, ParseTreeWalker
# from polymath.pmlang.antlr_generator.lexer import PMLangLexer
# from polymath.pmlang.symbols import PMLangListener
# from polymath.mgdfg.template_utils import visualize_component, parse_statement_str
# from polymath.mgdfg.serialization.pmlang_mgdfg import parse_file
#
# import os
#
# def test_temp():
#     class TempAssign(object):
#         def __init__(self):
#             self.test_val = 1
#
#     class TempChild(object):
#         def __init__(self, parent):
#             self.parent = parent
#     temp_assign = TempAssign()
#     temp_child = TempChild(temp_assign)
#     print(f"Parent: {temp_assign.test_val}\tChild: {temp_child.parent.test_val}")
#     temp_child.parent.test_val = 2
#     print(f"Parent: {temp_assign.test_val}\tChild: {temp_child.parent.test_val}")
#
# def test_backprop():
#     file = "backpropagation.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#     ext_full_path = os.path.abspath(base_path) + "/outputs"
#     visualize_component(pmlang_graph.components["avg_pool2d"], ext_full_path)
#
#
# def test_linear():
#     file = "linear.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#
#     pmlang_graph = parse_file(full_path)
#     ext_full_path = os.path.abspath(base_path) + "/outputs"
#     visualize_component(pmlang_graph.components["linear_regression"], ext_full_path)
#
# def test_logistic():
#     file = "logistic.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#
#     ext_full_path = os.path.abspath(base_path) + "/outputs"
#     visualize_component(pmlang_graph.components["main"], ext_full_path)
#
# def test_recommender():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#
#     ext_full_path = os.path.abspath(base_path) + "/outputs"
#     visualize_component(pmlang_graph.components["rec_model"], ext_full_path)
#     # load_store.save_program(pmlang_graph.program, output_mgdfg)
#
# def test_lenet():
#     file = "lenet.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#
#     ext_full_path = os.path.abspath(base_path) + "/outputs"
#     visualize_component(pmlang_graph.components["batch_flatten"], ext_full_path)
#
# def test_yolo():
#     file = "yolodnn.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     ext_full_path = os.path.abspath(base_path) + "/outputs"
#     pmlang_graph = parse_file(full_path)
#     visualize_component(pmlang_graph.components["batch_norm"], ext_full_path)
#
#
#
# def test_resnet():
#     file = "resnet18.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#
#
# def test_statement_str_parse():
#     statements = {"h = sum[i](w[i] * x[i]);": "assignment",
#                   "w[i] = w[i] - mu*g[i];": "assignment",
#                   "out[i][j] = 1.0 / (1.0 + e()^(-in[i][j]));": "assignment",
#                   "out[i][j][y][x] = in[i][j][y][x] > 0 ? in[i][j][y][x] : in[i][j][y][x]*alpha;": "assignment",
#                   "result[b][c][y][i] = sum[dy][dx][ic](padded_input[b][k][strides*i + dx][strides*y + dy]*kernels[c][k][dy][dx]);": "assignment",
#                   "add_bias(conv1_out, c1_bias,c1_bias_out);": "expression",
#                   "float conv1_weight[oc][ic][kernel_size][kernel_size], conv1_out[n][oc][oh][ow];": "declaration",
#                   "h = sigmoid(sum[i](w[i] * x[i]));": "assignment"
#                   }
#
#     for stat, stat_type in statements.items():
#         ast_obj = parse_statement_str(stat)
#         if stat_type == "assignment":
#             assert isinstance(ast_obj, PMLangParser.Assignment_statementContext)
#         elif stat_type == "declaration":
#             assert isinstance(ast_obj, PMLangParser.Declaration_statementContext)
#         elif stat_type == "expression":
#             assert isinstance(ast_obj, PMLangParser.Expression_statementContext)
#
#
