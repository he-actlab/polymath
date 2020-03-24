
# from polymath.mgdfg.base import Node, placeholder
# from polymath.mgdfg.nodes import constant, index, var_index, variable,\
#     sum, pb_store, pb_load
# import numpy as np
# from .util import compare_nodes
# from polymath.mgdfg.util import visualize
#
# import os

#
# def test_linear():
#     file = "linear.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#
#     pmlang_graph = parse_file(full_path)
#     mgdfg_gen(pmlang_graph.components)
#
#
# def test_backprop():
#     file = "backpropagation.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#     mgdfg_gen(pmlang_graph.components)
#
#
# def test_logistic():
#     file = "logistic.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#     mgdfg_gen(pmlang_graph.components)
#
# def test_recommender():
#     file = "recommender.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#     mgdfg_gen(pmlang_graph.components)
#
# def test_lenet():
#     file = "lenet.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#     mgdfg_gen(pmlang_graph.components)
#
# def test_yolo():
#     file = "yolodnn.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#     mgdfg_gen(pmlang_graph.components)
#
# def test_resnet():
#     file = "resnet18.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     pmlang_graph = parse_file(full_path)
#     mgdfg_gen(pmlang_graph.components)
