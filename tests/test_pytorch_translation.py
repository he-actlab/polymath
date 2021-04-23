# import polymath as pm
# from tests.util import linear, op_counts, logistic, svm, reco,\
#     dense, conv, two_layer_dense, pooling, backprop, batchnorm, \
#     global_avg_pool, lrn, np_softmax
# from pathlib import Path
# import pickle
# import pytest
# import pprint
# import numpy as np
# import copy
# BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks/onnx_files")
# CWD = Path(f"{__file__}").parent
# BASE_PATH = f"{CWD}/pmlang_examples"
# OUTPATH = f"{BASE_PATH}/outputs"
# ONNX_FILE_DIR = Path(f"{Path(__file__).parent}/onnx_examples")
#
#
# @pytest.mark.parametrize('benchmark_name, feature_dict, data_func, input_keys, output_key',[
#     ("linear", {'m': 54}, linear, {"y":"y:0", "x":"x:0", "w":"W:0"}, [("w", "W:0")]),
#     ("logistic", {'m': 54}, logistic, {"y":"y:0", "x":"x:0", "w":"W:0"}, [("w", "W:0")]),
#     ("svm", {'m': 54}, svm, {"y":"y:0", "x":"x:0", "w":"W:0"}, [("c", "mul_1:0")]),
#     ("backprop", {'l1': 8, 'l2':16, 'l3':4}, backprop, {"y":"y:0", "x":"x:0", "w1":"W1:0","w2":"W2:0"}, [("w1", "W1:0"), ("w2", "W2:0")]),
#     ("recommender", {'m': 30, 'n':28 , 'k': 3}, reco, {"x1":"x1:0", "x2":"x2:0", "w1":"w1:0", "w2":"w2:0",
#                                      "y1":"y1:0", "y2":"y2:0","r2":"r2:0", "r1":"r1:0"}, [("w1", "w1:0"),("w2", "w2:0")]),
# ])
# def test_convert_benchmarks(benchmark_name, feature_dict, data_func, input_keys, output_key):
#     feature_size = [str(v) for k,v in feature_dict.items()]
#     tabla_path = f"{OUTPATH}/{benchmark_name}_{'_'.join(feature_size)}_onnx_tabla.json"
#     ref_tabla_path = f"{OUTPATH}/{benchmark_name}_{'_'.join(feature_size)}_tabla.json"
#     filename = f"{benchmark_name}{'_'.join(feature_size)}.onnx"
#     filepath = f"{BENCH_DIR}/ml_algorithms/{filename}"
#     assert Path(filepath).exists()
#     graph = pm.from_pytorch(filepath, use_filename=False)
#     # Apply transformations and/or generate verilog using 'transformed_graph'
#
#     int_feat_dict = {k: int(v) for k,v  in feature_dict.items()}
#     _, ref_in_info, ref_out_info, ref_keys = data_func(**int_feat_dict)
#
#     int_feat_dict['coarse'] = True
#     ref_graph, in_info, out_info, ref_keys = data_func(**int_feat_dict)
#     translated_inputs = {input_keys[k]: v for k,v in in_info.items() if k in input_keys}
#     for i in output_key:
#         input_cpy = pickle.loads(pickle.dumps(translated_inputs))
#         np_res = out_info[i[0]]
#         onnx_res = graph(i[1], input_cpy)
#         np.testing.assert_allclose(np.squeeze(np_res), np.squeeze(onnx_res))
#
#     print(f"Starting tabla compilation\n\n")
#     tabla_ir, tabla_graph = pm.generate_tabla(graph,
#                                               feature_dict,
#                                               tabla_path,debug=False,
#                                               context_dict={}, add_kwargs=True)
#     ref_tabla_ir, ref_tabla_graph = pm.generate_tabla(ref_graph,
#                                               feature_dict,
#                                               ref_tabla_path,debug=False,
#                                               context_dict={}, add_kwargs=True)
#
#     ref_ocount_pass = pm.CountOpTypes(skip=['temp', 'parameter', ref_tabla_graph.name])
#     _ = ref_ocount_pass(ref_tabla_graph)
#     ocount_pass = pm.CountOpTypes(skip=['temp', 'parameter', 'output', 'write', tabla_graph.name])
#     _ = ocount_pass(tabla_graph)
#     pprint.pprint(ref_ocount_pass.op_types)
#     pprint.pprint(ocount_pass.op_types)
#     if set(ocount_pass.op_types.keys()) != set(ref_ocount_pass.op_types.keys()):
#         raise RuntimeError(f"Unequal amounts of operations for graphs:\n"
#               f"\tReference: {ref_ocount_pass.op_types.keys()}\n"
#               f"\tActual: {ocount_pass.op_types.keys()}")
#
#     for k,v in ocount_pass.op_types.items():
#         if v != ref_ocount_pass.op_types[k]:
#             raise RuntimeError(f"Unequal operations for key {k}:\n"
#                                f"\tRef: {ref_ocount_pass.op_types[k]}\n"
#                                f"\tActual: {v}\n")
#
#
#     assert len(ref_tabla_ir) == len(tabla_ir)
#
# @pytest.mark.parametrize('m',[
#     100
# ])
# def test_load_logistic(m):
#     benchmark_name = "logistic"
#     feature_dict = {'m': m}
#     input_keys = {"y":"y:0", "x":"x:0", "w":"W:0"}
#     output_key = [("w", "W:0")]
#     feature_size = [str(v) for k, v in feature_dict.items()]
#     filename = f"{benchmark_name}{'_'.join(feature_size)}.onnx"
#     filepath = f"{BENCH_DIR}/ml_algorithms/{filename}"
#     assert Path(filepath).exists()
#     graph = pm.from_pytorch(filepath)
#     int_feat_dict = {k: int(v) for k, v in feature_dict.items()}
#     _, ref_in_info, ref_out_info, ref_keys = logistic(**int_feat_dict)
#     int_feat_dict['coarse'] = True
#     ref_graph, in_info, out_info, ref_keys = logistic(**int_feat_dict)
#     translated_inputs = {input_keys[k]: v for k,v in in_info.items() if k in input_keys}
#     for i in output_key:
#         input_cpy = pickle.loads(pickle.dumps(translated_inputs))
#         np_res = out_info[i[0]]
#         onnx_res = graph(i[1], input_cpy)
#         np.testing.assert_allclose(np.squeeze(np_res), np.squeeze(onnx_res))
#
# @pytest.mark.parametrize('filepath',[
#     f"{BENCH_DIR}/ml_algorithms/logistic_"
# ])
# def test_load_onnx_files(filepath):
#     pass
#
# @pytest.mark.parametrize('m_',[
#     3
# ])
# def test_load_linear_regressor(m_):
#     shape_dict = {"m": m_}
#     m = pm.parameter("m")
#     mu = pm.parameter(name="mu", default=1.0)
#     x = pm.input("x", shape=(m))
#     y = pm.input("y")
#     w = pm.state("w", shape=(m))
#
#     graph = pm.linear_regressor_train(x, w, y, mu, m)
#     test_graph, input_info, out_info, keys = linear(m=m_, coarse=True)
#     assert len(test_graph.nodes.keys()) == len(graph.nodes.keys())
#     assert op_counts(test_graph) == op_counts(graph)
#
#     shape_val_pass = pm.NormalizeGraph(shape_dict)
#     new_graph = shape_val_pass(graph)
#     test_res = new_graph(keys, input_info)
#     np.testing.assert_allclose(test_res, out_info["w"])
#
#     test_graph_lowered, input_info, new_out_info, keys = linear(m=m_)
#     flatten_pass = pm.Lower({})
#     test_flatten_pass = pm.Lower({})
#     flattened_g = flatten_pass(new_graph)
#     ref_lowered = test_flatten_pass(test_graph_lowered, {})
#     assert len(ref_lowered.nodes.keys()) == len(flattened_g.nodes.keys())
#     assert op_counts(ref_lowered) == op_counts(flattened_g)
#
#     all_vals = flattened_g(keys, input_info)
#     np.testing.assert_allclose(new_out_info["w"], all_vals)
#
# @pytest.mark.parametrize('m_',[
#     3
# ])
# def test_load_nested_linear_regressor(m_):
#     shape_dict = {"m": m_}
#     with pm.Node(name="nested_linear") as graph:
#         m = pm.parameter(name="m")
#         mu = pm.parameter(name="mu", default=1.0)
#         x = pm.input("x", shape=(m))
#         y = pm.input("y")
#         w = pm.state("w", shape=(m))
#         pm.linear_regressor_train(x, w, y, mu, m, name="linear_regressor")
#         j = pm.index(0, m-1, name="j")
#         tw = (w[j] - 4).set_name("tw")
#
#     test_graph, input_info, out_info, keys = linear(m=m_, coarse=True)
#     shape_val_pass = pm.NormalizeGraph(shape_dict)
#     new_graph = shape_val_pass(graph)
#     test_res = new_graph("tw", input_info)
#     np.testing.assert_allclose(test_res, (out_info["w"] - 4))
#
#     ref_graph, input_info, new_out_info, keys = linear(m=m_)
#     flatten_pass = pm.Lower({})
#     keys = [f"tw/tw({i},)" for i in range(m_)]
#
#     flattened_g = flatten_pass(new_graph)
#     all_vals = flattened_g(keys, input_info)
#
# @pytest.mark.parametrize('m',[
#   55
# ])
# def test_translate_linear_regressor(m):
#     out_key_map = {"y": "y:0", "x": "x:0", "w": "W:0"}
#     in_key_map = [("w", "W:0")]
#     fpath = f"{ONNX_FILE_DIR}/linear_{m}.onnx"
#     shape_dict = {"m": m}
#     graph = pm.from_pytorch(fpath)
#     test_graph, input_info, out_info, keys = linear(m=m, coarse=True)
#     tinput_info = copy.deepcopy(input_info)
#     tkeys = copy.deepcopy(keys)
#     test_res = test_graph(tkeys, tinput_info)
#     np.testing.assert_allclose(test_res, (out_info["w"]))
#
#     onx_input_info = copy.deepcopy(input_info)
#     translated_inputs = {out_key_map[k]: v for k,v in input_info.items() if k in out_key_map}
#     onnx_res = graph(in_key_map[0][1], translated_inputs)
#
#     np.testing.assert_allclose(onnx_res, (out_info["w"]))
#
#     tabla_path = f"{OUTPATH}/{graph.name}{m}_tabla.json"
#     tabla_ir = pm.generate_tabla(graph,
#                                   shape_dict,
#                                   tabla_path)
#
#
#
# @pytest.mark.parametrize('m',[
#     54
# ])
# def test_translate_svm(m):
#     out_key_map = {"y": "y:0", "x": "x:0", "w": "W:0"}
#     in_key_map = [("w", "W:0")]
#
#     fpath = f"{ONNX_FILE_DIR}/svm_{m}.onnx"
#     shape_dict = {"m": m}
#     graph = pm.from_pytorch(fpath)
#
#     test_graph, input_info, out_info, keys = svm(m=m, coarse=True)
#     tinput_info = copy.deepcopy(input_info)
#
#     tkeys = copy.deepcopy(keys)
#     test_res = test_graph(tkeys, tinput_info)
#     np.testing.assert_allclose(test_res, (out_info["w"]))
#
#     translated_inputs = {out_key_map[k]: v for k,v in input_info.items() if k in out_key_map}
#
#     onnx_res = graph(in_key_map[0][1], translated_inputs)
#
#     np.testing.assert_allclose(onnx_res, (out_info["w"]))
#     tabla_path = f"{OUTPATH}/{graph.name}{m}_tabla.json"
#     tabla_ir, tabla_graph = pm.generate_tabla(graph,
#                                               shape_dict,
#                                               tabla_path, debug=False)
#
# @pytest.mark.parametrize('m, n, k', [
#     (3, 3, 2),
# ])
# def test_translate_reco(m, n, k):
#     shape_dict = {"m": m, "n": n, "k": k}
#     test_graph, input_info, out_info, keys = reco(m=m, n=n, k=k, coarse=True)
#
# @pytest.mark.parametrize('x_shape, w_shape', [
#     ((4,), (5, 4)),
# ])
# def test_translate_dense(x_shape, w_shape):
#
#     graph, input_info, out_info, keys = dense(x_shape, w_shape, coarse=True, debug_matrix=True)
#     tinput_info = copy.deepcopy(input_info)
#     res0 = graph("y", tinput_info)
#
#     np.testing.assert_allclose(res0, out_info["y"].astype(np.int32))
#
#     graph, input_info, out_info, keys = dense(x_shape, w_shape, coarse=False, debug_matrix=True)
#
#     lower_pass = pm.Lower({})
#     lowered_graph = lower_pass(graph)
#     res = lowered_graph(keys, input_info)
#     np.testing.assert_allclose(np.asarray(res).reshape(out_info["y"].shape), out_info["y"].astype(np.int32))
#
#
# @pytest.mark.parametrize('x1_shape, w1_shape, w2_shape', [
#     ((4,), (5, 4), (3, 5)),
# ])
# def test_translate_multi_dense(x1_shape, w1_shape, w2_shape):
#
#     graph, input_info, out_info, keys = two_layer_dense(x1_shape, w1_shape, w2_shape, coarse=True, debug_matrix=True)
#
#     tinput_info = copy.deepcopy(input_info)
#     res0 = graph(keys, tinput_info)
#     np.testing.assert_allclose(res0, out_info["y"].astype(res0.dtype))
#
#     graph, input_info, out_info, keys = two_layer_dense(x1_shape, w1_shape, w2_shape, coarse=False, debug_matrix=True)
#
#     lower_pass = pm.Lower({})
#     lowered_graph = lower_pass(graph)
#     res = lowered_graph(keys, input_info)
#     np.testing.assert_allclose(np.asarray(res).reshape(out_info["y"].shape), out_info["y"].astype(res[0].dtype))
#
# @pytest.mark.parametrize('data_shape, kernel_shape, stride', [
#     ((1, 6, 28, 28), (2, 2), 2),
# ])
# def test_avg_pool(data_shape, kernel_shape, stride):
#     data = np.random.randint(0, 5, data_shape)
#     tout = pooling(data, kernel_shape[0], kernel_shape[1], stride=stride)
#
#     out = pm.output(name="out")
#     n = pm.parameter("ns")
#     ic = pm.parameter("ic")
#     ih = pm.parameter("ih")
#     iw = pm.parameter("iw")
#     kh = pm.parameter("kh")
#     kw = pm.parameter("kw")
#     x = pm.input(name="data", shape=(n, ic, ih, iw))
#
#     g = pm.avg_pool2d(x, out, kh, kw, stride=stride, pad=0)
#     inp_info = {}
#     inp_info["data"] = data
#     inp_info["kh"] = kernel_shape[0]
#     inp_info["kw"] = kernel_shape[1]
#     test_out = g("out", inp_info)
#     np.testing.assert_allclose(test_out, tout)
#
# @pytest.mark.parametrize('x_shape, w_shape, params', [
#     ((1, 1, 32, 32), (6, 1, 5, 5), {"stride": 1, "pad": 0}),
#     ((1, 1, 4, 4), (2, 1, 2, 2), {"stride": 2, "pad": 1}),
#     ((1, 1, 32, 32), (2, 1, 4, 4), {"stride": 2, "pad": 1}),
# ])
# def test_translate_conv(x_shape, w_shape, params):
#     shape_dict = {"n": x_shape[0], "c": x_shape[1], "ih": x_shape[2], "iw": x_shape[3],
#                   "nf": w_shape[0], "kh": w_shape[2], "kw": w_shape[3],
#                   "stride": params["stride"], "pad": params["pad"]}
#
#     _, input_info, out_info, keys = conv(x_shape, w_shape, params, coarse=True, debug_matrix=False)
#
#     n = pm.parameter(name="n")
#     c = pm.parameter(name="ic")
#     ih = pm.parameter(name="ih")
#     iw = pm.parameter(name="iw")
#     nf = pm.parameter(name="nf")
#     kh = pm.parameter(name="kh")
#     kw = pm.parameter(name="kw")
#     x = pm.input(name="data", shape=(n, c, ih, iw))
#     w = pm.state(name="w", shape=(nf, c, kh, kw))
#     b = pm.state(name="bias", shape=(nf))
#     stride = pm.parameter(name="stride")
#     pad = pm.parameter(name="pad")
#     out = pm.output(name="out")
#     graph = pm.conv_bias(x, w, b, out, stride, pad)
#     tinput_info = copy.deepcopy(input_info)
#     res0 = graph("out", tinput_info)
#
#     np.testing.assert_allclose(res0, out_info["out"])
#
# @pytest.mark.parametrize('x_shape', [
#     (3, 3, 4, 4),
# ])
# def test_translate_flatten(x_shape):
#     x = np.random.randint(0, 5, x_shape)
#     data = pm.input("x", shape=x.shape)
#     out = pm.output("out")
#
#     g = pm.batch_flatten(data, out)
#
#     res = g("out", {"x": x})
#     print(res)
#     print(x.reshape(-1))
#     np.testing.assert_allclose(res, x.reshape(-1))
#
# @pytest.mark.parametrize('x_shape', [
#     (10,),
# ])
# def test_translate_reduce_sum(x_shape):
#     data = np.random.randint(-3, 3, x_shape)
#     np_res = np.sum(data)
#     graph = pm.Node("reduce")
#     pm_data = pm.input(name="a", shape=x_shape, graph=graph)
#     out = pm.output(name="out", graph=graph)
#     axis = (0,)
#     keepdims = 0
#
#     with graph:
#         pm.reduce_sum(pm_data, out, axes=axis, keepdims=keepdims)
#     pm_res = graph("out", {"a": data})
#     np.testing.assert_allclose(pm_res, np_res)
#
#
# @pytest.mark.parametrize('x_shape', [
#     (10,),
# ])
# def test_translate_elem_mul(x_shape):
#     a = np.random.randint(-3, 3, x_shape)
#     b = np.random.randint(-3, 3, x_shape)
#     np_res = a * b
#     graph = pm.Node("elem_mul")
#
#     pm_a = pm.input(name="a", shape=x_shape, graph=graph)
#     pm_b = pm.input(name="b", shape=x_shape, graph=graph)
#     pm_o = pm.output(name="out", shape=x_shape, graph=graph)
#     with graph:
#         pm.elem_mul(pm_a, pm_b, pm_o)
#     pm_res = graph("out", {"a": a, "b": b})
#     np.testing.assert_allclose(pm_res, np_res)
#
#
# @pytest.mark.parametrize('x_shape', [
#     (10,),
# ])
# def test_translate_vmul(x_shape):
#     a = np.random.randint(-3, 3, x_shape)
#     b = np.random.randint(-3, 3, x_shape)
#     np_res = a.dot(b)
#     with pm.Node("vmul") as pm_graph:
#         pm_a = pm.input(name="a", shape=x_shape)
#         pm_b = pm.input(name="b", shape=x_shape)
#         pm_o = pm.output(name="o", shape=x_shape)
#         pm_s = pm.output(name="out")
#         pm.elem_mul(pm_a, pm_b, pm_o)
#         pm.reduce_sum(pm_o, pm_s, axes=(0,), keepdims=0)
#
#     pm_res = pm_graph("out", {"a": a, "b": b})
#     np.testing.assert_allclose(pm_res, np_res)
#
# @pytest.mark.parametrize('x_shape, axis', [
#     ((1, 1024,), 1),
# ])
# def test_translate_softmax(x_shape, axis):
#     x = np.random.randint(0, 5, x_shape).astype(np.float)
#     data = pm.input("x", shape=x.shape)
#     out = pm.output("out")
#     g = pm.softmax(data, out, axis=1)
#     res = g("out", {"x": x})
#     np_res = np_softmax(x, axis=1)
#     np.testing.assert_allclose(np_res, res)
#
#
# @pytest.mark.parametrize('layer_name, param_dict, data_func, input_keys, output_key',[
#     ("conv", {'m': 54}, conv, {"y":"y:0", "x":"x:0", "w":"W:0"}, [("w", "W:0")]),
# ])
# def test_translate_layers(layer_name, param_dict, data_func, input_keys, output_key):
#     filename = f"full_dnns/tiny_yolo.onnx"
#     filepath = f"{BENCH_DIR}/{filename}"
#     assert Path(filepath).exists()
#     graph = pm.from_pytorch(filepath)
#
#
# @pytest.mark.parametrize('x_shape',[
#     ((3,2,4,3)),
# ])
# def test_batchnorm(x_shape):
#     graph, inp_info, out_info, keys = batchnorm(x_shape, coarse=True)
#     test_out = graph(keys[0], inp_info)
#     np.testing.assert_allclose(out_info[keys[0]], test_out)
#
#
# @pytest.mark.parametrize('x_shape',[
#     ((2,2, 3,3)),
# ])
# def test_global_avg_pool(x_shape):
#     graph, inp_info, out_info, keys = global_avg_pool(x_shape, coarse=True)
#     test_out = graph(keys[0], inp_info)
#     np.testing.assert_allclose(out_info[keys[0]], test_out)
#
# @pytest.mark.parametrize('x_shape, alpha, beta, bias, nsize',[
#     ((3,2,3,3), 0.0002, 0.5, 2.0, 2),
# ])
# def test_lrn(x_shape, alpha, beta, bias, nsize):
#     graph, inp_info, out_info, keys = lrn(x_shape, alpha, beta, bias, nsize, coarse=True)
#     test_out = graph(keys[0], inp_info)
#     np.testing.assert_allclose(out_info[keys[0]], test_out)
#
# def test_lenet():
#     filename = f"lenet.onnx"
#     full_path = f"{BENCH_DIR}/full_dnns"
#
#     filepath = f"{full_path}/{filename}"
#     pb_path = f"{full_path}/lenet.srdfg"
#
#     assert Path(filepath).exists()
#     graph = pm.from_pytorch(filepath)
#     pm.pb_store(graph, full_path)
#     node = pm.pb_load(pb_path, verbose=True)
#     assert len(node.nodes) == len(graph.nodes)
#     for name, n in node.nodes.items():
#         if n.op_name == "conv":
#             print(n.kwargs.keys())
#             break
#
#
# def test_resnet18():
#     filename = f"resnet18.onnx"
#     filepath = f"{BENCH_DIR}/full_dnns/{filename}"
#     assert Path(filepath).exists()
#     graph = pm.from_pytorch(filepath)
#     full_path = f"{BENCH_DIR}/full_dnns"
#     pb_path = f"{full_path}/resnet18.srdfg"
#     pm.pb_store(graph, full_path)
#
#     node = pm.pb_load(pb_path, verbose=True)
#     assert len(node.nodes) == len(graph.nodes)
#
# def test_resnet18_train():
#     filename = f"resnet18_train.onnx"
#     filepath = f"{BENCH_DIR}/full_dnns/{filename}"
#     assert Path(filepath).exists()
#     graph = pm.from_pytorch(filepath)
#     full_path = f"{BENCH_DIR}/full_dnns"
#     pb_path = f"{full_path}/resnet18_train.srdfg"
#     pm.pb_store(graph, full_path)
#
#     node = pm.pb_load(pb_path, verbose=True)
#     assert len(node.nodes) == len(graph.nodes)
#
#
# def test_maskrcnn():
#     MRCNN_PATH = f"{BENCH_DIR}/full_dnns/mask_rcnn/builtin"
#     filenames = [f"backbone_mrcnn_builtin_opt.onnx", f"rpn_mrcnn_builtin_opt.onnx"]
#     for f in filenames:
#         filepath = f"{MRCNN_PATH}/{f}"
#         assert Path(filepath).exists()
#         graph = pm.from_pytorch(filepath)
#
#
#
#
#
#
#
#
#
#
