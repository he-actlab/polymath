import polymath as pm
from pathlib import Path
import numpy as np
import pytest
import pickle
from collections import defaultdict

CWD = Path(f"{__file__}").parent
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"
from .util import logistic, linear, reco, svm, compare_tabla_dfg, set_shape_and_lower,\
    unwound_fft, backprop, conv, lenet, svm_wifi, svm_wifi_inf, logistic_inf, linear_inf
import pprint

@pytest.mark.parametrize('m_',[
    # 3, 55
    55
])
def test_linear_reg(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = linear(m=m_, coarse=True)


    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval, out_info["w"])
    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)
    # validation_path = f"{CWD}/tabla_examples/{graph.name}_{m_}.json"
    # compare_tabla_dfg(validation_path, tabla_ir, tabla_graph)

@pytest.mark.parametrize('m_',[
    55
])
def test_linear_reg_embedded_values(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = linear(m=m_, coarse=True)
    lgraph, input_info, out_info, keys = linear(m=m_, coarse=False)
    tabla_path = f"{OUTPATH}/{graph.name}_{m_}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

@pytest.mark.parametrize('l1, l2, l3',[
    (8, 16, 1),
    (8, 16, 4)
])
def test_backprop_embedded_values(l1, l2, l3):
    shape_dict = {"l1": l1, "l2": l2 , "l3": l3}
    graph, input_info, out_info, keys = backprop(l1, l2, l3, coarse=True, debug=False)

    test_out = graph(["w1","w2"], input_info)
    np.testing.assert_allclose(test_out[0], out_info["w1"])
    np.testing.assert_allclose(test_out[1], out_info["w2"])

    _, input_info, out_info, keys = backprop(l1, l2, l3, coarse=False, pbar=True)

    tabla_path = f"{OUTPATH}/{graph.name}_{l1}_{l2}_{l3}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info,
                                              add_kwargs=True, debug=True)

@pytest.mark.parametrize('m_',[
    55
])
def test_logreg_reg_embedded_values(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = logistic(m=m_, coarse=True)
    _, input_info, out_info, keys = logistic(m=m_, coarse=False)
    tabla_path = f"{OUTPATH}/{graph.name}_{m_}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)
@pytest.mark.parametrize('m, n, k',[
    (30, 25, 6)
])
def test_reco_embedded_values(m, n, k):
    shape_dict = {"m": m, "n": n, "k": k}
    graph, input_info, out_info, keys = reco(m=m, n=n, k=k, coarse=True)
    ngraph, input_info, out_info, keys = reco(m=m, n=n, k=k, coarse=False)
    tabla_path = f"{OUTPATH}/{graph.name}_{m}_{n}_{k}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)
@pytest.mark.parametrize('m',[
    200
])
def test_svm_embedded_values(m):
    shape_dict = {"m": m}
    graph, input_info, out_info, keys = svm(m=m, coarse=True)
    ngraph, input_info, out_info, keys = svm(m=m, coarse=False)
    tabla_path = f"{OUTPATH}/{graph.name}_{m}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)


@pytest.mark.parametrize('x_shape, w_shape, params', [
    ((1, 1, 16, 16), (3, 1, 3, 3), {"stride": 1, "pad": 0}),
    ((1, 1, 4, 4), (2, 1, 2, 2), {"stride": 2, "pad": 1}),
])
def test_conv_embedded_values(x_shape, w_shape, params):
    shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3],
                  "nf": w_shape[0], "kh": w_shape[2], "kw": w_shape[3],
                  "stride": params["stride"], "pad": params["pad"]}
    graph, input_info0, out_info, keys = conv(x_shape, w_shape, params, coarse=True, debug_matrix=True)
    ngraph, input_info1, out_info, keys = conv(x_shape, w_shape, params, coarse=False, debug_matrix=True)

    lower_pass = pm.Lower({})
    lowered = lower_pass(ngraph)

    res0 = np.asarray(lowered(keys, input_info1)).reshape(out_info["out"].shape)
    np.testing.assert_allclose(res0, out_info["out"])
    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info1, add_kwargs=True, debug=True)

@pytest.mark.parametrize('lr, delta, features, locations, train_size', [
    # (0.0001, 1, 139, 325, 7703),
    (0.0001, 1, 20, 30, 7703),
])
def test_svm_wifi(lr, delta, features, locations, train_size):
    shape_dict = {"n_locations": locations, "n_features": features}

    graph, input_info0, out_info, keys = svm_wifi(features, locations, coarse=True)
    tabla_path = f"{OUTPATH}/{graph.name}_{features}_{locations}_tabla.json"
    # res0 = graph(keys, input_info0)[0]
    # np.testing.assert_allclose(res0, out_info['weights'])

    ngraph, input_info1, out_info, keys = svm_wifi(features, locations, coarse=False)

    # tabla_path = f"{OUTPATH}/{graph.name}_{locations}_{features}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info1, add_kwargs=True)
    srdfg_path = f"{OUTPATH}/"

    pm.pb_store(tabla_graph, srdfg_path)



    # lower_pass = pm.Lower({})
    #
    # lowered = lower_pass(ngraph)
    # res1 = np.asarray(lowered(keys, input_info1)).reshape(out_info["weights"].shape)
    # np.testing.assert_allclose(res1, out_info["weights"])

@pytest.mark.parametrize('lr, delta, features, locations, train_size', [
    (0.0001, 1, 20, 30, 7703),
    # (0.0001, 1, 139, 325, 7703),
])
def test_svm_wifi_inference(lr, delta, features, locations, train_size):
    shape_dict = {"n_locations": locations, "n_features": features}

    graph, input_info0, out_info, keys = svm_wifi_inf(features, locations, coarse=True)
    # tabla_path = f"{OUTPATH}/{graph.name}_{features}_{locations}_tabla.json"
    #
    res0 = graph(keys, input_info0)[0]
    np.testing.assert_allclose(res0, out_info['scores'])
    #
    ngraph, input_info1, out_info, keys = svm_wifi_inf(features, locations, coarse=False)
    #
    # lower_pass = pm.Lower({})
    # lowered = lower_pass(ngraph)
    # res1 = np.asarray(lowered(keys, input_info1)).reshape(out_info["scores"].shape)
    # np.testing.assert_allclose(res1, out_info["scores"])

    tabla_path = f"{OUTPATH}/{graph.name}_{locations}_{features}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info1, add_kwargs=True)
    srdfg_path = f"{OUTPATH}/"

    pm.pb_store(tabla_graph, srdfg_path)



@pytest.mark.parametrize('m',[
    200
])
def test_svm_embedded_values(m):
    shape_dict = {"m": m}
    graph, input_info, out_info, keys = svm(m=m, coarse=True)
    ngraph, input_info, out_info, keys = svm(m=m, coarse=False)

    tabla_path = f"{OUTPATH}/{graph.name}_{m}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

@pytest.mark.parametrize('m_',[
    3, 54
])
def test_svm(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = svm(m=m_, coarse=True)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval, out_info["w"])


    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)
    validation_path = f"{CWD}/tabla_examples/{graph.name}_{m_}.json"
    compare_tabla_dfg(validation_path, tabla_ir, tabla_graph)


@pytest.mark.parametrize('m_',[
    3, 54
])
def test_logistic_reg(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = logistic(m=m_, coarse=True)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval, out_info["w"])

    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)

@pytest.mark.parametrize('m_',[
    # 3, 54
    5
])
def test_linear_reg_inf(m_):
    shape_dict = {"m": m_}

    graph, input_info, out_info, keys = linear_inf(m=m_, coarse=True)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval, out_info[keys])
    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)

@pytest.mark.parametrize('m_',[
    # 3, 54
    5
])
def test_logistic_reg_inf(m_):
    shape_dict = {"m": m_}

    graph, input_info, out_info, keys = logistic_inf(m=m_, coarse=True)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval, out_info[keys])

    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"

    # tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)


@pytest.mark.parametrize('m_, n_,k_', [
    (54, 54, 3),
])
def test_reco_state_write(m_, n_, k_):
    shape_dict = {"m": m_, "n": n_, "k": k_}
    graph, input_info, out_info, keys = reco(m=m_, n=n_, k=k_, coarse=True)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval[0], out_info["w1"])
    np.testing.assert_allclose(coarse_eval[1], out_info["w2"])
    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"
    lowered = set_shape_and_lower(graph, shape_dict)
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)

@pytest.mark.parametrize('m', [
    (4,),
    # (8,),
])
def test_fft(m):

    graph, tinput_info, tout_info, keys = unwound_fft(m, coarse=True)
    lgraph, input_info, out_info, keys = unwound_fft(m, coarse=False)
    input_info = {k: np.int16(v) for k,v in input_info.items()}
    shape_dict = {"N": m[0]}
    # out_real = input_info['x'].dot(input_info['M_real'])**2
    # out_imag = input_info['x'].dot(input_info['M_imag'])**2
    # out_t = np.sqrt(out_imag + out_real)
    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path,
                                              context_dict=input_info,
                                              add_kwargs=True,
                                              debug=False)
    for n in tabla_ir:
        if "computed" in n:
            print(n)




