import polymath as pm
from pathlib import Path
import numpy as np
import pytest
from .util import logistic, linear, reco, svm, compare_tabla_dfg, set_shape_and_lower\
    , fft, unwound_fft, matern32, backprop

@pytest.mark.parametrize('m_',[
    3, 55
])
def test_linear_reg(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = linear(m=m_, coarse=True)
    coarse_eval = graph(keys, input_info)
    assert np.allclose(coarse_eval, out_info["w"])

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)
    validation_path = f"{cwd}/tabla_examples/{graph.name}_{m_}.json"
    compare_tabla_dfg(validation_path, tabla_ir, tabla_graph)

@pytest.mark.parametrize('m_',[
    1000
])
def test_linear_reg_embedded_values(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = linear(m=m_, coarse=True)
    _, input_info, out_info, keys = linear(m=m_, coarse=False)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m_}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

@pytest.mark.parametrize('l1, l2, l3',[
    (32, 16, 32)
])
def test_backprop_embedded_values(l1, l2, l3):
    shape_dict = {"l1": l1, "l2": l2 , "l3": l3}
    graph, input_info, out_info, keys = backprop(l1, l2, l3, coarse=True)

    test_out = graph(["w1","w2"], input_info)

    assert np.allclose(test_out[0], out_info["w1"])
    assert np.allclose(test_out[1], out_info["w2"])

    _, input_info, out_info, keys = backprop(l1, l2, l3, coarse=False)

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{l1}_{l2}_{l3}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                              shape_dict,
                                              tabla_path,
                                              context_dict=input_info, add_kwargs=True)

@pytest.mark.parametrize('m_',[
    55
])
def test_logreg_reg_embedded_values(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = logistic(m_=m_, coarse=True)
    _, input_info, out_info, keys = logistic(m_=m_, coarse=False)
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}_{m_}_tabla.json"

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
    assert np.allclose(coarse_eval, out_info["w"])

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)
    validation_path = f"{cwd}/tabla_examples/{graph.name}_{m_}.json"
    compare_tabla_dfg(validation_path, tabla_ir, tabla_graph)


@pytest.mark.parametrize('m_',[
    3, 54
])
def test_logistic_reg(m_):
    shape_dict = {"m": m_}
    graph, input_info, out_info, keys = logistic(m_=m_, coarse=True)
    coarse_eval = graph(keys, input_info)
    assert np.allclose(coarse_eval, out_info["w"])

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}_tabla.json"

    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)
    # validation_path = f"{cwd}/tabla_examples/{graph.name}_{m_}.json"
    # compare_tabla_dfg(validation_path, tabla_ir, tabla_graph)


@pytest.mark.parametrize('m_, n_,k_', [
    (54, 54, 3),
])
def test_reco_state_write(m_, n_, k_):
    shape_dict = {"m": m_, "n": n_, "k": k_}
    graph, input_info, out_info, keys = reco(m_=m_, n_=n_, k_=k_, coarse=True)
    coarse_eval = graph(keys, input_info)
    assert np.allclose(coarse_eval[0], out_info["w1"])
    assert np.allclose(coarse_eval[1], out_info["w2"])
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    tabla_path = f"{full_path}/{graph.name}_tabla.json"
    lowered = set_shape_and_lower(graph, shape_dict)
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)
    # validation_path = f"{cwd}/tabla_examples/{graph.name}_{k_}.json"
    # compare_tabla_dfg(validation_path, tabla_ir, tabla_graph)

@pytest.mark.parametrize('m', [
    (128),
])
def test_fft(m):
    x = np.random.randint(-5,5, m).astype(np.complex)

    pm_output, np_output = unwound_fft(x)
    assert np.allclose(pm_output, np_output)
    # out_g = g("X", {"x": x})
    # out_g = g("x_rev", {"x": x})


@pytest.mark.parametrize('m', [
    (16),
])
def test_matern32(m):
    X = np.random.uniform(0.0, 1.0, (1, m)).astype(np.float64)
    Y = np.float64(np.random.randint(0, 100))

    mod = matern32(X, Y)
    print(mod)
    # gp_out = k(x)
    #
    # np_out = 1.0 * (1.0 + np.sqrt(3.0)*x) * np.exp(-np.sqrt(3.0)*x)
    # # print(gp_out[1])
    # print(np_out)
    # print(gp_out)
    # print(f"{np.allclose(gp_out, np_out)}")

