from polymath.srdfg.passes.compiler_passes import NormalizeGraph, Lower
import polymath as pm
import pprint
import numpy as np
from pathlib import Path
from tests.util import count_nodes, linear, reco

def test_linear_reg():
    m_ = 3
    graph, input_info, out_info, keys = linear(m=m_, coarse=True)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval, out_info["w"])



    fgraph, input_info, out_info, keys = linear(m=m_, coarse=False)
    lower_pass = Lower({})
    lowered_graph = lower_pass(fgraph, {})
    all_vals = lowered_graph(keys, input_info)
    out = np.asarray(all_vals).reshape(out_info["w"].shape)

    np.testing.assert_allclose(out, out_info["w"])
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/{graph.name}.srdfg"

    pm.pb_store(lowered_graph, full_path)

    loaded_node = pm.pb_load(pb_path)
    _, input_info, out_info, keys = linear(m=m_, coarse=False)

    loaded_res = loaded_node(keys, input_info)
    out = np.asarray(loaded_res).reshape(out_info["w"].shape)
    np.testing.assert_allclose(out, out_info["w"])

def test_reco():
    m_ = 3
    n_ = 3
    k_ = 2
    shape_dict = {"m": n_, "k": k_, "n": n_}
    graph, input_info, out_info, keys = reco(coarse=True, **shape_dict)
    coarse_eval = graph(keys, input_info)
    np.testing.assert_allclose(coarse_eval[0], out_info["w1"])
    np.testing.assert_allclose(coarse_eval[1], out_info["w2"])


    fgraph, input_info, out_info, keys = reco(coarse=False, **shape_dict)
    lower_pass = Lower({})
    lowered_graph = lower_pass(fgraph, {})
    all_vals = lowered_graph(keys, input_info)
    w1_elems = np.prod(out_info["w1"].shape)
    w2_elems = np.prod(out_info["w2"].shape)
    out1 = np.asarray(list(all_vals[0:w1_elems])).reshape(out_info["w1"].shape)
    out2 = np.asarray(list(all_vals[w1_elems:])).reshape(out_info["w2"].shape)

    np.testing.assert_allclose(out1, out_info["w1"])
    np.testing.assert_allclose(out2, out_info["w2"])
    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/pmlang_examples"
    full_path = f"{base_path}/outputs"
    pb_path = f"{full_path}/{graph.name}.srdfg"

    pm.pb_store(lowered_graph, full_path)

    loaded_node = pm.pb_load(pb_path)
    _, input_info, out_info, keys = reco(coarse=False, **shape_dict)

    loaded_res = loaded_node(keys, input_info)
    lres1 = np.asarray(list(loaded_res[0:w1_elems])).reshape(out_info["w1"].shape)
    lres2 = np.asarray(list(loaded_res[w1_elems:])).reshape(out_info["w2"].shape)
    np.testing.assert_allclose(lres1, out_info["w1"])
    np.testing.assert_allclose(lres2, out_info["w2"])

