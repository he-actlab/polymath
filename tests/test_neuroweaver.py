import numpy as np
import pytest
from tests.neuroweaver.nn import nn_impl
import polymath as pm
from pathlib import Path

CWD = Path(f"{__file__}").parent
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"

@pytest.mark.parametrize('num, latitude, longitude',[
    (3, 30, 90),
])
def test_nn(num, latitude, longitude):
    graph, inp_info, out_info, keys = nn_impl(num, latitude, longitude, coarse=True)
    shape_dict = {"num": num}

    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path)

    # test_out = graph("max_dist2", inp_info)
    # print()
    # print(test_out)
    # test_out = graph("sqrtz", inp_info)
    # print(test_out.shape)
    # print(graph("sqrtz", inp_info))
    # print(graph("max_idx1", inp_info))
    # print(graph("max_idx9", inp_info))
    # print(test_out)
    # np.testing.assert_allclose(out_info[keys[0]], test_out)
