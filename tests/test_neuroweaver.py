import numpy as np
import pytest
from tests.neuroweaver.nn import nn_impl, nn_impl_
from tests.neuroweaver.hotspot import hotspot_impl
from tests.neuroweaver.pathfinder import pathfinder_impl
from tests.neuroweaver.hotspot3d import impl_3D
import polymath as pm
from pathlib import Path

HOME = Path.home()
CWD = Path(f"{__file__}").parent
TABLA_PATH = f"{HOME}/ACTLab/rtml/project.rtml/tablav2/benchmarks/dfgs/polymath_generated"
BASE_PATH = f"{CWD}/pmlang_examples"
OUTPATH = f"{BASE_PATH}/outputs"

@pytest.mark.parametrize('num, latitude, longitude',[
    (3, 30, 90),
])
def test_nn(num, latitude, longitude):
    # graph, inp_info, out_info, keys = nn_impl(num, latitude, longitude, coarse=True)
    graph, inp_info, out_info, keys = nn_impl_(num, latitude, longitude, coarse=True)
    shape_dict = {"num": num}

    tabla_path = f"{TABLA_PATH}/{graph.name}_{num}.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path, debug=False)

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

@pytest.mark.parametrize('rows, cols',[
    (10, 10),
])
def test_pathfinder(rows, cols):
    graph = pathfinder_impl()
    tabla_path = f"{TABLA_PATH}/{graph.name}_{rows}_{cols}.json"
    shape_dict = {"rows": rows, "cols": cols}
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path, debug=False)

@pytest.mark.parametrize('num, latitude, longitude',[
    (3, 30, 90),
])
def test_hotspot(num, latitude, longitude):
    # graph, inp_info, out_info, keys = nn_impl(num, latitude, longitude, coarse=True)
    graph = hotspot_impl()
    shape_dict = {"row": 12, "col": 12}

    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path, debug=False)


@pytest.mark.parametrize('num, latitude, longitude',[
    (3, 30, 90),
])
def test_hotspot3d(num, latitude, longitude):
    # graph, inp_info, out_info, keys = nn_impl(num, latitude, longitude, coarse=True)
    graph = impl_3D()
    numCols = 4
    numRows = 4
    layers = 4
    shape_dict = {"numRows": numRows, "numCols": numCols, "layers":layers}

    tabla_path = f"{OUTPATH}/{graph.name}_tabla.json"
    tabla_ir, tabla_graph = pm.generate_tabla(graph, shape_dict, tabla_path, debug=False)