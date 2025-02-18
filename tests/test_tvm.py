import polymath as pm
from tests.util import linear, op_counts, logistic, svm, reco, dense, conv,\
    two_layer_dense, lenet, tvm_lenet
from pathlib import Path
# import tvm
import pytest
import pprint
import numpy as np
import copy

import pickle
from onnx import numpy_helper, helper, defs


# TODO: Fix this
def test_lenet():
    import tvm

    graph, inp_info, out_info, key = lenet(coarse=True)
    coarse_cpy = pickle.loads(pickle.dumps(inp_info))
    res = graph(key, coarse_cpy)
    np.testing.assert_allclose(res, out_info[key])
    tvm_code = pm.generate_tvm(graph, inp_info, "")
    pm_mod = tvm.IRModule.from_expr(tvm_code)
    pm_mod = tvm.relay.transform.InferType()(pm_mod)


    net = tvm_lenet()
    mod = tvm.IRModule.from_expr(net)
    mod = tvm.relay.transform.InferType()(mod)

    print(pm_mod)
    print(mod)
