import polymath as pm
import numpy as np
from polymath.codegen.tvmgen.tvm_pass import TVMPass, TVM_OPS
import json

def generate_tvm(graph, input_dict, filepath, context_dict=None):
    assert len(input_dict) > 0
    shape_dict = {k: v.shape if isinstance(v, np.ndarray) else v for k,v in input_dict.items()}
    shape_dict['populate'] = False
    shape_pass = pm.NormalizeGraph(shape_dict)
    lower_pass = pm.Lower(TVM_OPS)
    tvm_pass = TVMPass()
    shaped = shape_pass(graph)
    lowered = lower_pass(shaped)
    result = tvm_pass(lowered)
    return tvm_pass.tvm_ir['tvm_code']