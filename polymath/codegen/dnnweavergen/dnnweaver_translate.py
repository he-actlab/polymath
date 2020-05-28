import polymath as pm
import numpy as np
from polymath.codegen.dnnweavergen.dnnweaver_pass import DNNWeaverPass, DNNWEAVER_OPS
import json

def generate_dnnweaver(graph, input_dict, filepath, context_dict=None):
    assert len(input_dict) > 0
    shape_dict = {k: v.shape if isinstance(v, np.ndarray) else v for k,v in input_dict.items()}
    shape_dict['populate'] = False
    shape_pass = pm.NormalizeGraph(shape_dict)
    lower_pass = pm.Lower(DNNWEAVER_OPS)
    dnnw_pass = DNNWeaverPass()
    shaped = shape_pass(graph)
    lowered = lower_pass(shaped)
    result = dnnw_pass(lowered)
    return dnnw_pass.dnnw_ir['dnnweaver_code']