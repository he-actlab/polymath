import polymath as pm
import numpy as np
from polymath.codegen.dnnweavergen.dnnweaver_pass import DNNWeaverPass, DNNWEAVER_OPS
import json

def generate_dnnweaver(graph, input_dict, filepath, debug=False, add_kwargs=False, context_dict=None):
    shape_dict = {k: v.shape if isinstance(v, np.ndarray) else v for k,v in input_dict.items()}
    shape_dict['populate'] = False
    shape_pass = pm.NormalizeGraph(shape_dict, debug=debug)
    lower_pass = pm.Lower(DNNWEAVER_OPS, debug=debug)
    dnnw_pass = DNNWeaverPass(debug=debug)
    shaped = shape_pass(graph)
    lowered = lower_pass(shaped)
    result = dnnw_pass(lowered)
    return dnnw_pass.dnnw_ir['dnnweaver_code'], result