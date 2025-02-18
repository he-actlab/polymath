import polymath as pm
from .template_utils import _get_elem_indices
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools

class sgd(pm.Template):
    def define_graph(self, param, grad, lr=0.01, momentum=0.9, weight_decay=0.0, dampening=0.0, nesterov=False):
        data_idx, grad_idx, indices = _get_elem_indices(param, grad, param)

        if momentum != 0:
            param[indices] = param[data_idx] * momentum - lr * grad[grad_idx]
        else:
            param[indices] = param[data_idx] - lr * grad[grad_idx]


    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[0],)
