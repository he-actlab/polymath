import polymath as pm
from .template_utils import _get_indices, _get_single_node_indices
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools


class conv_grad_input(pm.Template):
    def define_graph(self, weight, dout, bias, out, stride=1, pad=0):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)


class conv_grad_weight(pm.Template):
    def define_graph(self, weight, dout, bias, out, stride=1, pad=0):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)