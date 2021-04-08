import polymath as pm
from .template_utils import _get_indices, _get_single_node_indices
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools
OPTIMIZERS = {'sgd': pm.sgd}
LOSS_FUNCS = {'cross_entropy': pm.cross_entropy_loss}

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


class gemm_grad_no_bias(pm.Template):
    def define_graph(self, inp, weight, grad, inp_grad, weight_grad, optimizer, **optimizer_kwargs):
        pm.gemm_no_bias(grad, inp, inp_grad, transA=True)
        pm.gemm_no_bias(grad, weight, weight_grad)
        # Weight update
        OPTIMIZERS[optimizer](weight, weight_grad, **optimizer_kwargs)

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3], self.args[4])


class gemm_grad(pm.Template):
    def define_graph(self, inp, weight, bias, grad, inp_grad, weight_grad, bias_grad, optimizer, optimizer_kwargs):
        pm.gemm_no_bias(grad, inp, inp_grad, transA=True)
        pm.gemm_no_bias(grad, weight, weight_grad)
        # Weight update
        OPTIMIZERS[optimizer](weight, weight_grad, **optimizer_kwargs)

        pm.reduce_sum(grad, bias_grad)
        OPTIMIZERS[optimizer](bias, bias_grad, **optimizer_kwargs)

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3])

    @property
    def outputs(self):
        return (self.args[4], self.args[5], self.args[6])