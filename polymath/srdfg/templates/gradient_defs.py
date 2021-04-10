import polymath as pm
from .template_utils import _get_indices, _get_single_node_indices, _get_elem_indices
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools
OPTIMIZERS = {'sgd': pm.sgd}
LOSS_FUNCS = {'cross_entropy': pm.cross_entropy_loss}

class batchnorm_grad(pm.Template):
    def define_graph(self, x, scale, b, mean, var, grad, x_grad,
                     scale_grad, b_grad, optimizer, optimizer_kwargs, eps=1e-5):
        indices = _get_single_node_indices(x, shape=x.shape)
        reduce_idx = (indices[0], indices[2], indices[3])
        N = np.prod((x.shape[0], x.shape[2], x.shape[3]))
        sum_grad = pm.sum([reduce_idx], grad[indices])
        mean_grad_y = sum_grad / N
        mean_x = pm.sum([reduce_idx], x[indices]) / N
        sqr_err = (x[indices] - mean_x[indices[1]])**2
        var_x = pm.sum([reduce_idx], sqr_err[indices]) / N
        grad_y_offset = (grad[indices] - mean_grad_y[indices[1]])
        x_offset = x[indices] - mean_x[indices[1]]
        var_eps = var_x[indices[1]] + eps
        offset_sum = pm.sum([reduce_idx], grad[indices]*x_offset[indices])
        new_mean = offset_sum[indices[1]] / N
        rsqrt_var = (pm.rsqrt(var_eps[indices[1]])).set_name(f"{x.name}_rsqrt_var")
        unsq_indices = _get_single_node_indices(rsqrt_var, shape=(1, x.shape[1], 1, 1))
        coeff = (scale[unsq_indices[1]] * rsqrt_var[unsq_indices])
        grad_sub = ((x_offset[indices] * new_mean[indices[1]])/ (var_eps[indices[1]]))
        x_grad[indices] = coeff[indices[1]] * (grad_y_offset[indices] - grad_sub[indices])
        scale_grad[indices[1]] = rsqrt_var[indices[1]] * offset_sum[indices[1]]
        b_grad[indices[1]] = sum_grad[indices[1]]
        OPTIMIZERS[optimizer](scale, scale_grad, **optimizer_kwargs)
        OPTIMIZERS[optimizer](b, b_grad, **optimizer_kwargs)



class global_average_pool_grad(pm.Template):
    def define_graph(self, data, grad, data_grad):
        pass

class max_pool_grad(pm.Template):
    def define_graph(self, data, grad, data_grad, kh, kw, stride=(1, 1), pad=(0,0)):
        pass

class flatten_grad(pm.Template):
    def define_graph(self, inp, grad, inp_grad):
        inp_grad.set_shape(inp.shape)

class elem_add_grad(pm.Template):
    def define_graph(self, a, b, grad, a_grad, b_grad):
        a_idx, grad_idx, indices = _get_elem_indices(a, grad, a_grad)
        a_grad[indices] = a[a_idx] + grad[grad_idx]
        b_grad[indices] = b[a_idx] + grad[grad_idx]

class relu_grad(pm.Template):
    def define_graph(self, x, grad, x_grad):
        x_idx, grad_idx, x_grad_idx = _get_elem_indices(x, grad, x_grad)
        x_grad[x_grad_idx] = grad[grad_idx] * (x[x_idx] >= 0)

class conv_grad_no_bias(pm.Template):
    def define_graph(self, inp, weight, grad, inp_grad, weight_grad, optimizer, optimizer_kwargs):
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


class conv_grad(pm.Template):
    def define_graph(self, inp, weight, bias, grad, inp_grad, weight_grad,
                     bias_grad, optimizer, optimizer_kwargs,
                     stride=1, pad=0):

        pm.conv_transpose_bias(grad, weight, bias, inp_grad, stride=stride, pad=pad, out_pad=0)
        pm.conv_bias(inp, grad, bias, weight_grad, stride=stride, pad=pad)
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


class gemm_grad_no_bias(pm.Template):
    def define_graph(self, inp, weight, grad, inp_grad, weight_grad, optimizer, optimizer_kwargs):
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
        # OPTIMIZERS[optimizer](weight, weight_grad)

        pm.reduce_sum(grad, bias_grad)
        OPTIMIZERS[optimizer](bias, bias_grad, **optimizer_kwargs)
        # OPTIMIZERS[optimizer](bias, bias_grad)

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3])

    @property
    def outputs(self):
        return (self.args[4], self.args[5], self.args[6])