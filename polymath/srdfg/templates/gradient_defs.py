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
    def define_graph(self, inp, weight, grad, inp_grad, weight_grad, optimizer, optimizer_kwargs,
                     stride=1, pad=0, dilation=1):
        min_sizes = []
        k = len(grad.shape) - 2

        for d in range(k):
            min_sizes.append(
                (grad.shape[d + 2] - 1) * stride
                - 2 * pad
                + (weight.shape[-1] - 1) * dilation
                + 1
            )

        grad_input_padding = tuple(inp.shape[-k + d] - min_sizes[d] for d in range(k))
        assert grad_input_padding[0] == grad_input_padding[1]
        pm.conv_transpose(grad, weight, inp_grad, stride=stride, pad=pad, out_pad=grad_input_padding[0])
        inp_indices = tuple(pm.index(0, s - 1) for s in inp.shape)
        grad_indices = tuple(pm.index(0, s - 1) for s in grad.shape)
        weight_indices = tuple(pm.index(0, s - 1) for s in weight.shape)
        inp_transposed = pm.temp("inp_t", shape=(inp.shape[1], inp.shape[0], inp.shape[2], inp.shape[3]))
        grad_transposed = pm.temp("grad_t", shape=(grad.shape[1], grad.shape[0], grad.shape[2], grad.shape[3]))
        wgt_grad_transposed = pm.temp("wgt_grad_t",
                                      shape=(weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3]))

        inp_transposed[inp_indices[1], inp_indices[0], inp_indices[2], inp_indices[3]] = inp[inp_indices]
        grad_transposed[grad_indices[1], grad_indices[0], grad_indices[2], grad_indices[3]] = grad[grad_indices]

        pm.conv(inp_transposed, grad_transposed, wgt_grad_transposed, stride=dilation, pad=pad, dilation=stride)
        weight_grad[weight_indices] = wgt_grad_transposed[
            weight_indices[1], weight_indices[0], weight_indices[2], weight_indices[3]]
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
                     stride=1, pad=0, dilation=1):
        min_sizes = []
        k = len(grad.shape) - 2

        for d in range(k):
            min_sizes.append(
                (grad.shape[d + 2] - 1) * stride
                - 2 * pad
                + (weight.shape[-1] - 1) * dilation
                + 1
            )

        grad_input_padding = tuple(inp.shape[-k + d] - min_sizes[d] for d in range(k))
        assert grad_input_padding[0] == grad_input_padding[1]
        pm.conv_transpose_bias(grad, weight, bias, inp_grad, stride=stride, pad=pad, out_pad=grad_input_padding[0])
        inp_indices = tuple(pm.index(0, s-1) for s in inp.shape)
        grad_indices = tuple(pm.index(0, s-1) for s in grad.shape)
        weight_indices = tuple(pm.index(0, s-1) for s in weight.shape)
        inp_transposed = pm.temp("inp_t", shape=(inp.shape[1], inp.shape[0], inp.shape[2], inp.shape[3]))
        grad_transposed = pm.temp("grad_t", shape=(grad.shape[1], grad.shape[0], grad.shape[2], grad.shape[3]))
        wgt_grad_transposed = pm.temp("wgt_grad_t", shape=(weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3]))

        inp_transposed[inp_indices[1], inp_indices[0], inp_indices[2], inp_indices[3]] = inp[inp_indices]
        grad_transposed[grad_indices[1], grad_indices[0], grad_indices[2], grad_indices[3]] = grad[grad_indices]

        pm.conv(inp_transposed, grad_transposed, wgt_grad_transposed, stride=dilation, pad=pad, dilation=stride)
        weight_grad[weight_indices] = wgt_grad_transposed[weight_indices[1], weight_indices[0], weight_indices[2], weight_indices[3]]
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



class cross_entropy_loss_grad(pm.Template):
    def define_graph(self, z, y, grad, grad_inp, reduction="mean"):
        indices = [pm.index(0, s - 1, name=f"{z.name}[{i}]") for i, s in enumerate(z.shape)]
        grad_inp[indices] = grad * (z[indices] - y[indices[0]]) / z.shape[0]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

AUTODIFF_OPS =  ['cross_entropy_loss_grad', 'sgd', 'relu_grad', 'max_pool_grad',
                     'global_average_pool_grad', 'elem_add_grad', 'flatten_grad', 'batch_norm_grad']