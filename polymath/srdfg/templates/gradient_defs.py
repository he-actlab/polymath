import polymath as pm
from .template_utils import _get_indices, _get_single_node_indices, _get_elem_indices, get_pad_tuple
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
import functools
OPTIMIZERS = {'sgd': pm.sgd}
LOSS_FUNCS = {'cross_entropy': pm.cross_entropy_loss}

class batchnorm_grad_x_mu(pm.Template):
    def define_graph(self, x, mean, x_mu):
        indices = _get_single_node_indices(x, shape=x.shape)
        x_mu[indices] = x[indices] - mean[indices[1]]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class batchnorm_grad_inv_std(pm.Template):

    def define_graph(self, var, inv_std):
        indices = _get_single_node_indices(var, shape=var.shape)
        inv_std[indices] = 1 / pm.sqrt(var[indices])
        return inv_std

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class batchnorm_grad_xhat(pm.Template):

    def define_graph(self, x_mu, inv_std, x_hat):
        indices = _get_single_node_indices(x_mu, shape=x_mu.shape)
        x_hat[indices] = x_mu[indices] * inv_std[indices[1]]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class batchnorm_grad_dx_rhs(pm.Template):

    def define_graph(self, gy, scaled_gy, dx_rhs):
        indices = _get_single_node_indices(gy, shape=gy.shape)
        dx_rhs[indices] = gy[indices] - scaled_gy[indices]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class batchnorm_grad_gamma_inv_std(pm.Template):

    def define_graph(self, gamma, inv_std, gam_mul_inv_std):
        indices = _get_single_node_indices(inv_std, shape=inv_std.shape)
        gam_mul_inv_std[indices] = gamma[indices] * inv_std[indices]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class batchnorm_grad_scaled_gy(pm.Template):

    def define_graph(self, dg_mul_xhat, dbeta, scaled_gy):
        inv_m = 1/(dg_mul_xhat.shape[0]*dg_mul_xhat.shape[2]*dg_mul_xhat.shape[3])
        indices = _get_single_node_indices(dg_mul_xhat, shape=dg_mul_xhat.shape)
        scaled_gy[indices] = (dg_mul_xhat[indices] + dbeta[indices[1]]) * inv_m

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class batchnorm_grad_dx(pm.Template):

    def define_graph(self, dx_rhs, g_mul_istd, dx):
        indices = _get_single_node_indices(dx_rhs, shape=dx_rhs.shape)
        dx[indices] = g_mul_istd[indices[1]] * dx_rhs[indices]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class batchnorm_grad_dgamma_mul_xhat(pm.Template):

    def define_graph(self, x_hat, dgamma, dg_mul_xhat):
        indices = _get_single_node_indices(x_hat, shape=x_hat.shape)
        dg_mul_xhat[indices] = dgamma[indices[1]] * x_hat[indices]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class batchnorm_grad_dgamma_xhat(pm.Template):

    def define_graph(self, grad, x_hat, dgamma, dg_mul_xhat):
        indices = _get_single_node_indices(grad, shape=grad.shape)
        reduce_idx = (indices[0], indices[2], indices[3])
        dgamma[indices[1]] = pm.sum([reduce_idx], grad[indices] * x_hat[indices])
        dg_mul_xhat[indices] = dgamma[indices[1]] * x_hat[indices]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2], self.args[3])

class batchnorm_grad_dgamma(pm.Template):

    def define_graph(self, grad, x_hat, dgamma):
        indices = _get_single_node_indices(grad, shape=grad.shape)
        reduce_idx = (indices[0], indices[2], indices[3])
        dgamma[indices[1]] = pm.sum([reduce_idx], grad[indices] * x_hat[indices])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class batchnorm_grad_dbeta(pm.Template):

    def define_graph(self, grad, dbeta):
        indices = _get_single_node_indices(grad, shape=grad.shape)
        reduce_idx = (indices[0], indices[2], indices[3])
        dbeta[indices[1]] = pm.sum([reduce_idx], grad[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class batchnorm_grad(pm.Template):
    def define_graph(self, x, scale, b, mean, var, grad, x_grad,
                     scale_grad, b_grad, optimizer, optimizer_kwargs, eps=1e-5):
        x_mu = pm.temp(name=f"{grad.name}_x_mu", shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        inv_std = pm.temp(name=f"{grad.name}_inv_std", shape=(x.shape[1],))
        x_hat = pm.temp(name=f"{grad.name}_x_hat", shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

        batchnorm_grad_x_mu(x, mean, x_mu)
        batchnorm_grad_inv_std(var, inv_std)
        batchnorm_grad_xhat(x_mu, inv_std, x_hat)
        batchnorm_grad_dbeta(grad, b_grad)

        dg_mul_xhat = pm.temp(name=f"{grad.name}_dg_mul_xhat", shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        gam_mul_inv_std = pm.temp(name=f"{grad.name}_gam_mul_inv_std", shape=(x.shape[1],))
        scaled_gy = pm.temp(name=f"{grad.name}_scaled_gy", shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        dx_rhs = pm.temp(name=f"{grad.name}_dx_rhs", shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

        # OPTION 1:
        # batchnorm_grad_dgamma_xhat(grad, x_hat, scale_grad, dg_mul_xhat)
        # OPTION 2:
        batchnorm_grad_dgamma(grad, x_hat, scale_grad)
        batchnorm_grad_dgamma_mul_xhat(x_hat, scale_grad, dg_mul_xhat)
        ##

        batchnorm_grad_gamma_inv_std(scale, inv_std, gam_mul_inv_std)
        batchnorm_grad_scaled_gy(dg_mul_xhat, b_grad, scaled_gy)
        batchnorm_grad_dx_rhs(grad, scaled_gy, dx_rhs)
        batchnorm_grad_dx(dx_rhs, gam_mul_inv_std, x_grad)

        with self.graph:
            OPTIMIZERS[optimizer](scale, scale_grad, **optimizer_kwargs)
            OPTIMIZERS[optimizer](b, b_grad, **optimizer_kwargs)

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4], self.args[5])

    @property
    def outputs(self):
        return (self.args[6], self.args[7], self.args[8])

class batchnorm_grad__(pm.Template):
    def define_graph(self, x, scale, b, mean, var, grad, x_grad,
                     scale_grad, b_grad, optimizer, optimizer_kwargs, eps=1e-5):
        indices = _get_single_node_indices(x, shape=x.shape)
        reduce_idx = (indices[0], indices[2], indices[3])

        x_mu = self.x_mu(x, mean, indices)
        inv_std = self.inv_std(var, indices)
        x_hat = self.xhat(x_mu, inv_std, indices)

        self.dbeta(b_grad, grad, reduce_idx, indices)
        # self.dgamma(scale_grad, grad, x_hat, reduce_idx, indices)
        inv_m = 1/(x.shape[0]*x.shape[2]*x.shape[3])
        dg_mul_xhat = self.dgamma_xhat(scale_grad, grad, x_hat, reduce_idx, indices)
        gam_mul_inv_std = self.gamma_mul_inv_std(scale, inv_std, indices)
        scaled_gy = self.scaled_gy(dg_mul_xhat, b_grad, inv_m, indices)
        dx_rhs = self.dx_rhs(grad, scaled_gy, indices)
        self.dx(gam_mul_inv_std, dx_rhs, x_grad, indices)
        with self.graph:
            OPTIMIZERS[optimizer](scale, scale_grad, **optimizer_kwargs)
            OPTIMIZERS[optimizer](b, b_grad, **optimizer_kwargs)

    def x_mu(self, x, mean, indices):
        x_mu = x[indices] - mean[indices[1]]
        return x_mu

    def inv_std(self, var, indices):
        inv_std = 1/pm.sqrt(var[indices[1]])
        return inv_std

    def xhat(self, x_mu, inv_std, indices):
        x_hat = x_mu[indices] * inv_std[indices[1]]
        return x_hat

    def dx_rhs(self, gy, scaled_gy, indices):
        dx_rhs = gy[indices] - scaled_gy[indices]
        return dx_rhs

    def gamma_mul_inv_std(self, gamma, inv_std, indices):
        gam_mul_inv_std = gamma[indices[1]]*inv_std[indices[1]]
        return gam_mul_inv_std

    def scaled_gy(self, dg_mul_xhat, dbeta, inv_m, indices):
        scaled_gy = (dg_mul_xhat[indices] + dbeta[indices[1]]) * inv_m
        return scaled_gy

    def dx(self, g_mul_istd, dx_rhs, dx, indices):
        dx[indices] = g_mul_istd[indices[1]] * dx_rhs[indices]

    def dgamma_xhat(self, dgamma, grad, x_hat, reduce_idx, indices):
        dgamma[indices[1]] = pm.sum([reduce_idx], grad[indices]*x_hat[indices])
        dg_mul_xhat = dgamma[indices[1]]*x_hat[indices]
        return dg_mul_xhat

    def dgamma(self, dgamma, grad, x_hat, reduce_idx, indices):
        dgamma[indices[1]] = pm.sum([reduce_idx], grad[indices]*x_hat[indices])

    def dbeta(self, dbeta, grad, reduce_idx, indices):
        dbeta[indices[1]] = pm.sum([reduce_idx], grad[indices])


class batchnorm_grad_(pm.Template):
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

        with self.graph:
            OPTIMIZERS[optimizer](scale, scale_grad, **optimizer_kwargs)
            OPTIMIZERS[optimizer](b, b_grad, **optimizer_kwargs)

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4], self.args[5])

    @property
    def outputs(self):
        return (self.args[6], self.args[7], self.args[8])

class global_average_pool_grad(pm.Template):
    def define_graph(self, data, grad, data_grad):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class max_pool_grad(pm.Template):
    def define_graph(self, data, grad, data_grad, kh, kw, stride=(1, 1), pad=(0,0)):
        data_grad.set_shape(data.shape)

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def kernel_size(self):
        return (self.args[3], self.args[4])

    @property
    def pad(self):
        return self.kwargs['pad']

class average_pool_grad(pm.Template):
    def define_graph(self, data, grad, data_grad, kh, kw, stride=(1, 1), pad=(0,0)):
        data_grad.set_shape(data.shape)

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def kernel_size(self):
        return (self.args[3], self.args[4])

    @property
    def pad(self):
        return self.kwargs['pad']

class flatten_grad(pm.Template):
    def define_graph(self, inp, grad, inp_grad):
        inp_grad.set_shape(inp.shape)

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_add_grad(pm.Template):
    def define_graph(self, a, b, grad, a_grad, b_grad):
        a_grad.set_shape(grad.shape)
        b_grad.set_shape(grad.shape)
        # a_idx, grad_idx, indices = _get_elem_indices(a, grad, a_grad)
        # pm.elem_add(a, grad, a_grad)
        # pm.elem_add(b, grad, b_grad)
        # a_grad[indices] = a[a_idx] + grad[grad_idx]
        # b_grad[indices] = b[a_idx] + grad[grad_idx]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3], self.args[4])

class relu_grad(pm.Template):
    def define_graph(self, x, grad, x_grad):
        assert x.shape == grad.shape and grad.shape == x_grad.shape, f"Gradient shape does not match input shape:\n" \
                                                                     f"Input shape: {x.shape}\n" \
                                                                     f"Gradient shape: {grad.shape}\n" \
                                                                     f"Output shape: {x_grad.shape}"
        x_idx, grad_idx, x_grad_idx = _get_elem_indices(x, grad, x_grad)
        x_grad[x_grad_idx] = grad[grad_idx] * (x[x_idx] >= 0)

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class elem_tanh_grad(pm.Template):
    def define_graph(self, x, grad, x_grad):

        x_idx, grad_idx, x_grad_idx = _get_elem_indices(x, grad, x_grad)
        # # x_grad[x_grad_idx] = grad[grad_idx] * (1 - pm.square(pm.tanh(x[x_idx])))
        # x_grad[x_grad_idx] = grad[grad_idx] * (1 - pm.tanh(x[x_idx])))

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class conv_grad_no_bias(pm.Template):
    def define_graph(self, inp, weight, grad, inp_grad, weight_grad, optimizer, optimizer_kwargs,
                     stride=1, pad=0, dilation=1):
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)
        batch, in_channel, in_height, in_width = inp.shape
        num_filter, channel, kernel_h, kernel_w = weight.shape
        _, _, grad_h, grad_w = grad.shape

        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad

        min_sizes = []
        k = len(grad.shape) - 2

        for d in range(k):
            min_sizes.append(
                (grad.shape[d + 2] - 1) * stride_w
                - 2 * pad[0]
                + (weight.shape[-1] - 1) * dilation_w
                + 1
            )
        grad_input_padding = tuple(inp.shape[-k + d] - min_sizes[d] for d in range(k))
        assert grad_input_padding[0] == grad_input_padding[1]
        pm.conv_transpose(grad, weight, inp_grad, stride=stride, pad=pad, out_pad=grad_input_padding[0])


        inp_transposed = pm.temp(name=f"transposed_{inp.name}", shape=(inp.shape[1], inp.shape[0], inp.shape[2], inp.shape[3]))
        grad_transposed = pm.state(name=f"transposed_{grad.name}", shape=(grad.shape[1], grad.shape[0], grad.shape[2], grad.shape[3]))
        pm.tensor_transpose(inp, inp_transposed, perm=(1, 0, 2, 3))
        pm.tensor_transpose(grad, grad_transposed, perm=(1, 0, 2, 3))

        wgt_grad_oh = int((inp_transposed.shape[2] + pad_top + pad_down - stride * (
                    grad_transposed.shape[2] - 1) - 1) / dilation_h + 1)
        wgt_grad_ow = int((inp_transposed.shape[3] + pad_left + pad_right - stride * (
                    grad_transposed.shape[3] - 1) - 1) / dilation_w + 1)
        wgt_grad_shape = (weight.shape[1], weight.shape[0], wgt_grad_oh, wgt_grad_ow)

        wgt_grad_transposed = pm.temp(name=f"transposed_{weight.name}", shape=wgt_grad_shape)

        pm.conv(inp_transposed, grad_transposed, wgt_grad_transposed, stride=dilation, pad=pad, dilation=stride)
        ic = pm.index(0, weight.shape[0] - 1)
        oc = pm.index(0, weight.shape[1] - 1)
        kh = pm.index(0, weight.shape[2] - 1)
        kw = pm.index(0, weight.shape[3] - 1)
        wgt_grad_transposed_unpadded = pm.temp(name=f"transposed_{weight.name}_unpadded", shape=(weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]))
        wgt_grad_transposed_unpadded[ic, oc, kh, kw] = wgt_grad_transposed[ic, oc, kh, kw]

        pm.tensor_transpose(wgt_grad_transposed, weight_grad, perm=(1, 0, 2, 3))
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

        ## test values
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)
        batch, in_channel, in_height, in_width = inp.shape
        num_filter, channel, kernel_h, kernel_w = weight.shape
        _, _, grad_h, grad_w = grad.shape

        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad


        min_sizes = []
        k = len(grad.shape) - 2

        for d in range(k):
            min_sizes.append(
                (grad.shape[d + 2] - 1) * stride_w
                - 2 * pad[0]
                + (weight.shape[-1] - 1) * dilation_w
                + 1
            )
        grad_input_padding = tuple(inp.shape[-k + d] - min_sizes[d] for d in range(k))
        assert grad_input_padding[0] == grad_input_padding[1]
        pm.conv_transpose_bias(grad, weight, bias, inp_grad, stride=stride, pad=pad, out_pad=grad_input_padding[0])


        inp_transposed = pm.temp(name=f"transposed_{inp.name}", shape=(inp.shape[1], inp.shape[0], inp.shape[2], inp.shape[3]))
        grad_transposed = pm.state(name=f"transposed_{grad.name}", shape=(grad.shape[1], grad.shape[0], grad.shape[2], grad.shape[3]))

        pm.tensor_transpose(inp, inp_transposed, perm=(1, 0, 2, 3))
        pm.tensor_transpose(grad, grad_transposed, perm=(1, 0, 2, 3))
        wgt_grad_oh = int((inp_transposed.shape[2] + pad_top + pad_down - stride*(grad_transposed.shape[2]- 1) - 1) / dilation_h + 1)
        wgt_grad_ow = int((inp_transposed.shape[3] + pad_left + pad_right - stride*(grad_transposed.shape[3]- 1) - 1) / dilation_w + 1)
        wgt_grad_shape = (weight.shape[1], weight.shape[0], wgt_grad_oh, wgt_grad_ow)
        wgt_grad_transposed = pm.temp(name=f"transposed_{weight.name}", shape=wgt_grad_shape)

        n = pm.conv(inp_transposed, grad_transposed, wgt_grad_transposed, stride=dilation, pad=pad, dilation=stride)

        ic = pm.index(0, weight.shape[0] - 1)
        oc = pm.index(0, weight.shape[1] - 1)
        kh = pm.index(0, weight.shape[2] - 1)
        kw = pm.index(0, weight.shape[3] - 1)
        wgt_grad_transposed_unpadded = pm.temp(name=f"transposed_{weight.name}_unpadded", shape=(weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]))
        wgt_grad_transposed_unpadded[ic, oc, kh, kw] = wgt_grad_transposed[ic, oc, kh, kw]

        pm.tensor_transpose(wgt_grad_transposed, weight_grad, perm=(1, 0, 2, 3))
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
        transA = False
        transB = False
        if grad.shape[1] != weight.shape[0]:
            indices = tuple([pm.index(0, s - 1) for s in weight.shape])
            weight_transposed = pm.state(name=f"{weight.name}_transposed", shape=(weight.shape[1], weight.shape[0]))
            weight_transposed[indices[1], indices[0]] = weight[indices]
            # pm.matmul(grad, weight_transposed, inp_grad, transA=transA, transB=transB, strict_shapes=True)
        else:
            pm.gemm_no_bias(grad, weight, inp_grad, transA=transA, transB=transB, strict_shapes=True)

        if grad.shape[0] != inp.shape[1]:
            indices = tuple([pm.index(0, s - 1) for s in inp.shape])
            # inp_transposed = pm.temp(name=f"{inp.name}_transposed", shape=(inp.shape[1], inp.shape[0]))
            inp_transposed = pm.state(name=f"{inp.name}_transposed", shape=(inp.shape[1], inp.shape[0]))
            inp_transposed[indices[1], indices[0]] = inp[indices]
            pm.gemm_no_bias(inp_transposed, grad, weight_grad, transA=transA, transB=transB, strict_shapes=True)
        else:
            pm.gemm_no_bias(inp, grad, weight_grad, transA=transA, transB=transB, strict_shapes=True)

        OPTIMIZERS[optimizer](weight, weight_grad, **optimizer_kwargs)

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3], self.args[4])


class gemm_grad(pm.Template):
    def define_graph(self, inp, weight, bias, grad, inp_grad, weight_grad, bias_grad, optimizer, optimizer_kwargs):
        transA = False
        transB = False

        if grad.shape[1] != weight.shape[0]:
            indices = tuple([pm.index(0, s - 1) for s in weight.shape])
            # weight_transposed = pm.temp(name=f"{weight.name}_transposed", shape=(weight.shape[1], weight.shape[0]))
            weight_transposed = pm.state(name=f"{weight.name}_transposed", shape=(weight.shape[1], weight.shape[0]))
            weight_transposed[indices[1], indices[0]] = weight[indices]
            pm.gemm_no_bias(grad, weight_transposed, inp_grad, transA=transA, transB=transB, strict_shapes=True)
        else:
            pm.gemm_no_bias(grad, weight, inp_grad, transA=transA, transB=transB, strict_shapes=True)

        if grad.shape[0] != inp.shape[1]:
            indices = tuple([pm.index(0, s-1) for s in inp.shape])
            # inp_transposed = pm.temp(name=f"{inp.name}_transposed", shape=(inp.shape[1], inp.shape[0]))
            inp_transposed = pm.state(name=f"{inp.name}_transposed", shape=(inp.shape[1], inp.shape[0]))
            inp_transposed[indices[1], indices[0]] = inp[indices]
            pm.gemm_no_bias(inp_transposed, grad, weight_grad, transA=transA, transB=transB, strict_shapes=True)
        else:
            pm.gemm_no_bias(inp, grad, weight_grad, transA=transA, transB=transB, strict_shapes=True)



        # Weight update
        assert weight_grad.shape == weight.shape

        OPTIMIZERS[optimizer](weight, weight_grad, **optimizer_kwargs)

        pm.reduce_sum(grad, bias_grad)
        OPTIMIZERS[optimizer](bias, bias_grad, **optimizer_kwargs)

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
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[3],)

# AUTODIFF_OPS =  ['cross_entropy_loss_grad', 'sgd', 'relu_grad', 'max_pool_grad', 'elem_tanh_grad',
#                      'global_average_pool_grad', 'elem_add_grad', 'flatten_grad', 'batchnorm_grad',
#                  'average_pool_grad']

AUTODIFF_OPS =  ['cross_entropy_loss_grad', 'sgd', 'relu_grad', 'max_pool_grad', 'elem_tanh_grad',
                     'global_average_pool_grad', 'elem_add_grad', 'flatten_grad', 'average_pool_grad',
                 'batchnorm_grad_x_mu', 'batchnorm_grad_inv_std', 'batchnorm_grad_xhat', 'batchnorm_grad_dbeta',
                 'batchnorm_grad_dgamma_xhat', 'batchnorm_grad_gamma_inv_std', 'batchnorm_grad_scaled_gy',
                 'batchnorm_grad_dx_rhs', 'batchnorm_grad_dx', 'batchnorm_grad_dgamma', 'batchnorm_grad_dgamma_mul_xhat']