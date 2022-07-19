import polymath as pm
from .template_utils import _get_indices, _get_single_node_indices, _get_elem_indices, get_pad_tuple
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
        inp_transposed = pm.temp(name=f"transposed_{inp.name}", shape=(inp.shape[1], inp.shape[0], inp.shape[2], inp.shape[3]))
        grad_transposed = pm.state(name=f"transposed_{grad.name}", shape=(grad.shape[1], grad.shape[0], grad.shape[2], grad.shape[3]))
        wgt_grad_transposed = pm.temp(name=f"transposed_{weight.name}",
                                      shape=(weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3]))
        pm.tensor_transpose(inp, inp_transposed, perm=(1, 0, 2, 3))
        pm.tensor_transpose(grad, grad_transposed, perm=(1, 0, 2, 3))

        pm.conv(inp_transposed, grad_transposed, wgt_grad_transposed, stride=dilation, pad=pad, dilation=stride)
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
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

AUTODIFF_OPS =  ['cross_entropy_loss_grad', 'sgd', 'relu_grad', 'max_pool_grad', 'elem_tanh_grad',
                     'global_average_pool_grad', 'elem_add_grad', 'flatten_grad', 'batchnorm_grad',
                 'average_pool_grad']