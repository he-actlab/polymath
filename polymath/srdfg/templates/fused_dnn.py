import polymath as pm
from collections import defaultdict
from .template_utils import _get_indices, _get_single_node_indices, _get_elem_indices, pad_node, \
    _dim_explicit, get_pad_tuple

class sqrt_reciprocal(pm.Template):
    def define_graph(self, data, out):
        pass

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class conv_bias_relu(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            print(pad)
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad

        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
            # out.set_shape()
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

            # out.set_shape((w.shape[0], oh, ow))
        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)
        out.set_shape(conv_out_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]

        # out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]
        conv_out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        out[o_indices + (y, x)] = (0 < conv_out[o_indices + (y, x)]) * conv_out[o_indices + (y, x)]


    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

class conv_bias_add_relu(pm.Template):
    def define_graph(self, data, w, bias, op1, out, stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        add_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_add_out")
        out.set_shape(conv_out_shape)

        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        # out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]
        conv_out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        add_out[o_indices + (y, x)] = conv_out[o_indices + (y, x)] + op1[conv_out[o_indices + (y, x)]]
        out[o_indices + (y, x)] = (0 < add_out[o_indices + (y, x)]) * add_out[o_indices + (y, x)]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3])

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def add_output(self):
        return self.nodes[f"{self.name}_add_out"]

    @property
    def outputs(self):
        return (self.args[4],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

class conv_bias_leaky_relu(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0, dilation=1, groups=1, alpha=1e-2):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
            # out.set_shape()
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

            # out.set_shape((w.shape[0], oh, ow))
        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)
        out.set_shape(conv_out_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]

        conv_out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        out[o_indices + (y, x)] = (0 < conv_out[o_indices + (y, x)]) * conv_out[o_indices + (y, x)] + (0 >= conv_out[o_indices + (y, x)]) * conv_out[o_indices + (y, x)] * alpha


    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def alpha(self):
        return self.kwargs['alpha']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

class conv_bias_add_leaky_relu(pm.Template):
    def define_graph(self, data, w, bias, op1, out, stride=1, pad=0, dilation=1, groups=1, alpha=1e-2):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h * (kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        add_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_add_out")
        out.set_shape(conv_out_shape)

        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]

        # out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]
        conv_out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        add_out[o_indices + (y, x)] = conv_out[o_indices + (y, x)] + op1[conv_out[o_indices + (y, x)]]
        out[o_indices + (y, x)] = (0 < add_out[o_indices + (y, x)]) * add_out[o_indices + (y, x)] + (0 >= add_out[o_indices + (y, x)]) * add_out[o_indices + (y, x)] * alpha


    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3])

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def add_output(self):
        return self.nodes[f"{self.name}_add_out"]

    @property
    def outputs(self):
        return (self.args[4],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def alpha(self):
        return self.kwargs['alpha']

class conv_bias_leaky_relu_add(pm.Template):
    def define_graph(self, data, w, bias, op1, out, stride=1, pad=0, dilation=1, groups=1, alpha=1e-2):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter

        oh = (in_height + pad_top + pad_down - dilation_h * (kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        leaky_relu_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_leaky_relu_out")
        out.set_shape(conv_out_shape)

        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]

        # out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]
        conv_out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        leaky_relu_out[o_indices + (y, x)] = (0 < conv_out[o_indices + (y, x)]) * conv_out[o_indices + (y, x)] + (0 >= conv_out[o_indices + (y, x)]) * conv_out[o_indices + (y, x)] * alpha
        out[o_indices + (y, x)] = leaky_relu_out[o_indices + (y, x)] + op1[conv_out[o_indices + (y, x)]]


    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3])

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def leaky_relu_output(self):
        return self.nodes[f"{self.name}_leaky_relu_out"]

    @property
    def outputs(self):
        return (self.args[4],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def alpha(self):
        return self.kwargs['alpha']

class conv_bias_relu_max_pool(pm.Template):
    def define_graph(self, data, w, bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     kernel_size=None, stride0=(1,1), pad0=(0,0)
                     ):
        if kernel_size is None:
            raise RuntimeError(f"Kernel size is a required parameter with no default value.\n"
                               f"Need to provide value.")
        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=pad, dilation=dilation, groups=groups)
        relu_out = self.define_relu(conv_out, indices)
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            kh, kw = kernel_size
        elif isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 2
            kh, kw = kernel_size[0], kernel_size[1]
        else:
            raise RuntimeError(f"Invalid type for kernel size")
        self.define_max_pool(relu_out, out, kh, kw, stride0, pad0)


    def define_max_pool(self, relu_out, out, kh, kw, stride, pad):
        oh = ((relu_out.shape[-2] + 2 * pad[0] - kh) // stride[0] + 1)
        ow = ((relu_out.shape[-1] + 2 * pad[1] - kw) // stride[1] + 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        m = pm.index(0, kh - 1)
        n = pm.index(0, kw - 1)
        ihp = (relu_out.shape[-2] + pad[0] * 2)
        iwp = relu_out.shape[-1] + pad[1] * 2
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        iy = pm.index(0, relu_out.shape[-2] - 1, name="iy")
        ix = pm.index(0, relu_out.shape[-1] - 1, name="ix")

        if len(relu_out.shape) > 3:
            b = pm.index(0, relu_out.shape[0] - 1, name="b")
            c = pm.index(0, relu_out.shape[1] - 1, name="c")

            o_indices = (b, c)
            p_shape = (relu_out.shape[0], relu_out.shape[1], ihp, iwp)
            out.set_shape((relu_out.shape[0], relu_out.shape[1], oh, ow))

        else:
            c = pm.index(0, relu_out.shape[0] - 1, name="c")
            o_indices = (c,)
            p_shape = (relu_out.shape[0], ihp, iwp)
            out.set_shape((relu_out.shape[0], oh, ow))

        padded = pm.temp(shape=p_shape)
        padded[o_indices, ihp_, iwp_] = 0
        padded[o_indices, iy + pad[0], ix + pad[1]] = relu_out[o_indices, iy, ix]
        out[o_indices, y, x] = pm.max([m, n], padded[o_indices, stride[0] * y + m, stride[1] * x + n])

    def define_relu(self, conv_out, indices):
        relu_out = pm.temp(shape=conv_out.shape, name=f"{self.name}_relu_out")
        relu_out[indices] = (0 < conv_out[indices]) * conv_out[indices]
        return relu_out


    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def stride0(self):
        return self.kwargs['stride0']

    @property
    def kernel_size(self):
        return self.kwargs['kernel_size']

    @property
    def pad0(self):
        return self.kwargs['pad0']

    @property
    def pad_int0(self):
        if isinstance(self.kwargs['pad0'], tuple):
            return self.kwargs['pad0'][0] + self.kwargs['pad0'][2]
        else:
            return self.kwargs['pad0']


    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def relu_output(self):
        return self.nodes[f"{self.name}_relu_out"]

class conv_bias_add_relu_global_avg_pool(pm.Template):
    def define_graph(self, data, w, bias, op1, out,
                     stride=1, pad=0, dilation=1, groups=1
                     ):

        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=pad, dilation=dilation, groups=groups)
        add_out = self.define_add(conv_out, op1, indices)
        relu_out = self.define_relu(add_out, indices)

        self.define_global_avg_pool(relu_out, out, indices)

    def define_add(self, conv_out, op1, indices):
        add_out = pm.temp(shape=conv_out.shape, name=f"{self.name}_add_out")
        add_out[indices] = conv_out[indices] + op1[indices]
        return add_out


    def define_global_avg_pool(self, relu_out, out, indices):
        m = pm.index(0, relu_out.shape[2]-1)
        n = pm.index(0, relu_out.shape[3]-1)
        h = relu_out.shape[2]
        w = relu_out.shape[3]
        # out[indices] = (1/(h*w)) * pm.sum([m, n], relu_out[indices[0], indices[1], m, n])
        out[indices] = pm.sum([m, n], relu_out[indices[0], indices[1], m, n]) * (1/(h*w))


    def define_relu(self, conv_out, indices):
        relu_out = pm.temp(shape=conv_out.shape, name=f"{self.name}_relu_out")
        relu_out[indices] = (0 < conv_out[indices]) * conv_out[indices]
        return relu_out


    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def add_output(self):
        return self.nodes[f"{self.name}_add_out"]

    @property
    def relu_output(self):
        return self.nodes[f"{self.name}_relu_out"]

class conv_bias_add(pm.Template):
    def define_graph(self, data, w, bias, op1, out, stride=1, pad=0, dilation=1, groups=1):
        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=pad, dilation=dilation, groups=groups)
        self.define_add(conv_out, op1, out, indices)

    def define_add(self, conv_out, op1, out, indices):
        out[indices] = conv_out[indices] + op1[indices]

    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3])

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def outputs(self):
        return (self.args[4],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def groups(self):
        return self.kwargs['groups']

class conv_bias_clip(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0, dilation=1, groups=1, minval=None, maxval=None):
        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=pad, dilation=dilation, groups=groups)
        self.define_clip(conv_out, out, indices, minval, maxval)

    def define_clip(self, conv_out, out, indices, minval, maxval):
        out[indices] = pm.clip(minval, maxval, conv_out[indices])

    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2],)

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def outputs(self):
        return (self.args[3],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']

class depthwise_conv_bias_clip(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0, dilation=1, groups=1, minval=None, maxval=None):
        conv_out, indices = self.define_depthwise_conv(data, w, bias, stride=stride, pad=pad, dilation=dilation, groups=groups)
        self.define_clip(conv_out, out, indices, minval, maxval)

    def define_clip(self, conv_out, out, indices, minval, maxval):
        out[indices] = pm.clip(minval, maxval, conv_out[indices])

    def define_depthwise_conv(self, clip_out, w, bias, stride, pad, groups, dilation, out=None):

        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = clip_out.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h * (kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1, name="y_")
        x = pm.index(0, ow - 1, name="x_")
        dy = pm.index(0, w.shape[2] - 1, name="dy")
        dx = pm.index(0, w.shape[3] - 1, name="dx")
        iy = pm.index(0, clip_out.shape[-2] - 1, name="iy")
        ix = pm.index(0, clip_out.shape[-1] - 1, name="ix")
        k = pm.index(0, clip_out.shape[-3] - 1, name="k")
        ihp = clip_out.shape[-2] + pad_top + pad_down
        iwp = clip_out.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        if len(clip_out.shape) > 3:
            b = pm.index(0, clip_out.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (clip_out.shape[0], clip_out.shape[1], ihp, iwp)
            out_shape = (clip_out.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (clip_out.shape[0], ihp, iwp)
            out_shape = (w.shape[0], oh, ow)

        if out is None:
            out = pm.temp(shape=out_shape, name=f"{self.name}_depthwise_conv_out")

        padded = pm.temp(shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0

        padded[p_indices + (iy + pad_top, ix + pad_left)] = clip_out[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        out[out_indices] = pm.sum([dy, dx, k], (
                padded[p_indices + (dy * dilation_h + stride * y, dx * dilation_w + stride * x)] * w[
            c, k, dy, dx])) + bias[c]
        return out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def outputs(self):
        return (self.args[3],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']

class conv_bias_clip_avg_pool(pm.Template):
    def define_graph(self, data, w, bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None,
                     kernel_size=None, stride0=(1,1), pad0=(0,0)
                     ):
        if kernel_size is None:
            raise RuntimeError(f"Kernel size is a required parameter with no default value.\n"
                               f"Need to provide value.")

        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=pad, dilation=dilation, groups=groups)
        clip_out = self.define_clip(conv_out, indices, minval, maxval)
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            kh, kw = kernel_size
        elif isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 2
            kh, kw = kernel_size[0], kernel_size[1]
        else:
            raise RuntimeError(f"Invalid type for kernel size")
        self.define_avg_pool(clip_out, out, kh, kw, stride0, pad0)


    def define_avg_pool(self, clip_out, out, kh, kw, stride, pad):
        oh = ((clip_out.shape[-2] + 2 * pad[0] - kh) // stride[0] + 1)
        ow = ((clip_out.shape[-1] + 2 * pad[1] - kw) // stride[1] + 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        m = pm.index(0, kh - 1)
        n = pm.index(0, kw - 1)
        ihp = (clip_out.shape[-2] + pad[0] * 2)
        iwp = clip_out.shape[-1] + pad[1] * 2
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        iy = pm.index(0, clip_out.shape[-2] - 1, name="iy")
        ix = pm.index(0, clip_out.shape[-1] - 1, name="ix")
        if len(clip_out.shape) > 3:
            b = pm.index(0, clip_out.shape[0] - 1, name="b")
            c = pm.index(0, clip_out.shape[1] - 1, name="c")

            o_indices = (b, c)
            p_shape = (clip_out.shape[0], clip_out.shape[1], ihp, iwp)

            out.set_shape((clip_out.shape[0], clip_out.shape[1], oh, ow))

        else:
            c = pm.index(0, clip_out.shape[0] - 1, name="c")
            o_indices = (c,)
            p_shape = (clip_out.shape[0], ihp, iwp)
            out.set_shape((clip_out.shape[0], oh, ow))

        padded = pm.temp(shape=p_shape)
        padded[o_indices, ihp_, iwp_] = 0
        padded[o_indices, iy + pad[0], ix + pad[1]] = clip_out[o_indices, iy, ix]
        out[o_indices, y, x] = pm.sum([m, n], padded[o_indices, stride[0] * y + m, stride[1] * x + n])* (1/(kh*kw))

    def define_clip(self, conv_out, indices, minval, maxval):
        clip_out = pm.temp(shape=conv_out.shape, name=f"{self.name}_clip_out")
        clip_out[indices] = pm.clip(minval, maxval, conv_out[indices])
        return clip_out


    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']

    @property
    def stride0(self):
        return self.kwargs['stride0']

    @property
    def kernel_size(self):
        return self.kwargs['kernel_size']

    @property
    def pad0(self):
        return self.kwargs['pad0']

    @property
    def pad_int0(self):
        if isinstance(self.kwargs['pad0'], tuple):
            return self.kwargs['pad0'][0] + self.kwargs['pad0'][2]
        else:
            return self.kwargs['pad0']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def clip_output(self):
        return self.nodes[f"{self.name}_clip_out"]

class conv_bias_clip_depthwise_conv_bias_add(pm.Template):
    def define_graph(self, data, w, bias, dw_conv_weight, dw_conv_bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None,
                     stride0=1, pad0=0,
                     dilation0=1, groups0=1,
                     ):
        ## Merging padding--special case
        ih = data.shape[2]
        iw = data.shape[3]

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        if not isinstance(pad0, (tuple, list)):
            pad0 = (pad0, pad0)

        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1


        i, n, kernel_h0, kernel_w0 = dw_conv_weight.shape
        # compute the output shape
        if not isinstance(dilation0, (tuple, list)):
            dilation_h0 = dilation_w0 = dilation0
        else:
            dilation_h0, dilation_w0 = dilation0
        dilated_kernel_h0 = (kernel_h0 - 1) * dilation_h0 + 1
        dilated_kernel_w0 = (kernel_w0 - 1) * dilation_w0 + 1

        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad


        if len(pad0) == 2:
            pad_top0, pad_left0, pad_down0, pad_right0 = get_pad_tuple(
                pad0, (dilated_kernel_h0, dilated_kernel_w0)
            )
        else:
            assert len(pad0) == 4
            pad_top0, pad_left0, pad_down0, pad_right0= pad0

        p1 = pad_top + pad_down
        p2 = pad_top0 + pad_down0
        s1 = stride
        kh = w.shape[2]
        kw = w.shape[3]
        oh_num = (ih + 2 * p1 - kh)
        ow_num = (iw + 2 * p1 - kw)
        oh1 = oh_num // s1 + 1
        ow1 = ow_num // s1 + 1
        ih1_ = ((oh1 + 2 * p2) - 1) * s1 + kh
        iw1_ = ((ow1 + 2 * p2) - 1) * s1 + kw
        if oh_num % s1 != 0:
            ih1_ += 1
            iw1_ += 1
        p1 = (ih1_ - ih) // 2
        p2 = 0
        self.kwargs['pad'] = p1
        self.kwargs['pad0'] = p2
        ## End special case

        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=p1, dilation=dilation, groups=groups)
        clip_out = self.define_clip(conv_out, indices, minval, maxval)
        self.define_depthwise_conv(clip_out, dw_conv_weight, dw_conv_bias, out,
                                   stride0,
                                   p2,
                                   groups0,
                                   dilation0
                                   )



    def define_depthwise_conv(self, clip_out, w, bias, out, stride, pad,  groups, dilation):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = clip_out.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1, name="y_")
        x = pm.index(0, ow - 1, name="x_")
        dy = pm.index(0, w.shape[2] - 1, name="dy")
        dx = pm.index(0, w.shape[3] - 1, name="dx")
        iy = pm.index(0, clip_out.shape[-2] - 1, name="iy")
        ix = pm.index(0, clip_out.shape[-1] - 1, name="ix")
        k = pm.index(0, clip_out.shape[-3] - 1, name="k")
        ihp = clip_out.shape[-2] + pad_top + pad_down
        iwp = clip_out.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        if len(clip_out.shape) > 3:
            b = pm.index(0, clip_out.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (clip_out.shape[0], clip_out.shape[1], ihp, iwp)
            out.set_shape((clip_out.shape[0], w.shape[0], oh, ow))
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (clip_out.shape[0], ihp, iwp)
            out.set_shape((w.shape[0], oh, ow))

        padded = pm.temp(shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0

        padded[p_indices + (iy + pad_top, ix + pad_left)] = clip_out[p_indices + (iy, ix)]

        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (
                    padded[p_indices + (dy * dilation_h + stride * y, dx * dilation_w + stride * x)] * w[
                c, k, dy, dx])) + bias[c]

    def define_clip(self, conv_out, indices, minval, maxval):
        clip_out = pm.temp(shape=conv_out.shape, name=f"{self.name}_clip_out")
        clip_out[indices] = pm.clip(minval, maxval, conv_out[indices])
        return clip_out


    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4])

    @property
    def outputs(self):
        return (self.args[5],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']


    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def clip_output(self):
        return self.nodes[f"{self.name}_clip_out"]

    @property
    def stride0(self):
        return self.kwargs['stride0']

    @property
    def pad0(self):
        return self.kwargs['pad0']

    @property
    def pad_int0(self):
        if isinstance(self.kwargs['pad0'], tuple):
            return self.kwargs['pad0'][0] + self.kwargs['pad0'][2]
        else:
            return self.kwargs['pad0']

    @property
    def groups0(self):
        return self.kwargs['groups0']

class conv_bias_clip_depthwise_conv_bias(pm.Template):
    def define_graph(self, data, w, bias, dw_conv_weight, dw_conv_bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None,
                     stride0=1, pad0=0,
                     dilation0=1, groups0=1,
                     ):
        ## Merging padding--special case
        ih = data.shape[2]
        iw = data.shape[3]

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        if not isinstance(pad0, (tuple, list)):
            pad0 = (pad0, pad0)

        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1


        i, n, kernel_h0, kernel_w0 = dw_conv_weight.shape
        # compute the output shape
        if not isinstance(dilation0, (tuple, list)):
            dilation_h0 = dilation_w0 = dilation0
        else:
            dilation_h0, dilation_w0 = dilation0
        dilated_kernel_h0 = (kernel_h0 - 1) * dilation_h0 + 1
        dilated_kernel_w0 = (kernel_w0 - 1) * dilation_w0 + 1

        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad


        if len(pad0) == 2:
            pad_top0, pad_left0, pad_down0, pad_right0 = get_pad_tuple(
                pad0, (dilated_kernel_h0, dilated_kernel_w0)
            )
        else:
            assert len(pad0) == 4
            pad_top0, pad_left0, pad_down0, pad_right0= pad0

        p1 = pad_top + pad_down
        p2 = pad_top0 + pad_down0
        s1 = stride
        kh = w.shape[2]
        kw = w.shape[3]
        oh_num = (ih + 2 * p1 - kh)
        ow_num = (iw + 2 * p1 - kw)
        oh1 = oh_num // s1 + 1
        ow1 = ow_num // s1 + 1
        ih1_ = ((oh1 + 2 * p2) - 1) * s1 + kh
        iw1_ = ((ow1 + 2 * p2) - 1) * s1 + kw
        if oh_num % s1 != 0:
            ih1_ += 1
            iw1_ += 1
        p1 = (ih1_ - ih) // 2
        p2 = 0
        self.kwargs['pad'] = p1
        self.kwargs['pad0'] = p2
        ## End special case

        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=p1, dilation=dilation, groups=groups)
        clip_out = self.define_clip(conv_out, indices, minval, maxval)
        self.define_depthwise_conv(clip_out, dw_conv_weight, dw_conv_bias, out,
                                   stride0,
                                   p2,
                                   groups0,
                                   dilation0
                                   )



    def define_depthwise_conv(self, clip_out, w, bias, out, stride, pad,  groups, dilation):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = clip_out.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1, name="y_")
        x = pm.index(0, ow - 1, name="x_")
        dy = pm.index(0, w.shape[2] - 1, name="dy")
        dx = pm.index(0, w.shape[3] - 1, name="dx")
        iy = pm.index(0, clip_out.shape[-2] - 1, name="iy")
        ix = pm.index(0, clip_out.shape[-1] - 1, name="ix")
        k = pm.index(0, clip_out.shape[-3] - 1, name="k")
        ihp = clip_out.shape[-2] + pad_top + pad_down
        iwp = clip_out.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        if len(clip_out.shape) > 3:
            b = pm.index(0, clip_out.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (clip_out.shape[0], clip_out.shape[1], ihp, iwp)
            out.set_shape((clip_out.shape[0], w.shape[0], oh, ow))
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (clip_out.shape[0], ihp, iwp)
            out.set_shape((w.shape[0], oh, ow))

        padded = pm.temp(shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0

        padded[p_indices + (iy + pad_top, ix + pad_left)] = clip_out[p_indices + (iy, ix)]

        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (
                    padded[p_indices + (dy * dilation_h + stride * y, dx * dilation_w + stride * x)] * w[
                c, k, dy, dx])) + bias[c]

    def define_clip(self, conv_out, indices, minval, maxval):
        clip_out = pm.temp(shape=conv_out.shape, name=f"{self.name}_clip_out")
        clip_out[indices] = pm.clip(minval, maxval, conv_out[indices])
        return clip_out


    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4])

    @property
    def outputs(self):
        return (self.args[5],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']


    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def clip_output(self):
        return self.nodes[f"{self.name}_clip_out"]

    @property
    def stride0(self):
        return self.kwargs['stride0']

    @property
    def pad0(self):
        return self.kwargs['pad0']

    @property
    def pad_int0(self):
        if isinstance(self.kwargs['pad0'], tuple):
            return self.kwargs['pad0'][0] + self.kwargs['pad0'][2]
        else:
            return self.kwargs['pad0']

    @property
    def groups0(self):
        return self.kwargs['groups0']

class conv_bias_clip_depthwise_conv_bias_add_clip(pm.Template):
    def define_graph(self, data, w, bias, dw_conv_weight, dw_conv_bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None,
                     stride0=1, pad0=0,
                     dilation0=1, groups0=1,
                     minval0=None, maxval0=None,
                     ):

        ## Merging padding--special case
        ih = data.shape[2]
        iw = data.shape[3]
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1


        i, n, kernel_h0, kernel_w0 = dw_conv_weight.shape
        # compute the output shape
        if not isinstance(dilation0, (tuple, list)):
            dilation_h0 = dilation_w0 = dilation0
        else:
            dilation_h0, dilation_w0 = dilation0
        dilated_kernel_h0 = (kernel_h0 - 1) * dilation_h0 + 1
        dilated_kernel_w0 = (kernel_w0 - 1) * dilation_w0 + 1

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        if not isinstance(pad0, (tuple, list)):
            pad0 = (pad0, pad0)

        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad


        if len(pad0) == 2:
            pad_top0, pad_left0, pad_down0, pad_right0 = get_pad_tuple(
                pad0, (dilated_kernel_h0, dilated_kernel_w0)
            )
        else:
            assert len(pad0) == 4
            pad_top0, pad_left0, pad_down0, pad_right0 = pad0

        p1 = pad_top + pad_down
        p2 = pad_top0 + pad_down0

        s1 = stride
        kh = w.shape[2]
        kw = w.shape[3]
        oh_num = (ih + 2 * p1 - kh)
        ow_num = (iw + 2 * p1 - kw)
        oh1 = oh_num // s1 + 1
        ow1 = ow_num // s1 + 1
        ih1_ = ((oh1 + 2 * p2) - 1) * s1 + kh
        iw1_ = ((ow1 + 2 * p2) - 1) * s1 + kw
        if oh_num % s1 != 0:
            ih1_ += 1
            iw1_ += 1
        p1 = (ih1_ - ih) // 2
        p2 = 0
        self.kwargs['pad'] = p1
        self.kwargs['pad0'] = p2
        ## End special case
        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=p1, dilation=dilation, groups=groups)
        clip_out = self.define_clip(conv_out, indices, minval, maxval)

        dw_conv_out, indices = self.define_depthwise_conv(clip_out, dw_conv_weight, dw_conv_bias,
                                   stride0,
                                   p2,
                                   groups0,
                                   dilation0
                                   )
        self.define_clip(dw_conv_out, indices, minval0, maxval0, clip_out=out)

    def define_depthwise_conv(self, clip_out, w, bias, stride, pad,  groups, dilation, out=None):

        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = clip_out.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1, name="y_")
        x = pm.index(0, ow - 1, name="x_")
        dy = pm.index(0, w.shape[2] - 1, name="dy")
        dx = pm.index(0, w.shape[3] - 1, name="dx")
        iy = pm.index(0, clip_out.shape[-2] - 1, name="iy")
        ix = pm.index(0, clip_out.shape[-1] - 1, name="ix")
        k = pm.index(0, clip_out.shape[-3] - 1, name="k")
        ihp = clip_out.shape[-2] + pad_top + pad_down
        iwp = clip_out.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        if len(clip_out.shape) > 3:
            b = pm.index(0, clip_out.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (clip_out.shape[0], clip_out.shape[1], ihp, iwp)
            out_shape = (clip_out.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (clip_out.shape[0], ihp, iwp)
            out_shape = (w.shape[0], oh, ow)

        if out is None:
            out = pm.temp(shape=out_shape, name=f"{self.name}_depthwise_conv_out")


        padded = pm.temp(shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0

        padded[p_indices + (iy + pad_top, ix + pad_left)] = clip_out[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        out[out_indices] = pm.sum([dy, dx, k], (
                    padded[p_indices + (dy * dilation_h + stride * y, dx * dilation_w + stride * x)] * w[
                c, k, dy, dx])) + bias[c]
        return out, out_indices


    def define_clip(self, conv_out, indices, minval, maxval, clip_out=None):
        if clip_out is None:
            clip_out = pm.temp(shape=conv_out.shape, name=f"{self.name}_clip_out")
        clip_out[indices] = pm.clip(minval, maxval, conv_out[indices])
        return clip_out


    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4])

    @property
    def outputs(self):
        return (self.args[5],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']


    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def clip_output(self):
        return self.nodes[f"{self.name}_clip_out"]

    @property
    def stride0(self):
        return self.kwargs['stride0']

    @property
    def pad0(self):
        return self.kwargs['pad0']

    @property
    def pad_int0(self):
        if isinstance(self.kwargs['pad0'], tuple):
            return self.kwargs['pad0'][0] + self.kwargs['pad0'][2]
        else:
            return self.kwargs['pad0']

    @property
    def groups0(self):
        return self.kwargs['groups0']

class conv_bias_clip_depthwise_conv_bias_clip(pm.Template):
    def define_graph(self, data, w, bias, dw_conv_weight, dw_conv_bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None,
                     stride0=1, pad0=0,
                     dilation0=1, groups0=1,
                     minval0=None, maxval0=None,
                     ):

        ## Merging padding--special case
        ih = data.shape[2]
        iw = data.shape[3]
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1


        i, n, kernel_h0, kernel_w0 = dw_conv_weight.shape
        # compute the output shape
        if not isinstance(dilation0, (tuple, list)):
            dilation_h0 = dilation_w0 = dilation0
        else:
            dilation_h0, dilation_w0 = dilation0
        dilated_kernel_h0 = (kernel_h0 - 1) * dilation_h0 + 1
        dilated_kernel_w0 = (kernel_w0 - 1) * dilation_w0 + 1

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        if not isinstance(pad0, (tuple, list)):
            pad0 = (pad0, pad0)

        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad


        if len(pad0) == 2:
            pad_top0, pad_left0, pad_down0, pad_right0 = get_pad_tuple(
                pad0, (dilated_kernel_h0, dilated_kernel_w0)
            )
        else:
            assert len(pad0) == 4
            pad_top0, pad_left0, pad_down0, pad_right0 = pad0

        p1 = pad_top + pad_down
        p2 = pad_top0 + pad_down0

        s1 = stride
        kh = w.shape[2]
        kw = w.shape[3]
        oh_num = (ih + 2 * p1 - kh)
        ow_num = (iw + 2 * p1 - kw)
        oh1 = oh_num // s1 + 1
        ow1 = ow_num // s1 + 1
        ih1_ = ((oh1 + 2 * p2) - 1) * s1 + kh
        iw1_ = ((ow1 + 2 * p2) - 1) * s1 + kw
        if oh_num % s1 != 0:
            ih1_ += 1
            iw1_ += 1
        p1 = (ih1_ - ih) // 2
        p2 = 0
        self.kwargs['pad'] = p1
        self.kwargs['pad0'] = p2
        ## End special case
        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=p1, dilation=dilation, groups=groups)
        clip_out = self.define_clip(conv_out, indices, minval, maxval)

        dw_conv_out, indices = self.define_depthwise_conv(clip_out, dw_conv_weight, dw_conv_bias,
                                   stride0,
                                   p2,
                                   groups0,
                                   dilation0
                                   )
        self.define_clip(dw_conv_out, indices, minval0, maxval0, clip_out=out)

    def define_depthwise_conv(self, clip_out, w, bias, stride, pad,  groups, dilation, out=None):

        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = clip_out.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1, name="y_")
        x = pm.index(0, ow - 1, name="x_")
        dy = pm.index(0, w.shape[2] - 1, name="dy")
        dx = pm.index(0, w.shape[3] - 1, name="dx")
        iy = pm.index(0, clip_out.shape[-2] - 1, name="iy")
        ix = pm.index(0, clip_out.shape[-1] - 1, name="ix")
        k = pm.index(0, clip_out.shape[-3] - 1, name="k")
        ihp = clip_out.shape[-2] + pad_top + pad_down
        iwp = clip_out.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        if len(clip_out.shape) > 3:
            b = pm.index(0, clip_out.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (clip_out.shape[0], clip_out.shape[1], ihp, iwp)
            out_shape = (clip_out.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (clip_out.shape[0], ihp, iwp)
            out_shape = (w.shape[0], oh, ow)

        if out is None:
            out = pm.temp(shape=out_shape, name=f"{self.name}_depthwise_conv_out")


        padded = pm.temp(shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0

        padded[p_indices + (iy + pad_top, ix + pad_left)] = clip_out[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        out[out_indices] = pm.sum([dy, dx, k], (
                    padded[p_indices + (dy * dilation_h + stride * y, dx * dilation_w + stride * x)] * w[
                c, k, dy, dx])) + bias[c]
        return out, out_indices


    def define_clip(self, conv_out, indices, minval, maxval, clip_out=None):
        if clip_out is None:
            clip_out = pm.temp(shape=conv_out.shape, name=f"{self.name}_clip_out")
        clip_out[indices] = pm.clip(minval, maxval, conv_out[indices])
        return clip_out


    def define_conv(self, data, w, bias,
                     stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out_indices = o_indices + (y, x)
        conv_out[out_indices] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        return conv_out, out_indices

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4])

    @property
    def outputs(self):
        return (self.args[5],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']


    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def clip_output(self):
        return self.nodes[f"{self.name}_clip_out"]

    @property
    def stride0(self):
        return self.kwargs['stride0']

    @property
    def pad0(self):
        return self.kwargs['pad0']

    @property
    def pad_int0(self):
        if isinstance(self.kwargs['pad0'], tuple):
            return self.kwargs['pad0'][0] + self.kwargs['pad0'][2]
        else:
            return self.kwargs['pad0']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups0(self):
        return self.kwargs['groups0']

class gemm_add_mean_sub_mul_mean_add_sqrt_reciprocal_mul_mul_mul_sub_add(pm.Template):
    def define_graph(self, data, wgt, bias, add_lhs1, add_lhs2, mul_lhs, sub_rhs, output,
                     alpha=1.0, beta=0.0, transA=None, transB=None, strict_shapes=False,
                     axes=(0,), axes0=(0,), keepdims=True, keepdims0=True
                     ):
        gemm_out, indices = self.define_gemm(data, wgt, bias,
                         alpha=alpha, beta=beta, transA=transA,
                         transB=transB, strict_shapes=strict_shapes
                         )
        add_out, add_indices = self.define_add(gemm_out, add_lhs1)
        mean_out, indices = self.define_mean(add_out, axes=axes, keepdims=keepdims)
        sub_out = pm.temp(shape=add_out.shape, name=f"{self.name}_sub_out")
        sub_out[add_indices] = (add_out[add_indices] - mean_out[indices]) * (add_out[add_indices] - mean_out[indices])



    def define_add(self, a, b):
        if len(a.shape) > len(b.shape):
            out_shape = a.shape
        else:
            out_shape = b.shape
        out = pm.temp(shape=out_shape, name=f"{self.name}_add_out")
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # a_idx, b_idx, indices = _get_binop_idx(a, b, out)
        out[indices] = (a[a_idx] + b[b_idx])
        return out, indices

    def define_mean(self, data, axes=(0, ), keepdims=True):
        indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axes])
        out_idx = tuple([indices[i] for i in range(len(indices)) if i not in axes])
        out_shape = []
        for i, s in enumerate(data.shape):
            if i in axes:
                if keepdims:
                    out_shape.append(1)
                else:
                    continue
            else:
                out_shape.append(s)
        out = pm.temp(shape=tuple(out_shape), name=f"{self.name}_reduce_mean_out")
        denom = 1
        for i in axes:
            denom *= data.shape[i]
        out[out_idx] = pm.sum([sum_idx], data[indices]) / (denom)

        return out, out_idx

    def define_gemm(self, a, b, c, alpha=1.0, beta=0.0, transA=None, transB=None, strict_shapes=False):
        if strict_shapes:
            assert b.shape[0] == a.shape[1]
            assert c.shape[0] == b.shape[1]
            assert bool(transB) == bool(transA) and bool(transA) == False, f"Strict shape check failed: {transA} != {transB}"

        if transA:
            i = pm.index(0, a.shape[1] - 1)
            j = pm.index(0, b.shape[0] - 1)
            k = pm.index(0, b.shape[1] - 1)
            out_shape = (a.shape[1], b.shape[1])
            y = pm.temp(shape=out_shape, name=f"{self.name}_gemm_out")
            y[i, k] = pm.sum([j], a[j, i]*b[j, k]) + c[i, k]
        elif transB:
            i = pm.index(0, a.shape[0] - 1)
            j = pm.index(0, b.shape[1] - 1)
            k = pm.index(0, b.shape[0] - 1)
            out_shape = (a.shape[0], b.shape[0])
            y = pm.temp(shape=out_shape, name=f"{self.name}_gemm_out")
            y[i, k] = pm.sum([j], a[i, j]*b[k, j]) + c[i, k]
        else:
            i = pm.index(0, a.shape[0] - 1)
            j = pm.index(0, b.shape[0] - 1)
            k = pm.index(0, b.shape[1] - 1)
            out_shape = (a.shape[0], b.shape[1])
            y = pm.temp(shape=out_shape, name=f"{self.name}_gemm_out")
            y[i, k] = pm.sum([j], a[i, j]*b[j, k]) + c[i, k]
        return y, (i, k)

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4], self.args[5], self.args[6])

    @property
    def outputs(self):
        return (self.args[7],)

class matmul_reshape_add_add_mean_sub_mul_mean_add_sqrt_reciprocal_mul_mul_mul_sub_add(pm.Template):
    def define_graph(self, data, wgt, add_lhs1, add_lhs2, add_lhs3, mul_lhs, sub_rhs, output,
                     axes=(0,), axes0=(0,), keepdims=True, keepdims0=True
                     ):
        self.names = defaultdict(int)
        gemm_out, indices = self.define_matmul(data, wgt)
        add_out0, add_indices = self.define_add(gemm_out, add_lhs1)
        add_out1, add_indices = self.define_add(add_out0, add_lhs1)
        mean_out, indices = self.define_mean(add_out1, axes=axes, keepdims=keepdims)
        sub_out = pm.temp(shape=add_out1.shape, name=f"{self.name}_sub_out")
        sub_out[add_indices] = (add_out1[add_indices] - mean_out[indices]) * (add_out1[add_indices] - mean_out[indices])


    def get_name(self, base_name):
        name = f"{self.names[base_name]}{base_name}"
        self.names[base_name] += 1
        return name

    def define_add(self, a, b):
        if len(a.shape) > len(b.shape):
            out_shape = a.shape
        else:
            out_shape = b.shape
        out = pm.temp(shape=out_shape, name=self.get_name(f"{self.name}_add_out"))
        a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # a_idx, b_idx, indices = _get_binop_idx(a, b, out)
        out[indices] = (a[a_idx] + b[b_idx])
        return out, indices

    def define_mean(self, data, axes=(0,), keepdims=True):
        indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axes])
        out_idx = tuple([indices[i] for i in range(len(indices)) if i not in axes])
        out_shape = []
        for i, s in enumerate(data.shape):
            if i in axes:
                if keepdims:
                    out_shape.append(1)
                else:
                    continue
            else:
                out_shape.append(s)
        out = pm.temp(shape=tuple(out_shape), name=f"{self.name}_reduce_mean_out")
        denom = 1
        for i in axes:
            denom *= data.shape[i]
        out[out_idx] = pm.sum([sum_idx], data[indices]) / (denom)

        return out, out_idx

    def define_matmul(self, a, w):
        indices = _get_single_node_indices(a)
        sum_idx = indices[-1]
        o_idx = pm.index(0, w.shape[0]-1) if w.shape[-1] == a.shape[-1] else pm.index(0, w.shape[1]-1)
        o_shape = (a.shape[0], w.shape[0]) if w.shape[-1] == a.shape[-1] else (a.shape[0], w.shape[1])
        w_idx = (o_idx, sum_idx) if w.shape[-1] == a.shape[-1] else (sum_idx, o_idx)
        out_idx = indices[:-1] + (o_idx,)
        out = pm.temp(shape=o_shape, name=f"{self.name}_matmul_out")

        out[out_idx] = pm.sum([sum_idx], a[indices]*w[w_idx])

        return out, out_idx

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4], self.args[5], self.args[6])

    @property
    def outputs(self):
        return (self.args[7],)

class matmul_mul_add_softmax(pm.Template):
    def define_graph(self, data, weight, mul_rhs, add_lhs, out, axis=(0,)):
        pass

    @property
    def axis(self):
        return self.kwargs['axis']

    @property
    def inputs(self):
        return (self.args[0],self.args[1], self.args[2], self.args[3],)

    @property
    def outputs(self):
        return (self.args[-1],)

class gemm_reshape_transpose(pm.Template):
    def define_graph(self, data, wgt, bias, output,
                     alpha=1.0, beta=0.0, transA=None, transB=None, strict_shapes=False, perm=None):
        pass


    @property
    def perm(self):
        return self.kwargs['perm']

    @property
    def inputs(self):
        return (self.args[0],self.args[1], self.args[2],)

    @property
    def outputs(self):
        return (self.args[-1],)

class matmul_transpose(pm.Template):
    def define_graph(self, data, w, output,
                     perm=None):
        pass

    @property
    def inputs(self):
        return (self.args[0],self.args[1],)

    @property
    def outputs(self):
        return (self.args[-1],)

class gemm_pow_mul_add_mul_tanh_add_mul_mul(pm.Template):
    def define_graph(self, data, wgt, bias, mul_lhs0, mul_lhs1, add_lhs, mul_lhs2, output,
                     alpha=1.0, beta=0.0, transA=None, transB=None, strict_shapes=False,
                     exp=None):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4], self.args[5], self.args[6])

    @property
    def outputs(self):
        return (self.args[-1],)

class matmul_add(pm.Template):
    def define_graph(self, data, wgt, bias, out):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[-1],)

class matmul_add_gelu(pm.Template):
    def define_graph(self, data, wgt, bias, out):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[-1],)

class matmul_add_add(pm.Template):
    def define_graph(self, data, wgt, bias, add_lhs, out):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3])

    @property
    def outputs(self):
        return (self.args[-1],)

class mul_add(pm.Template):
    def define_graph(self, data, mul_rhs, add_rhs, out):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[-1],)


class pow_mul_add_tanh_mul(pm.Template):
    def define_graph(self, data, mul_lhs1, add_lhs, mul_lhs2, out, exp=None):
        self.kwargs['mul_lhs1'] = mul_lhs1.default
        self.kwargs['add_lhs'] = add_lhs.default
        self.kwargs['mul_lhs2'] = mul_lhs2.default

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def exp(self):
        return self.kwargs['exp']

class matmul_div_add(pm.Template):
    def define_graph(self, data, wgt, div_rhs, add_rhs, out):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[3],)

    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def div_lhs(self):
        return self.args[2].default

class add_add(pm.Template):
    def define_graph(self, data, add1_rhs, add2_rhs, out):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[-1],)

class sub_mul(pm.Template):
    def define_graph(self, data, sub_rhs, mul_rhs, out):
        self.kwargs['mul_rhs'] = mul_rhs.default
        self.kwargs['sub_rhs'] = data.default

    @property
    def inputs(self):
        return (self.args[1],)

    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def mul_rhs(self):
        return self.args[2].default

    @property
    def sub_rhs(self):
        return self.args[0].default

class mean_sub_pow_mean_add_sqrt_div(pm.Template):
    def define_graph(self, data, add_val, out,
                     axes=None,
                     keepdims=True,
                     exp=None,
                     axes0=None,
                     keepdims0=None):
        pass

    @property
    def axes(self):
        return self.kwargs['axes']

    @property
    def axes0(self):
        return self.kwargs['axes0']

    @property
    def exp(self):
        return self.kwargs['exp']

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[-1],)

class sub_pow(pm.Template):
    def define_graph(self, data, sub_lhs, out,
                     exp=None):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1],)

    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def exp(self):
        return self.kwargs['exp']

class add_sqrt_div(pm.Template):
    def define_graph(self, data, add_lhs, div_rhs, out):
        self.kwargs['add_lhs'] = add_lhs.default

    @property
    def inputs(self):
        return (self.args[0], self.args[2])

    @property
    def outputs(self):
        return (self.args[-1],)


    @property
    def add_lhs(self):
        return self.args[1].default

class div_add(pm.Template):
    def define_graph(self, data, div_rhs, add_rhs, out):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[2],)

    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def div_lhs(self):
        return self.args[1].default

class add_relu(pm.Template):
    def define_graph(self, data, add_rhs, out):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1],)

    @property
    def outputs(self):
        return (self.args[-1],)

class add_leaky_relu(pm.Template):
    def define_graph(self, data, add_rhs, out, alpha=None):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1],)

    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def alpha(self):
        return self.kwargs['alpha']

class leaky_relu_add(pm.Template):
    def define_graph(self, data, add_rhs, out, alpha=None):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1],)

    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def alpha(self):
        return self.kwargs['alpha']

class clip_depthwise_conv(pm.Template):
    def define_graph(self, data, w, out, stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        add_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_add_out")
        out.set_shape(conv_out_shape)

        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        # out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]
        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx]))
        # add_out[o_indices + (y, x)] = conv_out[o_indices + (y, x)] + op1[conv_out[o_indices + (y, x)]]
        # out[o_indices + (y, x)] = (0 < add_out[o_indices + (y, x)]) * add_out[o_indices + (y, x)]

    @property
    def inputs(self):
        return (self.args[0], self.args[1],)

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]


    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']

class clip_depthwise_conv_bias(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        add_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_add_out")
        out.set_shape(conv_out_shape)

        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]


    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']

class clip_depthwise_conv_bias_clip(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None, minval0=None, maxval0=None):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(dilation, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        out_channel = num_filter
        oh = (in_height + pad_top + pad_down - dilation_h*(kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w*(kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
        c = pm.index(0, w.shape[0] - 1)
        y = pm.index(0, oh - 1)
        x = pm.index(0, ow - 1)
        dy = pm.index(0, w.shape[2] - 1)
        dx = pm.index(0, w.shape[3] - 1)
        iy = pm.index(0, data.shape[-2] - 1)
        ix = pm.index(0, data.shape[-1] - 1)
        k = pm.index(0, data.shape[-3] - 1)
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1)
        iwp_ = pm.index(0, iwp - 1)
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1)
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            conv_out_shape = (data.shape[0], w.shape[0], oh, ow)
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            conv_out_shape = (w.shape[0], oh, ow)

        conv_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_conv_out")
        add_out = pm.temp(shape=conv_out_shape, name=f"{self.name}_add_out")
        out.set_shape(conv_out_shape)

        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        # out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]
        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]
        # add_out[o_indices + (y, x)] = conv_out[o_indices + (y, x)] + op1[conv_out[o_indices + (y, x)]]
        # out[o_indices + (y, x)] = (0 < add_out[o_indices + (y, x)]) * add_out[o_indices + (y, x)]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2],)

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        if isinstance(self.kwargs['pad'], tuple):
            return self.kwargs['pad'][0] + self.kwargs['pad'][2]
        else:
            return self.kwargs['pad']

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']

    @property
    def minval0(self):
        return self.kwargs['minval']

    @property
    def maxval0(self):
        return self.kwargs['maxval']

class bias_add_clip(pm.Template):
    def define_graph(self, data, bias, out,  minval=None, maxval=None):
        pass
        # add_out[o_indices + (y, x)] = conv_out[o_indices + (y, x)] + op1[conv_out[o_indices + (y, x)]]
        # out[o_indices + (y, x)] = (0 < add_out[o_indices + (y, x)]) * add_out[o_indices + (y, x)]

    @property
    def inputs(self):
        return (self.args[0], self.args[1],)


    @property
    def outputs(self):
        return (self.args[-1],)

    @property
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']
