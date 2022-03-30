import polymath as pm
from .template_utils import _get_indices, _get_single_node_indices, _get_elem_indices, pad_node, \
    _dim_explicit, get_pad_tuple

class elem_sqrt_reciprocal(pm.Template):
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

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def groups(self):
        return self.kwargs['groups']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

class conv_bias_elem_add_relu(pm.Template):
    def define_graph(self, data, w, bias, op1, out, stride=1, pad=0, dilation=1, groups=1):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def groups(self):
        return self.kwargs['groups']

class conv_bias_leaky_relu(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0, dilation=1, groups=1, alpha=1e-2):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def groups(self):
        return self.kwargs['groups']

    @property
    def alpha(self):
        return self.kwargs['alpha']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

class conv_bias_elem_add_leaky_relu(pm.Template):
    def define_graph(self, data, w, bias, op1, out, stride=1, pad=0, dilation=1, groups=1, alpha=1e-2):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def groups(self):
        return self.kwargs['groups']

    @property
    def alpha(self):
        return self.kwargs['alpha']

class conv_bias_relu_max_pool(pm.Template):
    def define_graph(self, data, w, bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     kernel_size=None, max_pool_stride=(1,1), max_pool_pad=(0,0)
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
        self.define_max_pool(relu_out, out, kh, kw, max_pool_stride, max_pool_pad)


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

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def max_pool_stride(self):
        return self.kwargs['max_pool_stride']

    @property
    def kernel_size(self):
        return self.kwargs['kernel_size']

    @property
    def max_pool_pad(self):
        return self.kwargs['max_pool_pad']


    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def relu_output(self):
        return self.nodes[f"{self.name}_relu_out"]


class conv_bias_elem_add_relu_global_avg_pool(pm.Template):
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

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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


class conv_bias_elem_add(pm.Template):
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

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def groups(self):
        return self.kwargs['groups']


class conv_bias_elem_clip_avg_pool(pm.Template):
    def define_graph(self, data, w, bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None,
                     kernel_size=None, avg_pool_stride=(1,1), avg_pool_pad=(0,0)
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
        self.define_avg_pool(clip_out, out, kh, kw, avg_pool_stride, avg_pool_pad)


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

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def minval(self):
        return self.kwargs['minval']

    @property
    def maxval(self):
        return self.kwargs['maxval']

    @property
    def avg_pool_stride(self):
        return self.kwargs['avg_pool_stride']

    @property
    def kernel_size(self):
        return self.kwargs['kernel_size']

    @property
    def avg_pool_pad(self):
        return self.kwargs['avg_pool_pad']

    @property
    def groups(self):
        return self.kwargs['groups']

    @property
    def conv_output(self):
        return self.nodes[f"{self.name}_conv_out"]

    @property
    def clip_output(self):
        return self.nodes[f"{self.name}_clip_out"]


class conv_bias_elem_clip_depthwise_conv_bias(pm.Template):
    def define_graph(self, data, w, bias, dw_conv_weight, dw_conv_bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None,
                     depthwise_conv_bias_stride=1, depthwise_conv_bias_pad=0,
                     depthwise_conv_bias_dilation=1, depthwise_conv_bias_groups=1,
                     ):
        ## Merging padding--special case
        ih = data.shape[2]
        iw = data.shape[3]
        p1 = pad
        p2 = depthwise_conv_bias_pad
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
        self.kwargs['depthwise_conv_bias_pad'] = p2
        ## End special case

        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=p1, dilation=dilation, groups=groups)
        clip_out = self.define_clip(conv_out, indices, minval, maxval)
        self.define_depthwise_conv(clip_out, dw_conv_weight, dw_conv_bias, out,
                                   depthwise_conv_bias_stride,
                                   p2,
                                   depthwise_conv_bias_groups,
                                   depthwise_conv_bias_dilation
                                   )



    def define_depthwise_conv(self, clip_out, w, bias, out, stride, pad,  groups, dilation):
        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = clip_out.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def depthwise_conv_bias_stride(self):
        return self.kwargs['depthwise_conv_bias_stride']

    @property
    def depthwise_conv_bias_pad(self):
        return self.kwargs['depthwise_conv_bias_pad']

    @property
    def depthwise_conv_bias_groups(self):
        return self.kwargs['depthwise_conv_bias_groups']



class conv_bias_elem_clip_depthwise_conv_bias_elem_clip(pm.Template):
    def define_graph(self, data, w, bias, dw_conv_weight, dw_conv_bias, out,
                     stride=1, pad=0, dilation=1, groups=1,
                     minval=None, maxval=None,
                     depthwise_conv_bias_stride=1, depthwise_conv_bias_pad=0,
                     depthwise_conv_bias_dilation=1, depthwise_conv_bias_groups=1,
                     elem_clip_minval=None, elem_clip_maxval=None,
                     ):
        ## Merging padding--special case
        ih = data.shape[2]
        iw = data.shape[3]
        p1 = pad
        p2 = depthwise_conv_bias_pad
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
        self.kwargs['depthwise_conv_bias_pad'] = p2
        ## End special case
        conv_out, indices = self.define_conv(data, w, bias, stride=stride, pad=p1, dilation=dilation, groups=groups)
        clip_out = self.define_clip(conv_out, indices, minval, maxval)

        dw_conv_out, indices = self.define_depthwise_conv(clip_out, dw_conv_weight, dw_conv_bias,
                                   depthwise_conv_bias_stride,
                                   p2,
                                   depthwise_conv_bias_groups,
                                   depthwise_conv_bias_dilation
                                   )
        self.define_clip(dw_conv_out, indices, elem_clip_minval, elem_clip_maxval, clip_out=out)

    def define_depthwise_conv(self, clip_out, w, bias, stride, pad,  groups, dilation, out=None):

        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = clip_out.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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

        if not isinstance(stride, (tuple, list)):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if not isinstance(stride, (tuple, list)):
            pad = (pad, pad)

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = w.shape
        # compute the output shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            pad, (dilated_kernel_h, dilated_kernel_w)
        )
        out_channel = num_filter
        oh = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
        ow = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
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
    def depthwise_conv_bias_stride(self):
        return self.kwargs['depthwise_conv_bias_stride']

    @property
    def depthwise_conv_bias_pad(self):
        return self.kwargs['depthwise_conv_bias_pad']

    @property
    def depthwise_conv_bias_groups(self):
        return self.kwargs['depthwise_conv_bias_groups']