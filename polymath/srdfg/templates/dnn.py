import polymath as pm
from .template_utils import _get_indices, _get_single_node_indices, _get_elem_indices, pad_node, \
    _dim_explicit, get_pad_tuple
from .tensor_transformations import pad_tensor_inlined, flip_tensor_inlined
from polymath.srdfg.util import squeeze_shape
from numbers import Integral
import numpy as np
from math import ceil
import functools

class cross_entropy_loss(pm.Template):
    def define_graph(self, z, y, loss, reduction="mean"):
        a = pm.temp(name=f"temp_{y.name}", shape=z.shape)

        i = pm.index(0, z.shape[1]-1, name="i")
        indices = [pm.index(0, s - 1, name=f"{z.name}[{i}]") for i, s in enumerate(z.shape)]
        indices[1] = i
        indices = tuple(indices)
        maxes = pm.max([i], z[indices], name="maxes")
        exp_val = pm.exp((z[indices] - maxes[indices[0]]))
        lse_stable = pm.log(pm.sum([i], exp_val[indices], name="testing_lse"), name="lse_stable")
        a[indices] = z[indices] - maxes[indices[0]] - lse_stable[indices[0]]
        gathered = pm.gather_elements(a, pm.reshape(y, (a.shape[0], 1), name="reshaped1"), axis=1, shape=(y.shape[0],), name="gathered_elem")
        reshaped = pm.reshape(-1*gathered, (y.shape[0],), name="other_reshape")
        idx = (pm.index(0, a.shape[0] - 1),)
        if reduction == "none":
            loss.set_shape(reshaped.shape)
            loss[idx] = reshaped[idx]
        elif reduction == "mean":
            loss.set_shape((z.shape[0],))
            denom = 1
            for s in reshaped.shape:
                denom = denom*s
            loss[0] = pm.sum([idx[0]], reshaped[idx], name="test_sum_name")/denom
        elif reduction == "sum":
            loss.set_shape((z.shape[0],))
            loss[0] = pm.sum([idx[0]], reshaped[idx])


        # TODO: Get this working
        # pm.log_softmax(z, a, axis=1)
        # pm.nll_loss(a, y, loss, reduction=reduction)

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class concat(pm.Template):
    def define_graph(self, *input_tensors, axis=None):
        assert isinstance(input_tensors, tuple)
        output_tensor = input_tensors[-1]

    @property
    def inputs(self):
        return self.args[0][:-1]

    @property
    def outputs(self):
        return (self.args[0][-1],)

    @property
    def axis(self):
        return self.kwargs['axis']



class nll_loss(pm.Template):
    def define_graph(self, logs, targets, out, reduction="mean"):
        # gathered = pm.gather_elements(logs, pm.reshape(targets, shape=(logs.shape[0], 1)), axis=1)
        gathered = pm.gather_elements(logs, pm.reshape(targets, (logs.shape[0], 1)), axis=1)
        # reshaped = pm.reshape(-1*gathered, shape=(logs.shape[0],))
        reshaped = pm.reshape(-1*gathered, (logs.shape[0],))
        idx = (pm.index(0, logs.shape[0] - 1),)
        if reduction == "none":
            out.set_shape(reshaped.shape)
            out[idx] = reshaped[idx]
        elif reduction == "mean":
            out.set_shape((1,))
            denom = 1
            for s in reshaped.shape:
                denom = denom*s
            out[0] = pm.sum([idx[0]], reshaped[idx])/denom
        elif reduction == "sum":
            out.set_shape((1,))
            out[0] = pm.sum([idx[0]], reshaped[idx])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class log_softmax(pm.Template):
    def define_graph(self, data, out, axis=0):
        out.set_shape(data.shape)
        i = pm.index(0, data.shape[axis]-1, name="i")
        indices = [pm.index(0, s - 1, name=f"{data.name}[{i}]") for i, s in enumerate(data.shape)]
        indices[axis] = i
        indices = tuple(indices)
        maxes = pm.max([i], data[indices], name="maxes")
        lse_stable = pm.log(pm.sum([i], pm.exp((data[indices] - maxes[indices[0]]))), name="lse_stable")
        out[indices] = data[indices] - maxes[indices[0]] - lse_stable[indices[0]]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class avg_pool(pm.Template):
    def define_graph(self, data, out, kernel_size=None, stride=(1,1), pad=(0,0)):
        if kernel_size is None:
            raise RuntimeError(f"Kernel size is a required parameter with no default value.\n"
                               f"Need to provide value.")
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            kh, kw = kernel_size
        elif isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 2
            kh, kw = kernel_size[0], kernel_size[1]
        else:
            raise RuntimeError(f"Invalid type for kernel size")
        sx, sy = stride
        oh = ((data.shape[-2] + 2 * pad[0] - kh) // stride[0] + 1)
        ow = ((data.shape[-1] + 2 * pad[1] - kw) // stride[1] + 1)

        y = pm.index(0, oh - 1, name="y")
        x = pm.index(0, ow - 1, name="x")
        m = pm.index(0, kh - 1, name="m")
        n = pm.index(0, kw - 1, name="n_")
        ihp = (data.shape[-2] + pad[0] * 2)
        iwp = data.shape[-1] + pad[1] * 2
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        iy = pm.index(0, data.shape[-2] - 1, name="iy")
        ix = pm.index(0, data.shape[-1] - 1, name="ix")

        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1, name="b")
            c = pm.index(0, data.shape[1] - 1, name="c")

            o_indices = [b, c]
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], data.shape[1], oh, ow))

        else:
            c = pm.index(0, data.shape[0] - 1, name="c")
            o_indices = [c]
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((data.shape[0], oh, ow))
        o_indices = tuple(o_indices)
        padded = pm.temp(shape=p_shape)
        padded[o_indices + (ihp_, iwp_)] = 0
        padded[o_indices + (iy + pad[0], ix + pad[1])] = data[o_indices + (iy, ix)]
        out[o_indices + (y, x)] = pm.sum([m, n], padded[o_indices + (sx*y + m, sy*x + n)]) * (1/(kh*kw))

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def kernel_size(self):
        return self.kwargs['kernel_size']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        return self.kwargs['pad'][0] + self.kwargs['pad'][2]

class dense(pm.Template):
    def define_graph(self, x, w, y):
        i = pm.index(0, (w.shape[1] - 1), name="i")
        j = pm.index(0, (w.shape[0] - 1), name="j")
        y.set_shape((w.shape[0]))
        y[j] = pm.sum([i], w[j, i] * x[i], name="h")

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class roi_align(pm.Template):
    def define_graph(self, x, rois, batch_indices, out, mode='avg',
                  output_height=1, output_width=1,
                  sampling_ratio=0, spatial_scale=1.0):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)


class dense_sigmoid(pm.Template):
    def define_graph(self, x, w, y):
        i = pm.index(0, (w.shape[1] - 1), name="i")
        j = pm.index(0, (w.shape[0] - 1), name="j")
        y[j] = pm.sigmoid(pm.sum([i], w[j, i] * x[i], name="h"))

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)



class relu(pm.Template):
    def define_graph(self, inp, out):
        out.set_shape(inp.shape)
        # indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in inp.shape])
        indices = tuple([pm.index(0, s - 1) for s in inp.shape])
        out[indices] = (0 < inp[indices]) * inp[indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class gelu(pm.Template):
    def define_graph(self, inp, out):
        out.set_shape(inp.shape)
        # indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in inp.shape])
        indices = tuple([pm.index(0, s - 1) for s in inp.shape])
        out[indices] = (0 < inp[indices]) * inp[indices]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class leaky_relu(pm.Template):
    def define_graph(self, inp, out, alpha=1e-2):
        out.set_shape(inp.shape)
        # indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in inp.shape])
        indices = tuple([pm.index(0, s - 1) for s in inp.shape])

        out[indices] = (0 < inp[indices]) * inp[indices] + (0 >= inp[indices]) * inp[indices] * alpha

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def alpha(self):
        return self.kwargs['alpha']

class one_hot(pm.Template):
    def define_graph(self, indices, depth, values, out, axis=None):
        pass

    @property
    def inputs(self):
        return (self.args[0], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

    @property
    def axis(self):
        return self.kwargs['axis']


class relu1d(pm.Template):
    def define_graph(self, inp, out):
        i = pm.index(0, inp.shape[0] - 1, name="i")
        out.set_shape(inp.shape)
        out.write((0 < inp[i]) * inp[i])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class conv_bias(pm.Template):
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
        # compute the output shapeget
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

        if len(pad) == 2:
            pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
                pad, (dilated_kernel_h, dilated_kernel_w)
            )
        else:
            assert len(pad) == 4
            pad_top, pad_left, pad_down, pad_right = pad
        self.kwargs['pad'] = (pad_top, pad_left, pad_down, pad_right)

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
            out.set_shape((data.shape[0], w.shape[0], oh, ow))
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((w.shape[0], oh, ow))

        # padded = pm.temp(shape=p_shape)

        padded = pm.temp(shape=p_shape)

        padded[p_indices + (ihp_, iwp_)] = 0
        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]
        # out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]
        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy*dilation_h + stride*y, dx*dilation_w + stride*x)] * w[c, k, dy, dx])) + bias[c]

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
        return self.kwargs['pad'][0] + self.kwargs['pad'][2]

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

# TODO: Make flexible for different conv shapes
class conv_transpose_bias(pm.Template):
    def define_graph(self, data, wgt, bias, out, stride=1, pad=0, out_pad=0):
        n, c, h, w = data.shape
        dim_in, dim_out, kh, kw = wgt.shape

        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)
        #### OLD METHOD ###
        # y = pm.temp(name=f"{data.name}_reshaped", shape=(n*c, h*w, 1, 1))
        # pm.tensor_reshape(data, y, y.shape)
        #
        #
        # y1 = pm.temp(name=f"{data.name}_reshaped1")
        # pm.tensor_pad(y, y1, ((0, 0), (0, 0), (0, stride_h), (0, stride_w)))
        #
        # y2 = pm.temp(name=f"{data.name}_reshaped2", shape=(n * c, h, w, 1 + stride_h, 1 + stride_w))
        # pm.tensor_reshape(y1, y2, y2.shape)
        #
        # y3 = pm.temp(name=f"{data.name}_permuted", shape=(n * c, h, 1 + stride_h, w, 1 + stride_w))
        # pm.tensor_transpose(y2, y3, (0, 1, 3, 2, 4))
        #
        # y4 = pm.temp(name=f"{data.name}_reshaped3", shape=(n, c, h * (1 + stride_h), w * (1 + stride_w)))
        # pm.tensor_reshape(y3, y4, y4.shape)
        # ph, pw = kh - pad[0] - 1, kw - pad[1] - 1
        #
        # w_perm = pm.temp(name=f"{wgt.name}_perm", shape=(wgt.shape[1], wgt.shape[0], wgt.shape[2], wgt.shape[3]))
        # w_perm_flip = pm.temp(name=f"{wgt.name}_flip", shape=(wgt.shape[1], wgt.shape[0], wgt.shape[2], wgt.shape[3]))
        #
        # pm.tensor_transpose(wgt, w_perm, (1, 0, 2, 3))
        # pm.tensor_flip(w_perm, w_perm_flip, (2, 3))
        #
        # y5 = pm.temp(name=f"{data.name}_pad2")
        # pm.tensor_pad(y4, y5, ((0, 0), (0, 0), (pw, pw - stride_w + out_pad), (ph, ph - stride_h + out_pad)))

        ### END OLD METHOD

        ## NEW METHOD
        # First flip the weights
        w_perm = pm.temp(name=f"{wgt.name}_perm", shape=(wgt.shape[1], wgt.shape[0], wgt.shape[2], wgt.shape[3]))
        w_perm_flip = pm.temp(name=f"{wgt.name}_flip", shape=(wgt.shape[1], wgt.shape[0], wgt.shape[2], wgt.shape[3]))
        pm.tensor_transpose(wgt, w_perm, (1, 0, 2, 3))
        # pm.tensor_flip(w_perm, w_perm_flip, (2, 3))
        flip_tensor_inlined(w_perm, w_perm_flip, (2, 3))

        # Next, dilate the input tensor
        p_shape = (data.shape[0], data.shape[1], (data.shape[2] - 1)*stride + 1, (data.shape[3] - 1)*stride + 1)
        dilated = pm.temp(name=f"{data.name}_grad_dilated", shape=p_shape)
        b_idx = pm.index(0, data.shape[0] - 1)
        ic_idx = pm.index(0, data.shape[1] - 1)
        ih_idx = pm.index(0, data.shape[2] - 1)
        iw_idx = pm.index(0, data.shape[3] - 1)
        dilated[b_idx, ic_idx, ih_idx*stride_h, iw_idx*stride_w] = data[b_idx, ic_idx, ih_idx, iw_idx]

        padded = pm.temp(name=f"{data.name}_padded_grad")
        ### Replacing tensor pad op with inlined version
        # pm.tensor_pad(dilated, padded, ((0, 0), (0, 0), (kh-1-pad[0], kh-1-pad[0] + out_pad), (kw-1-pad[0], kw-1-pad[0] + out_pad)))
        pad_tensor_inlined(dilated, padded, ((0, 0), (0, 0), (kh-1-pad[0], kh-1-pad[0] + out_pad), (kw-1-pad[0], kw-1-pad[0] + out_pad)))

        ### End tensor pad
        # print(f"\nOut shape: {out.shape}, Input shape: {padded.shape}, unpadded: {data.shape}, stride: {stride_h}, kh: {kh}, pad: {pad}, {out.shape}, out pad: {out_pad}")

        pm.conv_bias(padded, w_perm_flip, bias, out, stride=1, pad=0)
        # print(f"Shape after {out.name}: {out.shape}, {out_pad}")


    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

class conv_transpose(pm.Template):
    def define_graph(self, data, wgt, out, stride=1, pad=0, out_pad=0):
        n, c, h, w = data.shape
        dim_in, dim_out, kh, kw = wgt.shape

        if not isinstance(stride, (tuple, list)):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if not isinstance(pad, (tuple, list)):
            pad = (pad, pad)


        # y = pm.temp(name=f"{data.name}_reshaped", shape=(n*c, h*w, 1, 1))
        # pm.tensor_reshape(data, y, y.shape)
        #
        #
        # y1 = pm.temp(name=f"{data.name}_reshaped1")
        # pm.tensor_pad(y, y1, ((0, 0), (0, 0), (0, sh), (0, sw)))
        #
        # y2 = pm.temp(name=f"{data.name}_reshaped2", shape=(n * c, h, w, 1 + sh, 1 + sw))
        # pm.tensor_reshape(y1, y2, y2.shape)
        #
        # y3 = pm.temp(name=f"{data.name}_permuted", shape=(n * c, h, 1 + sh, w, 1 + sw))
        # pm.tensor_transpose(y2, y3, (0, 1, 3, 2, 4))
        #
        # y4 = pm.temp(name=f"{data.name}_reshaped3", shape=(n, c, h * (1 + sh), w * (1 + sw)))
        # pm.tensor_reshape(y3, y4, y4.shape)
        # ph, pw = kh - pad - 1, kw - pad - 1
        #
        # w_perm = pm.temp(name=f"{wgt.name}_perm", shape=(wgt.shape[1], wgt.shape[0], wgt.shape[2], wgt.shape[3]))
        # w_perm_flip = pm.temp(name=f"{wgt.name}_flip", shape=(wgt.shape[1], wgt.shape[0], wgt.shape[2], wgt.shape[3]))
        #
        # pm.tensor_transpose(wgt, w_perm, (1, 0, 2, 3))
        # pm.tensor_flip(w_perm, w_perm_flip, (2, 3))
        #
        # y5 = pm.temp(name=f"{data.name}_pad2")
        # ## Replacing tensor pad with inlined version
        # # pm.tensor_pad(y4, y5, ((0, 0), (0, 0), (pw, pw - sw + out_pad), (ph, ph - sh + out_pad)))
        # pad_tensor_inlined(y4, y5, ((0, 0), (0, 0), (pw, pw - sw + out_pad), (ph, ph - sh + out_pad)))
        #
        # # end inlined
        # pm.conv(y5, w_perm_flip, out, pad=0, stride=1)
        w_perm = pm.temp(name=f"{wgt.name}_perm", shape=(wgt.shape[1], wgt.shape[0], wgt.shape[2], wgt.shape[3]))
        w_perm_flip = pm.temp(name=f"{wgt.name}_flip", shape=(wgt.shape[1], wgt.shape[0], wgt.shape[2], wgt.shape[3]))
        pm.tensor_transpose(wgt, w_perm, (1, 0, 2, 3))

        # pm.tensor_flip(w_perm, w_perm_flip, (2, 3))
        flip_tensor_inlined(w_perm, w_perm_flip, (2, 3))

        # Next, dilate the input tensor
        p_shape = (data.shape[0], data.shape[1], (data.shape[2] - 1) * stride + 1, (data.shape[3] - 1) * stride + 1)
        dilated = pm.temp(name=f"{data.name}_grad_dilated", shape=p_shape)
        b_idx = pm.index(0, data.shape[0] - 1)
        ic_idx = pm.index(0, data.shape[1] - 1)
        ih_idx = pm.index(0, data.shape[2] - 1)
        iw_idx = pm.index(0, data.shape[3] - 1)
        dilated[b_idx, ic_idx, ih_idx * stride_h, iw_idx * stride_w] = data[b_idx, ic_idx, ih_idx, iw_idx]

        padded = pm.temp(name=f"{data.name}_padded_grad")
        ### Replacing tensor pad op with inlined version
        # pm.tensor_pad(dilated, padded, ((0, 0), (0, 0), (kh-1-pad[0], kh-1-pad[0] + out_pad), (kw-1-pad[0], kw-1-pad[0] + out_pad)))
        pad_tensor_inlined(dilated, padded, (
        (0, 0), (0, 0), (kh - 1 - pad[0], kh - 1 - pad[0] + out_pad), (kw - 1 - pad[0], kw - 1 - pad[0] + out_pad)))

        ### End tensor pad
        # print(f"\nOut shape: {out.shape}, Input shape: {padded.shape}, unpadded: {data.shape}, stride: {stride_h}, kh: {kh}, pad: {pad}, {out.shape}, out pad: {out_pad}")

        pm.conv(padded, w_perm_flip, out, stride=1, pad=0)

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class avg_pool2d(pm.Template):
    def define_graph(self, inp, out, kh, kw, stride=1, pad=0):
        oh = ((inp.shape[2] + 2 * pad - kh) // stride + 1)
        ow = ((inp.shape[3] + 2 * pad - kw) // stride + 1)
        out.set_shape((inp.shape[0], inp.shape[1], oh, ow))

        b = pm.index(0, inp.shape[0]-1, name="b")
        c = pm.index(0, inp.shape[1]-1, name="c")
        y = pm.index(0, oh-1, name="y")
        x = pm.index(0, ow-1, name="x")
        m = pm.index(0, kh-1, name="m")
        n = pm.index(0, kw-1, name="n_")
        ihp = (inp.shape[2] + pad*2)
        iwp = inp.shape[3] + pad*2
        ihp_ = pm.index(0, ihp-1, name="ihp")
        iwp_ = pm.index(0, iwp-1, name="iwp")
        iy = pm.index(0, inp.shape[2]-1, name="iy")
        ix = pm.index(0, inp.shape[3]-1, name="ix")
        padded = pm.temp(shape=(inp.shape[0], inp.shape[1], ihp, iwp))
        padded[b, c, ihp_, iwp_] = 0
        padded[b, c, iy + pad, ix + pad] = inp[b, c, iy, ix]
        out[b, c, y, x] = ((1/(kh*kw)) * pm.sum([m, n], padded[b, c, stride*y + m, stride*x + n]))

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)


class batch_flatten(pm.Template):
    def define_graph(self, data, out):
        out.set_shape((data.shape[0]*data.shape[1]*data.shape[2]*data.shape[3],))
        m = data.shape[1]
        n = data.shape[2]
        p = data.shape[3]

        i = pm.index(0, data.shape[0]-1, name="i")
        j = pm.index(0, m-1, name="j")
        k = pm.index(0, n-1, name="k")
        l = pm.index(0, p-1, name="l")
        out[((i*m + j)*n + k)*p + l] = data[i, j, k, l]

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)



class batch_norm(pm.Template):
    def define_graph(self, x, scale, b, mean, var, out, eps=1e-05, momentum=0.9, spatial=1):
        indices = _get_single_node_indices(out, shape=out.shape)
        if len(out.shape) > 3:
            i = indices[1]
        else:
            i = indices[0]
        out[indices] = scale[i]*(x[indices] - mean[i])/pm.sqrt(var[i] + eps) + b[i]

    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2], self.args[3], self.args[4])

    @property
    def outputs(self):
        return (self.args[5],)


class elem_sigmoid(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.sigmoid(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)


class loop(pm.Template):
    def define_graph(self, v_initial, out, cond=None, max_trip_count=None):
        pass

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class nms(pm.Template):
    def define_graph(self, boxes, scores, out, max_output_boxes_per_class=0, iou_threshold=0, score_threshold=-1, center_point_box=0):
        pass

class elem_where(pm.Template):
    def define_graph(self, condition, x, y, out):
        x_idx, y_idx, indices = _get_elem_indices(x, y, out)
        out[indices] = condition[indices] * x[x_idx] + condition[indices] * y[y_idx]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def false_value(self):
        return self.args[2]

    @property
    def outputs(self):
        return (self.args[3],)


class scatter_elements(pm.Template):
    def define_graph(self, data, indices, updates, out, axis=0):
        pass


    @property
    def inputs(self):
        return (self.args[0], self.args[1], self.args[2])

    @property
    def outputs(self):
        return (self.args[3],)

class elem_cast(pm.Template):
    def define_graph(self, x, out, to=None):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.cast(to, x[indices], shape=out.shape)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_floor(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.floor(x[indices], shape=out.shape)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_ceil(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.ceil(x[indices], shape=out.shape)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_clip(pm.Template):
    def define_graph(self, x, out, minval=None, maxval=None):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.clip(minval, maxval, x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def minval(self):
        return self.minval

    @property
    def maxval(self):
        return self.maxval


class topk(pm.Template):
    def define_graph(self, x, k, out, out_indices, largest=1, sorted=1, axis=-1):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.clip(min, max, x[indices])

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class split(pm.Template):
    def define_graph(self, x, *out, split=None, axis=-1):
        pass

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return self.args[1]


class softmax(pm.Template):
    def define_graph(self, data, out, axis=0):
        out.set_shape(data.shape)
        indices = tuple([pm.index(0, s - 1) for s in data.shape])
        sum_idx = tuple([indices[i] for i in axis])
        out_idx = tuple([indices[i] for i in range(len(indices)) if i not in axis])

        # i = pm.index(0, data.shape[axis]-1, name="i")
        # j = pm.index(0, data.shape[axis]-1, name="j")
        # indices = [pm.index(0, s - 1, name=f"{data.name}[{i}]") for i, s in enumerate(data.shape)]
        # indices_denom = indices
        # indices_denom[axis] = j
        # indices[axis] = i
        # indices = tuple(indices)
        # indices_denom = tuple(indices_denom)
        # mval = pm.max([i], data[indices], name="max_test")
        # e_x = pm.exp((data[indices] - mval), name="e_x")
        # out[indices] = e_x[indices] / pm.sum([indices_denom[axis]], e_x[indices_denom], name="denom")
        e_x = pm.exp((data[indices] ), name="e_x")
        out[indices] = e_x[indices] / pm.sum([sum_idx], e_x[indices])

        # out[indices] =
    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def axis(self):
        return self.kwargs['axis']

    @property
    def outputs(self):
        return (self.args[1],)

class elem_tanh(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.tanh(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)


class elem_if(pm.Template):
    def define_graph(self, condition, out):
        pass
        # a_idx, b_idx, indices = _get_elem_indices(a, b, out)
        # out[indices] = (a[a_idx] == b[b_idx])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[2],)

class elem_exp(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.exp(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_sqrt(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.sqrt(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class elem_log(pm.Template):
    def define_graph(self, x, out):
        indices = _get_single_node_indices(out, shape=out.shape)
        out[indices] = pm.log(x[indices])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class dropout(pm.Template):
    # TODO: Fix and test indices here
    def define_graph(self, x, y, ratio=0.0):
        indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in y.shape])
        y[indices] = x[indices] * 1.0 / (1 - ratio)

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)



class global_avg_pool(pm.Template):
    def define_graph(self, x, out):
        # indices = tuple([pm.index(0, s - 1) if s > 1 else 0 for s in shape])
        indices = _get_single_node_indices(out, shape=out.shape)
        m = pm.index(0, x.shape[2]-1)
        n = pm.index(0, x.shape[3]-1)
        h = x.shape[2]
        w = x.shape[3]
        out[indices] = (1/(h*w)) * pm.sum([m, n], x[indices[0], indices[1], m, n])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

class conv(pm.Template):
    def define_graph(self, data, w, out, stride=1, pad=0, dilation=1, groups=1):

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
        self.kwargs['pad'] = (pad_top, pad_left, pad_down, pad_right)

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
            b = pm.index(0, data.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], w.shape[0], oh, ow))
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((w.shape[0], oh, ow))

        padded = pm.temp(shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0

        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]

        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy * dilation_h + stride * y, dx * dilation_w + stride * x)] * w[c, k, dy, dx]))

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
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        return self.kwargs['pad'][0] + self.kwargs['pad'][2]

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

class bias_add(pm.Template):
    def define_graph(self, data, bias, out):
        assert data.shape[1] == bias.shape[0]
        assert data.shape == out.shape
        N, C, H, W = data.shape
        n = pm.index(0, N - 1)
        c = pm.index(0, C - 1)
        h = pm.index(0, H - 1)
        w = pm.index(0, W - 1)
        out[n,c,h,w] = data[n,c,h,w] + bias[c]

    @property
    def inputs(self):
        return (self.args[0], self.args[1])

    @property
    def outputs(self):
        return (self.args[2],)

class depthwise_conv(pm.Template):
    def define_graph(self, data, w, out, stride=1, pad=0,  groups=1, dilation=1):

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
        oh = (in_height + pad_top + pad_down - dilation_h * (kernel_h - 1) - 1) / stride_h + 1
        ow = (in_width + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1
        oh = int(oh)
        ow = int(ow)

        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1, name="y_")
        x = pm.index(0, ow - 1, name="x_")
        dy = pm.index(0, w.shape[2] - 1, name="dy")
        dx = pm.index(0, w.shape[3] - 1, name="dx")
        iy = pm.index(0, data.shape[-2] - 1, name="iy")
        ix = pm.index(0, data.shape[-1] - 1, name="ix")
        k = pm.index(0, data.shape[-3] - 1, name="k")
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")

        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], w.shape[0], oh, ow))
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((w.shape[0], oh, ow))

        padded = pm.temp(shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0

        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]

        # out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy + stride*y, dx + stride*x)] * w[c, k, dy, dx])) + bias[c]

        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy * dilation_h + stride * y, dx * dilation_w + stride * x)] * w[c, k, dy, dx]))

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
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        return self.kwargs['pad'][0] + self.kwargs['pad'][2]

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']


class depthwise_conv_bias(pm.Template):
    def define_graph(self, data, w, bias, out, stride=1, pad=0,  groups=1, dilation=1):

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
        c = pm.index(0, w.shape[0] - 1, name="c")
        y = pm.index(0, oh - 1, name="y_")
        x = pm.index(0, ow - 1, name="x_")
        dy = pm.index(0, w.shape[2] - 1, name="dy")
        dx = pm.index(0, w.shape[3] - 1, name="dx")
        iy = pm.index(0, data.shape[-2] - 1, name="iy")
        ix = pm.index(0, data.shape[-1] - 1, name="ix")
        k = pm.index(0, data.shape[-3] - 1, name="k")
        ihp = data.shape[-2] + pad_top + pad_down
        iwp = data.shape[-1] + pad_left + pad_right
        ihp_ = pm.index(0, ihp - 1, name="ihp")
        iwp_ = pm.index(0, iwp - 1, name="iwp")
        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0] - 1, name="b")
            o_indices = (b, c)
            p_indices = (b, k,)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], w.shape[0], oh, ow))
        else:
            o_indices = (c,)
            p_indices = (k,)
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((w.shape[0], oh, ow))

        padded = pm.temp(shape=p_shape)
        padded[p_indices + (ihp_, iwp_)] = 0

        padded[p_indices + (iy + pad_top, ix + pad_left)] = data[p_indices + (iy, ix)]


        out[o_indices + (y, x)] = pm.sum([dy, dx, k], (padded[p_indices + (dy * dilation_h + stride * y, dx * dilation_w + stride * x)] * w[c, k, dy, dx])) + bias[c]

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
        return self.kwargs['pad'][0] + self.kwargs['pad'][2]

    @property
    def dilation_int(self):
        return self.kwargs['dilation']

    @property
    def groups(self):
        return self.kwargs['groups']

class lrn(pm.Template):
    def define_graph(self, x, y, alpha, beta, bias, nsize):
        n = pm.index(0, x.shape[0] - 1)
        c = pm.index(0, x.shape[1] - 1)
        h = pm.index(0, x.shape[2] - 1)
        w = pm.index(0, x.shape[3] - 1)
        c_ = pm.index(0, x.shape[1] - 1)
        ext = pm.temp(name="extended", shape=tuple([*x.shape, x.shape[-3]]))

        bounds = pm.output(name="bounds", shape=(x.shape[1], x.shape[1]))
        radius = nsize//2
        hbool = ((((x.shape[1] > (c + radius + 1)) * (c + radius)) + (x.shape[1] <= (c + radius + 1)) * (
                    x.shape[1] - 1)) >= c_)
        lbool = ((((c - radius) > 0) * (c - radius)) + (((c - radius) <= 0) * 0) <= c_)
        bounds[c, c_] = hbool*lbool
        ext[n, c, h, w, c_] = x[n, c_, h, w] * bounds[c, c_]
        # y[n, c, h, w] = x[n,c,h,w] / ((bias + (alpha/nsize) * pm.sum([c_], ext[n, c, h, w, c_]**2))**beta)
        y[n, c, h, w] = x[n,c,h,w] / ((bias + (alpha / nsize) * pm.sum([c_], ext[n, c, h, w, c_]**2)) ** beta)

    @property
    def inputs(self):
        return (self.args[0], self.args[2], self.args[3], self.args[4], self.args[5])

    @property
    def outputs(self):
        return (self.args[1],)



class max_pool(pm.Template):
    def define_graph(self, data, out, kernel_size=None, stride=(1,1), pad=(0,0)):
        if kernel_size is None:
            raise RuntimeError(f"Kernel size is a required parameter with no default value.\n"
                               f"Need to provide value.")
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            kh, kw = kernel_size
        elif isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 2
            kh, kw = kernel_size[0], kernel_size[1]
        else:
            raise RuntimeError(f"Invalid type for kernel size: {type(kernel_size)}, value: {kernel_size}")
        oh = ((data.shape[-2] + 2 * pad[0] - kh) // stride[0] + 1)
        ow = ((data.shape[-1] + 2 * pad[1] - kw) // stride[1] + 1)

        y = pm.index(0, oh-1)
        x = pm.index(0, ow-1)
        m = pm.index(0, kh-1)
        n = pm.index(0, kw-1)
        ihp = (data.shape[-2] + pad[0] * 2)
        iwp = data.shape[-1] + pad[1] * 2
        ihp_ = pm.index(0, ihp-1, name="ihp")
        iwp_ = pm.index(0, iwp-1, name="iwp")
        iy = pm.index(0, data.shape[-2] - 1, name="iy")
        ix = pm.index(0, data.shape[-1] - 1, name="ix")

        if len(data.shape) > 3:
            b = pm.index(0, data.shape[0]-1, name="b")
            c = pm.index(0, data.shape[1] - 1, name="c")

            o_indices = (b,c)
            p_shape = (data.shape[0], data.shape[1], ihp, iwp)
            out.set_shape((data.shape[0], data.shape[1], oh, ow))

        else:
            c = pm.index(0, data.shape[0] - 1, name="c")
            o_indices = (c,)
            p_shape = (data.shape[0], ihp, iwp)
            out.set_shape((data.shape[0], oh, ow))

        padded = pm.temp(shape=p_shape)
        padded[o_indices, ihp_, iwp_] = 0
        padded[o_indices, iy + pad[0], ix + pad[1]] = data[o_indices, iy, ix]
        out[o_indices, y, x] = pm.max([m, n], padded[o_indices, stride[0]*y + m, stride[1]*x + n])

    @property
    def inputs(self):
        return (self.args[0],)

    @property
    def outputs(self):
        return (self.args[1],)

    @property
    def stride(self):
        return self.kwargs['stride']

    @property
    def kernel_size(self):
        return self.kwargs['kernel_size']

    @property
    def pad(self):
        return self.kwargs['pad']

    @property
    def pad_int(self):
        return self.kwargs['pad'][0] + self.kwargs['pad'][2]
