import polymath as pm
from numbers import Integral
import numpy as np

def get_transpose(data, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.transpose(data, out)
    return out
    # return transpose(data, out, shape=shape)

def get_elem_sigmoid(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return elem_sigmoid(x, out, shape=shape)
    pm.elem_sigmoid(x, out)
    return out

def get_softmax(x, axis=1, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.softmax(x, out, axis=axis)
    return out

def get_elem_tanh(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_tanh(x, out)
    return out

def get_elem_sqrt(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_sqrt(x, out)
    return out

def get_elem_log(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_log(x, out)
    return out

def get_elem_exp(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_exp(x, out)
    return out

def get_topk(x, k, largest=1, sorted=1, axis=-1, shape=None, name=None, out=None, out_indices=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    if not out_indices:
        out_indices = pm.output(name=name, shape=shape)
    pm.topk(x, k, out, out_indices, largest=largest, sorted=sorted, axis=axis)
    return out


# TODO: Need to convert this to a node with an output
def get_elem_cast(data, to=None, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_cast(data, out, to)
    return out

def get_elem_floor(data, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_floor(data, out)
    return out

def get_elem_clip(data, min=None, max=None, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_clip(data, out, min=min, max=max)
    return out

def get_elem_ceil(data, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_ceil(data, out)
    return out

def get_elem_add(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return elem_add(a, b, out, shape=shape)
    pm.elem_add(a, b, out)
    return out

def get_elem_min(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_min(a, b, out)
    return out

def get_elem_max(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_min(a, b, out)
    return out

def get_elem_div(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_div(a, b, out)
    return out

def get_elem_sub(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_sub(a, b, out)
    return out

def get_scatter_elements(data, indices, updates, axis=0, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.scatter_elements(data, indices, updates, out, axis=axis)
    return out

def get_where(condition, x, y, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_sub(a, b, out)
    return out

def get_elem_greater(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_greater(a, b, out)
    return out

def get_elem_not(a, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_not(a, out)
    return out

def get_elem_or(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_or(a, b, out)
    return out

def get_elem_and(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_and(a, b, out)
    return out

def get_elem_equal(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_equal(a, b, out)
    return out

def get_elem_nonzero(a, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_nonzero(a, out)
    return out

def get_elem_if(condition, else_branch=None, then_branch=None, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_nonzero(condition, out)
    return out

def get_elem_less(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_less(a, b, out)
    return out

def get_elem_mul(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_mul(a, b, out)
    return out


def get_reduce_sum(x, shape=None, name=None, out=None, axes=(0,), **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    if isinstance(axes, Integral):
        axes = (axes,)
    elif isinstance(axes, list):
        axes = tuple(axes)
    else:
        assert isinstance(axes, tuple)
    pm.reduce_sum(x, out, axes=axes)
    return out

def get_reduce_prod(x, shape=None, name=None, out=None, axes=(0,), **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    if isinstance(axes, Integral):
        axes = (axes,)
    elif isinstance(axes, list):
        axes = tuple(axes)
    else:
        assert isinstance(axes, tuple)
    pm.reduce_prod(x, out, axes=axes)
    return out

def get_reduce_min(x, shape=None, name=None, out=None, axes=(0,), **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    if isinstance(axes, Integral):
        axes = (axes,)
    elif isinstance(axes, list):
        axes = tuple(axes)
    else:
        assert isinstance(axes, tuple)
    pm.reduce_min(x, out, axes=axes)
    return out

def get_reduce_max(x, shape=None, name=None, out=None, axes=(0,), **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    if isinstance(axes, Integral):
        axes = (axes,)
    elif isinstance(axes, list):
        axes = tuple(axes)
    else:
        assert isinstance(axes, tuple)
    pm.reduce_max(x, out, axes=axes)
    return out

def get_matmul(a, b, out=None, **kwargs):

    if not out:
        out = pm.output(shape=kwargs['shape'], name=kwargs['name'])
    pm.matmul(a, b, out)
    return out

def get_elem(a, b, **kwargs):

    if len(a.shape) == len(b.shape):
        return pm.matmul(a, b, **kwargs)
    elif len(a.shape) > len(b.shape):
        return pm.rvmatmul(a, b, **kwargs)
    else:
        return pm.lvmatmul(a, b, **kwargs)

def get_lrn(x, alpha=None, beta=None, bias=None, size=None, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.lrn(x, out, alpha=alpha, beta=beta, bias=bias, nsize=size)
    return out

# TODO: Add concat to transformations
def get_concat(*inputs, axis=None, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    indices = [pm.index(0, s - 1) if s > 1 else 0 for s in shape]
    for idx, i in enumerate(inputs):
        indices[axis] = pm.index(idx*i.shape[axis], (idx+1)*i.shape[axis]-1)
        j = pm.index(0, i.shape[axis]-1)
        out[tuple(indices)] = i[tuple(indices[:axis] + [j] + indices[axis+1:])]
    return out

def get_conv_transpose(x, w, bias=None, dilations=None, group=None, kernel_shape=None, pads=None, auto_pad=None,
             output_padding=None,
             strides=None,
             shape=None,
             name=None,
             out=None):
    if not out:
        out = pm.output(shape=shape, name=name)

    if auto_pad:

        h_out = np.ceil(x.shape[-2] / strides[0])
        w_out = np.ceil(x.shape[-1] / strides[1])
        ph = max(0, (h_out - 1) * strides[0] + kernel_shape[0] - x.shape[-2])
        pw = max(0, (w_out - 1) * strides[1] + kernel_shape[1] - x.shape[-1])
        pads = [0,0,0,0]
        if auto_pad == "SAME_LOWER":
            pads[0] = np.floor(ph//2)
            pads[1] = ph - pads[0]
            pads[2] = np.floor(pw//2)
            pads[3] = pw - pads[2]
        elif auto_pad == "SAME_UPPER":
            pads[1] = np.floor(ph // 2)
            pads[0] = ph - pads[1]
            pads[3] = np.floor(pw // 2)
            pads[2] = pw - pads[3]

    if bias:
        pm.conv_transpose_bias(x, w, bias, out, int(strides[0]), int(pads[-2]), out_pad=output_padding)
        return out
    else:
        pm.conv_transpose(x, w, out, int(strides[0]), int(pads[-2]), out_pad=output_padding)
        return out


def get_conv(x, w, bias=None, dilations=None, group=None, kernel_shape=None, pads=None, auto_pad=None,
             strides=None,
             shape=None,
             name=None,
             out=None):
    if not out:
        out = pm.output(shape=shape, name=name)

    if auto_pad:

        h_out = np.ceil(x.shape[-2] / strides[0])
        w_out = np.ceil(x.shape[-1] / strides[1])
        ph = max(0, (h_out - 1) * strides[0] + kernel_shape[0] - x.shape[-2])
        pw = max(0, (w_out - 1) * strides[1] + kernel_shape[1] - x.shape[-1])
        pads = [0,0,0,0]
        if auto_pad == "SAME_LOWER":
            pads[0] = np.floor(ph//2)
            pads[1] = ph - pads[0]
            pads[2] = np.floor(pw//2)
            pads[3] = pw - pads[2]
        elif auto_pad == "SAME_UPPER":
            pads[1] = np.floor(ph // 2)
            pads[0] = ph - pads[1]
            pads[3] = np.floor(pw // 2)
            pads[2] = pw - pads[3]

    if bias:
        pm.conv_bias(x, w, bias, out, int(strides[0]), int(pads[-2]))
        return out
    else:
        pm.conv(x, w, out, int(strides[0]), int(pads[-2]))
        return out

def get_roi_align(x, rois, batch_indices, mode='avg',
                  output_height=1, output_width=1,
                  sampling_ratio=0, spatial_scale=1.0,
                  name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.roi_align(x, rois, batch_indices, out, mode=mode,
                  output_height=output_height, output_width=output_width,
                  sampling_ratio=sampling_ratio, spatial_scale=spatial_scale)

def get_batch_norm(x, s, b, mean, var, spatial=None, momentum=None,  epsilon=None, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.batch_norm(x, s, b, mean, var, out, epsilon, momentum)
    return out

def get_relu(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.relu(x, out)
    return out

def get_leaky_relu(x, alpha=0.01, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.leaky_relu(x, out, alpha=alpha)
    return out

def get_global_avg_pool(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.global_avg_pool(x, out)
    return out

def get_avg_pool(x, auto_pad=None, ceil_mode=0, kernel_shape=None, pads=None,
                 strides=None,
                 shape=None,
                 name=None,
                 out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    if auto_pad:
        if ceil_mode == 0:
            h_out = np.floor(x.shape[-2] / strides[0])
            w_out = np.floor(x.shape[-1] / strides[1])
        else:
            h_out = np.ceil(x.shape[-2] / strides[0])
            w_out = np.ceil(x.shape[-1] / strides[1])
        ph = max(0, (h_out - 1) * strides[0] + kernel_shape[0] - x.shape[-2])
        pw = max(0, (w_out - 1) * strides[1] + kernel_shape[1] - x.shape[-1])
        pads = [0, 0, 0, 0]
        if auto_pad == "SAME_LOWER":
            pads[0] = np.floor(ph // 2)
            pads[1] = ph - pads[0]
            pads[2] = np.floor(pw // 2)
            pads[3] = pw - pads[2]
        elif auto_pad == "SAME_UPPER":
            pads[1] = np.floor(ph // 2)
            pads[0] = ph - pads[1]
            pads[3] = np.floor(pw // 2)
            pads[2] = pw - pads[3]

    pm.avg_pool(x, out, kernel_shape[0], kernel_shape[1], (int(strides[0]), int(strides[1])),
                (int(pads[0]), int(pads[2])))
    return out

def get_max_pool(x, ceil_mode=0, kernel_shape=None, pads=None, auto_pad=None,
                 strides=None,
                 shape=None,
                 name=None,
                 out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    int_fn = np.ceil if ceil_mode != 0 else np.floor
    if auto_pad:
        h_out = int_fn(x.shape[-2] / strides[0])
        w_out = int_fn(x.shape[-1] / strides[1])
        ph = max(0, (h_out - 1) * strides[0] + kernel_shape[0] - x.shape[-2])
        pw = max(0, (w_out - 1) * strides[1] + kernel_shape[1] - x.shape[-1])
        pads = [0,0,0,0]
        if auto_pad == "SAME_LOWER":
            pads[0] = np.floor(ph//2)
            pads[1] = ph - pads[0]
            pads[2] = np.floor(pw//2)
            pads[3] = pw - pads[2]
        elif auto_pad == "SAME_UPPER":
            pads[1] = np.floor(ph // 2)
            pads[0] = ph - pads[1]
            pads[3] = np.floor(pw // 2)
            pads[2] = pw - pads[3]
    pm.max_pool(x, out, kernel_shape[0], kernel_shape[1], (int(strides[0]), int(strides[1])), (int(pads[0]),int(pads[2])))
    return out


def get_dropout(x, ratio=None, training_mode=False, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    if training_mode:
        pm.dropout(x, out, ratio=ratio)
    else:
        pm.dropout(x, out)
    return out


def get_flatten(x, axis=1, name=None, shape=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.coarse_flatten(x, out, axis=axis)
    return out

def get_gather(data, indices, axis=0, name=None, shape=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.elem_gather(data, indices, out, axis=axis)
    return out

def get_cross_entropy_loss(scores, labels, ignore_index=-100, reduction="mean", name=None, shape=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.cross_entropy_loss(scores, labels, out, reduction=reduction)
    return out

def get_gemm(a, b , c=None, shape=None, name=None, alpha=None,
             beta=None,
             transA=None,
             transB=None,
             out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    if transB:
        assert len(b.shape) == 2
        b.shape = (b.shape[1], b.shape[0])
        transB = False
    if c:
        pm.gemm(a, b, c, out, alpha=alpha, beta=beta, transA=transA, transB=transB, strict_shapes=True)
    else:
        t_c = pm.temp(shape=shape)
        i = pm.index(0, shape[0]-1)
        j = pm.index(0, shape[1]-1)
        t_c[i, j] = 0
        pm.gemm(a, b, t_c, out, alpha=alpha, beta=beta, transA=transA, transB=transB, strict_shapes=True)
    return out


# TODO: Make range operation
def get_range(start, limit, delta, shape=None, name=None):
    value = np.arange(start, limit, delta)
    assert value.shape == shape
    y = pm.parameter(name=name, shape=shape, default=value)
    return y

# TODO: Fix this to be an actual operation
def get_shape(x, *args, name=None, shape=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    return out

# TODO: Fix this operation
def get_expand(input, shape_input, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    return out

def get_pad(input, shape_input, name=None, shape=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    return out

def get_slice(input, starts, ends, axes=-1, steps=1, name=None, shape=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    return out

def get_split(input, split=None, axis=-1, name=None, shape=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.split(input, out, split=split, axis=axis)
    return out

def get_constant_of_shape(input_var, name=None, shape=None, out=None, **kwargs):
    if not out:
        out = pm.state(name=name, shape=shape)
    return out

def get_loop(v_initial, cond=None, max_trip_count=None, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.loop(v_initial, out, cond=cond, max_trip_count=max_trip_count)
    return out



NODE_NAMES = {
            "aten::sub": get_elem_sub,
            "aten::sub_": get_elem_sub,
            "aten::max": get_elem_max,
            "aten::min": get_elem_min,
            "aten::mul": get_elem_mul,
            "aten::mul_": get_elem_mul,
            "aten::div": get_elem_div,
            "aten::div_": get_elem_div,
            "aten::true_divide": get_elem_div,
            "aten::where": get_where,
            "aten::topk": get_topk,
            "aten::relu": get_relu,
            "aten::relu_": get_relu,
            "aten::max_pool2d": get_max_pool,
            "aten::_convolution": get_conv,
            "aten::softmax": get_softmax,
            "aten::batch_norm": get_batch_norm,
            "aten::transpose": get_transpose,
            "aten::transpose_": get_transpose,
            "aten::t": get_transpose,
            "aten::sigmoid": get_elem_sigmoid,
            "aten::avg_pool2d": get_avg_pool,
            "aten::linear": get_matmul,
            "aten::matmul": get_matmul,
            "aten::bmm": get_matmul,
            "aten::expand": get_expand,
            "aten::sum": get_reduce_sum,
            "aten::prod": get_reduce_prod,
            "aten::tanh": get_elem_tanh,
            "aten::log": get_elem_log,
            "aten::exp": get_elem_exp,
            "aten::sqrt": get_elem_sqrt,
            "aten::ceil": get_elem_ceil,
            "aten::floor": get_elem_floor,
            "aten::expand_as": get_expand,
            "aten::lt": get_elem_less,
            "aten::gt": get_elem_greater,
            "aten::eq": get_elem_equal,
            "aten::logical_not": get_elem_not,
            "aten::mm": get_matmul,
            "aten::add": get_elem_add,
            "aten::add_": get_elem_add,
            "aten::gather": get_gather,
            "torchvision::nms": self.nms,
            "torchvision::roi_align": get_roi_align,
            "aten::__and__": get_elem_and,
            "aten::logical_and": get_elem_and,
            "aten::nonzero": get_elem_nonzero,
            "aten::scatter": get_scatter_elements,
            "aten::__not__": get_elem_not,
            # "aten::scalar_tensor": self.scalar_tensor,
            # "aten::tensor": get_,  # used for example in tensor(1.0)
    # "aten::arange": self.arange,
    # "aten::meshgrid": self.meshgrid,
    # "aten::floor_divide": self.make_elemwise("floor_divide"),
    # "aten::floor_divide_": self.make_elemwise("floor_divide"),
    # "aten::addcdiv": self.addcdiv,
    # "aten::addcmul": self.addcmul,
    # "aten::ones": self.ones,
    # "aten::ones_like": self.ones_like,
    # "aten::zeros": self.zeros,
    # "aten::zeros_like": self.zeros_like,
    # "aten::full": self.full,
    # "aten::full_like": self.full_like,
    # "aten::linspace": self.linspace,
    # "aten::reciprocal": self.reciprocal,
    # "aten::repeat": self.repeat,
    # "aten::repeat_interleave": self.repeat_interleave,
    # "aten::to": self.to,
    # "aten::squeeze": self.squeeze,
    # "aten::unsqueeze": self.unsqueeze,
    # "aten::unsqueeze_": self.unsqueeze,
    # "aten::cat": self.concatenate,
    # "aten::slice": self.slice,
    # "aten::narrow": self.narrow,
    # "aten::split": self.split,
    # "aten::split_with_sizes": self.split_with_sizes,
    # "aten::select": self.select,
    # "aten::take": get_t,
    # "aten::prelu": self.prelu,
    # "aten::leaky_relu": self.leaky_relu,
    # "aten::leaky_relu_": self.leaky_relu,
    # "aten::elu": self.elu,
    # "aten::elu_": self.elu,
    # "aten::celu": self.celu,
    # "aten::gelu": self.gelu,
    # "aten::selu": self.selu,
    # "aten::log_sigmoid": self.log_sigmoid,
    # "aten::adaptive_avg_pool2d": self.adaptive_avg_pool_2d,
    # "aten::adaptive_max_pool2d": self.adaptive_max_pool_2d,
    # "aten::max_pool2d_with_indices": self.maxpool_2d_with_indices,
    # "aten::max_pool1d": self.maxpool_1d,
    # "aten::max_pool3d": self.maxpool_3d,
    # "aten::hardtanh": self.hardtanh,
    # "aten::hardtanh_": self.hardtanh,
    # "aten::threshold": self.threshold,
    # "aten::threshold_": self.threshold,
    # "aten::contiguous": self.contiguous,
    # "aten::instance_norm": self.instance_norm,
    # "aten::layer_norm": self.layer_norm,
    # "aten::group_norm": self.group_norm,
    # "aten::softplus": self.softplus,
    # "aten::log_softmax": self.log_softmax,
    # "aten::clone": self.clone,
# "aten::flatten": self.flatten,
#             "aten::addmm": self.addmm,
#             "aten::size": self.size,
#             "aten::view": self.view,
#             "aten::reshape": self.reshape,
    # "aten::avg_pool1d": self.make_avg_pool(1),
    # "aten::avg_pool3d": self.make_avg_pool(3),
    # "aten::dropout": self.dropout,
    # "aten::dropout_": self.dropout,
    # "aten::feature_dropout": self.dropout,
    # "aten::alpha_dropout": self.dropout,
    # "aten::mean": self.mean,
    # "aten::chunk": self.chunk,
    # "aten::Int": self.int,
    # "prim::NumToTensor": self.numtotensor,
    # "prim::ImplicitTensorToNum": self.tensortonum,
    # "aten::ScalarImplicit": self.tensortonum,
    # "aten::constant_pad_nd": self.make_pad("constant"),
    # "aten::reflection_pad1d": self.make_pad("reflect"),
    # "aten::reflection_pad2d": self.make_pad("reflect"),
    # "aten::replication_pad1d": self.make_pad("edge"),
    # "aten::replication_pad2d": self.make_pad("edge"),
    # "aten::replication_pad3d": self.make_pad("edge"),
    # "aten::permute": self.transpose,
    # "aten::argmin": self.make_reduce("argmin"),
    # "aten::argmax": self.make_reduce("argmax"),
    # "aten::norm": self.norm,
    # "aten::frobenius_norm": self.frobenius_norm,
    # "aten::std": self.std,
    # "aten::var": self.variance,
    # "aten::abs": self.make_unary("abs"),
    # "aten::neg": self.make_unary("negative"),
    # "aten::cos": self.make_unary("cos"),
    # "aten::cosh": self.make_unary("cosh"),
    # "aten::sin": self.make_unary("sin"),
    # "aten::sinh": self.make_unary("sinh"),
    # "aten::tan": self.make_unary("tan"),
    # "aten::acos": self.make_unary("acos"),
    # "aten::asin": self.make_unary("asin"),
    # "aten::atan": self.make_unary("atan"),
    # "aten::log1p": self.log1p,
    # "aten::log2": self.make_unary("log2"),
    # "aten::log10": None,
    # "aten::round": get_elem_,
    # "aten::rsqrt": get_elem_r,
    # "aten::sign": self.make_unary("sign"),
    # "aten::erf": self.make_unary("erf"),
    # "aten::trunc": self.make_unary("trunc"),
    # "aten::unbind": self.unbind,
    # "aten::isfinite": self.make_unary("isfinite"),
    # "aten::isinf": self.make_unary("isinf"),
    # "aten::isnan": self.make_unary("isnan"),
    # "aten::clamp": self.clamp,
    # "aten::clamp_": get_c,
    # "aten::detach": self.identity,
    # "aten::upsample_bilinear2d": self.make_upsample("bilinear"),
    # "aten::upsample_nearest2d": self.make_upsample("nearest_neighbor"),
    # "aten::upsample_trilinear3d": self.make_upsample3d("trilinear"),
    # "aten::upsample_nearest3d": self.make_upsample3d("nearest_neighbor"),
    # "aten::logical_xor": self.logical_xor,
    # "aten::ne": get_elem_,
    # "aten::le": self.make_elemwise("less_equal"),
    # "aten::ge": self.make_elemwise("greater_equal"),
    # "aten::bitwise_not": self.bitwise_not,
    # "aten::bitwise_xor": self.bitwise_xor,
    # "aten::Bool": self.Bool,
    # "aten::Float": self.Float,
    # "aten::pow": get_elem_p,
    # "aten::len": self.list_len,
    # "torchvision::deform_conv2d": self.deform_conv2d,
    # "aten::index": self.index,
    # "aten::logsumexp": self.logsumexp,
    # "aten::index_select": self.select,
    # "aten::type_as": self.type_as,
    # "aten::__getitem__": self.list_getitem,
    # "aten::stack": self.stack,
    # "aten::one_hot": self.one_hot,
    # "aten::embedding": self.embedding,
    # "aten::rsub": self.rsub,
    # "aten::adaptive_avg_pool3d": self.adaptive_avg_pool_3d,
    # "aten::adaptive_max_pool3d": self.adaptive_max_pool_3d,
    # "aten::_shape_as_tensor": self.shape_as_tensor,
            # "aten::__interpolate": self.interpolate,
            # "aten::nonzero_numpy": self.nonzero_numpy,
            # "aten::index_put": self.index_put,
            # "aten::index_put_": self.index_put,
            # "aten::IntImplicit": self.identity,
            # "aten::numel": self.numel,
            # "aten::empty": self.empty,
            # "aten::bincount": self.bincount,
            # "aten::scatter_add": self.scatter_add,
            # "aten::hardswish_": self.hard_swish,
            # "aten::hardswish": self.hard_swish,
            # "aten::hardsigmoid_": self.hard_sigmoid,
            # "aten::hardsigmoid": self.hard_sigmoid,
            # "aten::cumsum": self.cumsum,
            # "aten::masked_fill": self.masked_fill,
            # "aten::masked_select": self.masked_select,
            # "aten::argsort": self.argsort,
            # "aten::sort": self.sort,
            # "aten::_unique2": self.unique,
            # "aten::is_floating_point": is_floating_point,
            # "aten::pixel_shuffle": pixel_shuffle,
            # "aten::device": self.none,
            # "prim::device": self.none,
        }
