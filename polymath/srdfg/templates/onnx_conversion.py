import polymath as pm
from numbers import Integral
import numpy as np

def get_transpose(data, perm=None, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    if perm is None:
        perm = tuple(list(reversed([i for i in data.shape])))
    pm.tensor_transpose(data, out, perm=perm)
    return out

def get_elem_sigmoid(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
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

def get_topk(x, k, largest=1, sorted=1, axis=-1, shapes=None, name=None, out=None, out_indices=None, **kwargs):
    if not out:
        out = pm.output(name=name[0], shape=shapes[0])
    if not out_indices:
        out_indices = pm.output(name=name[1], shape=shapes[1])
    pm.topk(x, k, out, out_indices, largest=largest, sorted=sorted, axis=axis)
    return out, out_indices


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

def get_scatter(data, indices, updates, axis=0, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.scatter_elements(data, indices, updates, out, axis=axis)
    return out

def get_where(condition, x, y, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_where(condition, x, y, out)
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
    pm.concat(*(inputs + (out,)), axis=axis)
    # indices = [pm.index(0, s - 1) if s > 1 else 0 for s in shape]
    # for idx, i in enumerate(inputs):
    #     indices[axis] = pm.index(idx*i.shape[axis], (idx+1)*i.shape[axis]-1)
    #     j = pm.index(0, i.shape[axis]-1)
    #     out[tuple(indices)] = i[tuple(indices[:axis] + [j] + indices[axis+1:])]
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
    return out

def get_batch_norm(x, s, b, running_mean, running_var, spatial=None, momentum=None,  epsilon=None, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    mean = pm.output(name=f"{x.name}_mean", shape=running_mean.shape)
    var = pm.output(name=f"{x.name}_var", shape=running_var.shape)
    pm.mean_var(x, mean, var, axis=(0,2,3))
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

def get_split(input, split=None, axis=-1, name=None, shapes=None, outputs=None, **kwargs):
    if not outputs:
        outputs = []
        for idx, n in enumerate(name):
            out = pm.output(name=n, shape=shapes[idx])
            outputs.append(out)
    else:
        assert isinstance(outputs, list) and len(outputs) == len(name)
    pm.split(input, *tuple(outputs), split=split, axis=axis)
    return tuple(outputs)

def get_constant_of_shape(input_var, name=None, shape=None, out=None, **kwargs):
    if not out:
        out = pm.state(name=name, shape=shape)
    return out

def get_loop(v_initial, cond=None, max_trip_count=None, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.loop(v_initial, out, cond=cond, max_trip_count=max_trip_count)
    return out

def get_nms(boxes, scores, max_output_boxes_per_class=0, iou_threshold=0, score_threshold=-1, center_point_box=0, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.nms(boxes, scores, out, max_output_boxes_per_class=max_output_boxes_per_class,
           iou_threshold=iou_threshold, score_threshold=score_threshold, center_point_box=center_point_box)
    return out

NODE_NAMES = {
    "Add": get_elem_add,
    "AveragePool": get_avg_pool,
    "And": get_elem_and,
    "BatchNormalization": get_batch_norm,
    "Ceil": get_elem_ceil,
    "Cast": get_elem_cast,
    "Constant": pm.parameter,
    "ConstantOfShape": get_constant_of_shape,
    "Concat": get_concat,
    "Conv": get_conv,
    "ConvTranspose": get_conv_transpose,
    "Clip": get_elem_clip,
    "Div": get_elem_div,
    "Dropout": get_dropout,
    "Exp": get_elem_exp,
    "Equal": get_elem_equal,
    "Expand": get_expand,
    "Flatten": get_flatten,
    "Floor": get_elem_floor,
    "Greater": get_elem_greater,
    "Gemm": get_gemm,
    "Gather": get_gather,
    "GlobalAveragePool": get_global_avg_pool,
    "If": get_elem_if,
    "Identity": pm.identity,
    "Less": get_elem_less,
    "LinearRegressor": pm.linear_regressor_train,
    "LeakyRelu": get_leaky_relu,
    "LRN": get_lrn,
    "Loop": get_loop,
    "Log": get_elem_log,
    "MaxPool": get_max_pool,
    "MatMul": get_matmul,
    "Mul": get_elem_mul,
    "Min": get_elem_min,
    "Max": get_elem_max,
    "Not": get_elem_not,
    "NonZero": get_elem_nonzero,
    "NonMaxSuppression": get_nms,
    "Or": get_elem_or,
    "Pad": get_pad,
    "Relu": get_relu,
    "Range": get_range,
    "RoiAlign": get_roi_align,
    "Reshape": pm.onnx_reshape,
    "ReduceSum": get_reduce_sum,
    "ReduceMin": get_reduce_min,
    "ReduceProd": get_reduce_prod,
    "ReduceMax": get_reduce_max,
    "Resize": pm.onnx_resize,
    "Squeeze": pm.onnx_squeeze,
    "Shape": get_shape,
    "Split": get_split,
    "SoftmaxCrossEntropyLoss": get_cross_entropy_loss,
    "Scatter": get_scatter,
    "ScatterElements": get_scatter_elements,
    "Sqrt": get_elem_sqrt,
    "Sub": get_elem_sub,
    "Slice": get_slice,
    "Softmax": get_softmax,
    "Sigmoid": get_elem_sigmoid,
    "SVMClassifier": pm.svm_classifier_train,
    "Transpose": get_transpose,
    "Tanh": get_elem_tanh,
    "TopK": get_topk,
    "Unsqueeze": pm.onnx_unsqueeze,
    "Where": get_where
}
