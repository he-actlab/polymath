import polymath as pm
from numbers import Integral
import numpy as np

def get_transpose(data, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.transpose(data, out, shape=shape)
    return out
    # return transpose(data, out, shape=shape)

def get_elem_sigmoid(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return elem_sigmoid(x, out, shape=shape)
    pm.elem_sigmoid(x, out, shape=shape)
    return out

def get_softmax(x, axis=1, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return softmax(x, out, axis=axis, shape=shape)
    pm.softmax(x, out, axis=axis, shape=shape)
    return out

def get_elem_tanh(x, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return elem_tanh(x, out, shape=shape)
    pm.elem_tanh(x, out, shape=shape)
    return out


# TODO: Need to convert this to a node with an output
def get_elem_cast(data, to=None, shape=None, name=None, out=None, **kwargs):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_cast(data, out, to, shape=shape)
    # return pm.cast(to, data[indices], name=name, shape=shape)
    return out

def get_elem_add(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    # return elem_add(a, b, out, shape=shape)
    pm.elem_add(a, b, out, shape=shape)
    return out

def get_elem_sub(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_sub(a, b, out, shape=shape)
    return out

def get_elem_greater(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_greater(a, b, out, shape=shape)
    return out

def get_elem_mul(a, b, shape=None, name=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.elem_mul(a, b, out, shape=shape)
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
    pm.reduce_sum(x, out, shape=shape, axes=axes, **kwargs)
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

def get_batch_norm(x, s, b, mean, var, spatial=None, momentum=None,  epsilon=None, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    pm.batch_norm(x, s, b, mean, var, out, epsilon, momentum, shape=shape)
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
    pm.global_avg_pool(x, out, shape=shape)
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
                (int(pads[0]), int(pads[2])), shape=shape)
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
    pm.max_pool(x, out, kernel_shape[0], kernel_shape[1], (int(strides[0]), int(strides[1])), (int(pads[0]),int(pads[2])), shape=shape)
    return out


def get_dropout(x, ratio=None, training_mode=False, shape=None, name=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    if training_mode:
        pm.dropout(x, out, ratio=ratio, shape=shape)
    else:
        pm.dropout(x, out, shape=shape)
    return out


def get_flatten(x, axis=1, name=None, shape=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.coarse_flatten(x, out, axis=axis, shape=shape)
    return out

def get_gather(data, indices, axis=0, name=None, shape=None, out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    pm.elem_gather(data, indices, out, axis=axis, shape=shape)
    return out

def get_gemm(a, b , c=None, shape=None, name=None, alpha=None,
             beta=None,
             transA=None,
             transB=None,
             out=None):
    if not out:
        out = pm.output(shape=shape, name=name)
    if c:
        pm.gemm(a, b, c, out, shape=shape, alpha=alpha, beta=beta, transA=transA, transB=transB)
    else:
        t_c = pm.temp(shape=shape)
        i = pm.index(0, shape[0]-1)
        j = pm.index(0, shape[1]-1)
        t_c[i, j] = 0
        pm.gemm(a, b, t_c, out, shape=shape, alpha=alpha, beta=beta, transA=transA, transB=transB)
    return out


# TODO: Make range operation
def get_range(start, limit, delta, shape=None, name=None):
    value = np.arange(start, limit, delta)
    assert value.shape == shape
    y = pm.parameter(name=name, shape=shape, default=value)
    return y

# TODO: Fix this to be an actual operation
def get_shape(x, *args, name=None, shape=None, **kwargs):
    x.graph.nodes[name] = x.shape
    return x.graph.nodes[name]

# TODO: Fix this operation
def get_expand(input, shape_input, name=None, shape=None, out=None):
    if not out:
        out = pm.output(name=name, shape=shape)
    return out


NODE_NAMES = {"SVMClassifier": pm.svm_classifier_train,
              "Conv": get_conv,
              "MatMul": get_matmul,
              "Concat": get_concat,
              "GlobalAveragePool": get_global_avg_pool,
              "AveragePool": get_avg_pool,
              "Flatten": get_flatten,
              "LRN": get_lrn,
              "Relu": get_relu,
              "LeakyRelu": get_leaky_relu,
              "BatchNormalization": get_batch_norm,
              "MaxPool": get_max_pool,
              "Gemm": get_gemm,
              "Dropout": get_dropout,
              "Mul": get_elem_mul,
              "Sub": get_elem_sub,
              "Add": get_elem_add,
              "Softmax": get_softmax,
              "Transpose": get_transpose,
              "Sigmoid": get_elem_sigmoid,
              "Tanh": get_elem_tanh,
              "Greater": get_elem_greater,
              "Shape": get_shape,
              "Gather": get_gather,
              "Range": get_range,
              "Expand": get_expand,
              "LinearRegressor": pm.linear_regressor_train,
              "Cast": get_elem_cast,
              "Constant": pm.parameter,
              "Reshape": pm.onnx_reshape,
              "Identity": pm.identity,
              "ReduceSum": get_reduce_sum,
              "Unsqueeze": pm.onnx_unsqueeze,
              "Squeeze": pm.onnx_squeeze,
              "Resize": pm.onnx_resize,
              }