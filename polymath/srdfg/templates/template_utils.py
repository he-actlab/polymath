import polymath as pm
import numpy as np

def format_idx(x, reverse=True):
    if reverse:
        return tuple(list(reversed(x)))
    else:
        return tuple(x)

def _get_indices(node, all_indices, tgt_shape):
    indices = []

    if node.shape == pm.DEFAULT_SHAPES[0]:
        return tuple(indices)

    for idx, i in enumerate(all_indices):
        if len(node.shape) > idx and tgt_shape[idx] == node.shape[idx]:
            indices.append(i)
    if tgt_shape != node.shape:
        for idx, i in enumerate(node.shape):
            if i != tgt_shape[idx]:
                indices.insert(idx, 0)
    return tuple(indices)


def _get_binop_idx(node_a, node_b, out_node):
    # TODO: Figure out what to do about multiple dimensions with the same value
    cnt = 0
    op1 = []
    op2 = []
    all_ops = []

    for i in node_a.shape:
        if i == 1:
            op1.append(0)
            # all_ops.append(0)
        else:
            idx = pm.index(0, i - 1)
            op1.append(idx)
            all_ops.append(idx)
            cnt += 1

    for i in node_b.shape:
        if i in node_a.shape:
            idx = node_a.shape.index(i)
            op2.append(op1[idx])
        elif i == 1:
            op2.append(0)
            # all_ops.append(0)
        else:
            idx = pm.index(0, i - 1)
            op2.append(idx)
            all_ops.append(idx)
            cnt += 1
    if out_node.is_shape_finalized():
        all_ops = []
        for s in out_node.shape:
            if s in node_a.shape:
                idx = node_a.shape.index(s)
                all_ops.append(idx)
            else:
                assert s in node_b.shape, f"Output shape value {s} not in other shapes"
                idx = node_b.shape.index(s)
                all_ops.append(idx)

    return op1, op2, all_ops


def _get_single_node_indices(node, shape=None):
    if node.shape == pm.DEFAULT_SHAPES[0]:
        return tuple([])
    else:
        if not shape:
            shape = node.shape
        indices = tuple([pm.index(0, s - 1) for s in shape])
        return indices


def _get_reduce_node_indices(a, b, output, axis):
    if output.shape == pm.DEFAULT_SHAPES[0]:
        return tuple([])
    else:
        if not output.shape:
            raise RuntimeError
        indices = tuple([pm.index(0, s - 1) for s in output.shape])
        return indices


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


# Use numpy broadcasting rules
def _get_elem_indices(node_a, node_b, node_c, zero_indices=True):
    broadcastable = is_broadcastable(node_a.shape, node_b.shape)

    a_idx = []
    b_idx = []
    out_idx = []
    nmap = {}
    reverse = True

    if not broadcastable:
        reverse = False

        a_idx = [None] * len(node_a.shape)
        b_idx = [None] * len(node_b.shape)
        a_map = {}
        b_map = {}
        for s in node_c.shape:
            idx = pm.index(0, s - 1)
            out_idx.append(idx)
            if s in node_a.shape:
                start = 0
                if s in a_map:
                    start = a_map[s]
                sidx = node_a.shape.index(s, start)
                a_idx[sidx] = idx
                a_map[s] = sidx

            if s in node_b.shape:
                start = 0
                if s in b_map:
                    start = b_map[s]
                sidx = node_b.shape.index(s, start)
                b_idx[sidx] = idx
                b_map[s] = sidx

        for i in range(len(a_idx)):
            if a_idx[i] is None:
                assert node_a.shape[i] == 1
                a_idx[i] = 0

        for i in range(len(b_idx)):
            if b_idx[i] is None:
                assert node_b.shape[i] == 1
                b_idx[i] = 0

    else:
        if node_a.shape == node_b.shape and node_c.shape == node_a.shape:
            indices = _get_single_node_indices(node_a)
            return indices, indices, indices
        elif node_a.shape == pm.DEFAULT_SHAPES[0] and node_b.shape == pm.DEFAULT_SHAPES[0]:
            idx = format_idx([])
            return idx, idx, idx
        elif node_a.shape == pm.DEFAULT_SHAPES[0]:
            idx = format_idx([])
            indices = _get_single_node_indices(node_b)
            return idx, indices, indices
        elif node_b.shape == pm.DEFAULT_SHAPES[0]:
            idx = format_idx([])
            indices = _get_single_node_indices(node_a)
            return indices, idx, indices

        if len(node_a.shape) > len(node_b.shape):
            small_node = node_b
            lg_node = node_a
            nmap["small"] = b_idx
            nmap["large"] = a_idx
        else:
            small_node = node_a
            lg_node = node_b
            nmap["small"] = a_idx
            nmap["large"] = b_idx

        for i in range(-1, -len(lg_node.shape) - 1, -1):
            if len(small_node.shape) < abs(i):
                idx = pm.index(0, lg_node.shape[i] - 1)
                nmap["large"].append(idx)
                out_idx.append(idx)
            elif node_a.shape[i] == node_b.shape[i]:
                if node_a.shape[i] != 1:
                    idx = pm.index(0, node_a.shape[i] - 1)
                    a_idx.append(idx)
                    b_idx.append(idx)
                    out_idx.append(idx)
            elif node_a.shape[i] == 1:
                idx = pm.index(0, node_b.shape[i] - 1)
                if zero_indices:
                    a_idx.append(0)  # TESTING
                b_idx.append(idx)
                out_idx.append(idx)
            elif node_b.shape[i] == 1:
                idx = pm.index(0, node_a.shape[i] - 1)
                a_idx.append(idx)
                if zero_indices:
                    b_idx.append(0)  # TESTING
                out_idx.append(idx)
            else:
                raise RuntimeError(f"Unable to broadcast indices:\n"
                                   f"{node_a.name}: {node_a.shape}\n"
                                   f"{node_b.name}: {node_b.shape}\n")
    return format_idx(a_idx, reverse), format_idx(b_idx, reverse), format_idx(out_idx, reverse)

def dilate(var: pm.placeholder, strides, name=None):
    n = len(var.shape)
    assert len(strides) == n
    out_shape = ()
    nz_indices = ()
    shape_idx = ()

    for i in range(n):
        out_shape += ((var.shape[i] - 1) * strides[i] + 1,)
        nz_indices += (pm.index(0, out_shape[i] - 1, stride=strides[i]),)
        shape_idx += (pm.index(0, out_shape[i] - 1),)

    padded = pm.temp(name=name, shape=out_shape)
    padded[shape_idx] = 0
    padded[(shape_idx[0])] = 0

# def get_pad_tuple(pad_size):
#     if isinstance(pad_size, (tuple, list)):
#         if len(pad_size) == 2:
#             pad_h = pad_size[0] * 2
#             pad_w = pad_size[1] * 2
#         elif len(pad_size) == 4:
#             return pad_size[0], pad_size[2], pad_size[1], pad_size[3]
#         else:
#             raise ValueError("Size of padding can only be 2 or 4")
#     else:
#         assert isinstance(pad_size, int)
#         pad_h = pad_w = pad_size * 2
#
#     pad_top = (pad_h + 1) // 2
#     pad_left = (pad_w + 1) // 2
#     return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def get_pad_tuple(padding, kernel):
    """Common code to get the pad option
    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    kernel : tuple of int
        Conv kernel size
    Returns
    -------
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # pad_h = pad_w = padding * 2
    if len(padding) == 4:
        return padding[0], padding[1], padding[2], padding[3]
    else:
        pad_h = padding[0] * 2
        pad_w = padding[1] * 2
        pad_top = (pad_h + 1) // 2
        pad_left = (pad_w + 1) // 2
        return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left

def pad_node(data: pm.Node, padded_out: pm.Node, pad_size, kernel, pad_val=0):
    assert len(data.shape) == 4
    p_top, p_bottom, p_left, p_right = get_pad_tuple(pad_size, kernel)
    oh = data.shape[2] + p_top + p_bottom
    ow = data.shape[3] + p_left + p_right

    padded_shape = (data.shape[0], data.shape[1], oh, ow)
    if padded_out.is_shape_finalized() and padded_out.shape != (1,):
        assert padded_shape == padded_out.shape, f"Unequal shapes for padding:\n" \
                                                 f"Target shape: {padded_shape}\n" \
                                                 f"Set shape: {padded_out.shape}"
    padded_out.set_shape(padded_shape)
    n_idx = pm.index(0, data.shape[0]-1)
    c_idx = pm.index(0, data.shape[1]-1)
    oh_idx = pm.index(0, oh-1)
    ih_idx = pm.index(0, data.shape[2]-1)
    ow_idx = pm.index(0, ow-1)
    iw_idx = pm.index(0, data.shape[3] - 1)
    padded_out[(n_idx, c_idx, oh_idx, ow_idx)] = pad_val
    padded_out[(n_idx, c_idx, ih_idx + p_top, iw_idx + p_left)] = data[(n_idx, c_idx, ih_idx, iw_idx)]
    return padded_out

def reshape_node(data: pm.Node, reshaped_out: pm.Node, shape: tuple, dim_combinations):
    assert np.prod(data.shape) == np.prod(shape)
    assert len(dim_combinations) == len(shape)
    src_indices = []
    dst_indices = []

    for s in data.shape:
        idx = pm.index(0, s-1)
        src_indices.append(idx)

    # STEP 0: idx3*1 + 0
    # STEP 1: idx3 + shape[3]*
    for dc in reversed(dim_combinations):
        idx = 0
        idx_offset = 1
        add_dim = 0
        for d in reversed(dc):
            idx = src_indices[d]*idx_offset + add_dim
            idx_offset = data.shape[d]

def _get_indices_for_dim(x, dim):
    assert len(x.shape) < dim
    idx = pm.index(0, x.shape[dim] - 1)
    return idx


def _dim_explicit(a_shp, dim):
    if dim is None:
        return dim

    if dim < 0:
        dim = len(a_shp) + dim
    return dim


def _get_conv_shape_1axis(
    image_shape, kernel_shape, border_mode, subsample, dilation=1
):
    """This function compute the output shape of convolution operation.
    Copied and simplified from theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py
    Parameters
    ----------
    image_shape: int
        Corresponds to the input image shape on a given axis.
    kernel_shape: int
        Corresponds to the kernel shape on a given axis.
    border_mode: string or int. If it is a string, it must be
        'valid' or 'full'.
    subsample: int. It must correspond to the subsampling on the
        considered axis.
    dilation: int. It must correspond to the dilation on the
        considered axis.
    Returns
    -------
    out_shp: int corresponding to the output image shape on the
        considered axis.
    """
    # Implicit dilated kernel shape
    dil_kernel_shape = (kernel_shape - 1) * dilation + 1
    if border_mode == "full":
        pad_l = pad_r = dil_kernel_shape - 1
    elif border_mode == "valid":
        pad_l = pad_r = 0
    else:
        assert border_mode >= 0
        pad_l = pad_r = border_mode

    # In case of symbolic shape, we want to build the smallest graph
    # (image_shape + 2 * pad - dil_kernel_shape) // subsample + 1
    out_shp = image_shape - dil_kernel_shape
    if pad_l != 0:
        out_shp += pad_l
    if pad_r != 0:
        out_shp += pad_r
    if subsample != 1:
        out_shp = out_shp // subsample
    out_shp = out_shp + 1

    return out_shp

def _get_conv_output_shape(
    image_shape, kernel_shape, border_mode, subsample, filter_dilation=(0, 0)
):
    """This function compute the output shape of convolution operation.
    Copied and simplified from Theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py
    Parameters
    ----------
    image_shape: tuple of int corresponding to the input
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of input channels, height and width (and
        possibly depth) of the image. None where undefined.
    kernel_shape: tuple of int corresponding to the
        kernel shape. For a normal convolution, its four (for 2D convolution)
        or five (for 3D convolution) elements must correspond respectively to :
        number of output channels, number of input channels, height and width
        (and possibly depth) of the kernel.
        For an unshared 2D convolution, its six channels must correspond to :
        number of output channels, height and width of the output, number of
        input channels, height and width of the kernel.
        None where undefined.
    border_mode: string, or tuple of int. If it is a string, it must be 'valid'
        or 'full'. If it is a tuple, its two (or three) elements respectively
        correspond to the padding on height and width (and possibly depth)
        axis.
    subsample: tuple of int. Its two or three elements
        respectively correspond to the subsampling on height and width (and
        possibly depth) axis.
    filter_dilation: tuple of int. Its two or three
        elements correspond respectively to the dilation on height and width axis.
    Returns
    -------
    output_shape: tuple of int corresponding to the output image shape. Its
        four element must correspond respectively to: batch size, number of
        output channels, height and width of the image.
    """
    bsize, imshp = image_shape[0], image_shape[2:]

    convdim = len(image_shape) - 2
    nkern, kshp = kernel_shape[0], kernel_shape[-convdim:]

    if isinstance(border_mode, tuple):
        out_shp = tuple(
            _get_conv_shape_1axis(
                imshp[i],
                kshp[i],
                border_mode[i],
                subsample[i],
                filter_dilation[i],
            )
            for i in range(len(subsample))
        )
    else:
        out_shp = tuple(
            _get_conv_shape_1axis(
                imshp[i], kshp[i], border_mode, subsample[i], filter_dilation[i]
            )
            for i in range(len(subsample))
        )
    return (bsize, nkern) + out_shp