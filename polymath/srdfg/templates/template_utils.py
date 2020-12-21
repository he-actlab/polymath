import polymath as pm


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