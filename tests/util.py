from polymath.mgdfg.base import Node
import numpy as np
import polymath as pm
from itertools import product
from collections import defaultdict
import pprint
import onnx
from onnx import helper, numpy_helper, mapping
from onnx import AttributeProto, TensorProto, GraphProto
import json


def pooling(data, kh, kw, pad=0,  stride=2):
    N, C, H, W = data.shape

    # Check dimensions
    assert (W + 2 * pad - kh) % stride == 0, 'width does not work'
    assert (H + 2 * pad - kw) % stride == 0, 'height does not work'
    dpadded = np.pad(data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # Create output
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    out = np.zeros((N, C, OH, OW))
    for b in range(N):
        for c in range(C):
            for y in range(OH):
                for x in range(OW):
                    for m in range(kh):
                        for n in range(kw):
                            out[b][c][y][x] += dpadded[b][c][stride*y + m][stride*x + n]
                    out[b][c][y][x] /= (kh*kw)
    return out


def np_svm(input_info):
    out_info = {}
    out_info["x*w"] = input_info["x"]*input_info["w"]
    out_info["h"] = np.sum(out_info["x*w"])
    out_info["c"] = input_info["y"]*out_info["h"]
    out_info["ny"] = 0 - input_info["y"]
    out_info["p"] = (out_info["c"] > 1) * out_info["ny"]
    out_info["g"] = out_info["p"] * input_info["x"]
    out_info["mu*g"] = input_info["mu"] * out_info["g"]
    out_info["w"] = input_info["w"] - out_info["mu*g"]
    return out_info

def svm(m=3, coarse=False):
    with pm.Node(name="svm") as graph:
        m_ = pm.parameter("m")
        mu = pm.parameter(name="mu", default=1.0)
        x = pm.input("x", shape=(m_))
        y = pm.input("y")
        w = pm.state("w", shape=(m_))
        i = pm.index(0, (m_ - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")
        c = (y*h).set_name("c")
        ny = (0 - y).set_name("ny")
        p_ = pm.cast(np.float32, (c > y))
        p = (p_*ny).set_name("p")
        g = (p * x[i]).set_name("g")
        w[i] = w[i] - mu * g[i]

    if coarse:
        in_info, keys, out_info = svm_data_gen(m=m)
        return graph, in_info, out_info, keys
    else:
        shape_val_pass = pm.NormalizeGraph({"m": m})
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = svm_data_gen(m=m, lowered=True)
        return new_graph, in_info, out_info, keys

def svm_data_gen(m=3, mu=1.0, lowered=False):
    input_info = {}
    input_info["x"] = np.random.randint(-3, 50, m)
    input_info["w"] = np.random.randint(-3, 50, m)
    input_info["y"] = np.random.randint(-3, 50, 1)[0]
    input_info["mu"] = mu
    out_info = np_svm(input_info)
    if lowered:
        all_keys = []
        for p in range(m):
            w_key = f"w/w({p},)"
            all_keys.append(w_key)
            input_info[w_key] = input_info["w"][p]
            input_info[f"x/x({p},)"] = input_info["x"][p]
        input_info.pop("w")
        input_info.pop("x")
    else:
        all_keys = "w"

    return input_info, all_keys, out_info


def set_shape_and_lower(graph, shape_dict):
    shape_pass = pm.NormalizeGraph(shape_dict)
    lower_pass = pm.Lower({})
    shaped = shape_pass(graph)
    lowered = lower_pass(shaped)
    return lowered

def compare_tabla_dfg(truth_path, gen_dfg, pm_graph, print_ops=True, map_node_ids=False):
    with open(truth_path) as truth_file:
        data = json.load(truth_file)

    if print_ops:
        debug_print_tabla(data, gen_dfg, pm_graph)
    keys = [k.split("/")[-1] for k, n in pm_graph.nodes.items() if n.op_name in ["input", "state"]]
    keys += ["source", "sink"]
    if map_node_ids:
        map_tabla_nodes(keys, gen_dfg, data)


def map_tabla_nodes(keys, gen_dfg, data):
    id_map = {}
    for k in gen_dfg:
        if k["operation"] in keys:
            k["operation"] = k["operation"].replace("(", "[").replace(")", "]").replace(",]", "]").replace(", ", "][")
            truth_node = get_tabla_item_by_op(data, k["operation"])
            id_map[k["id"]] = truth_node["id"]

            if len(truth_node["children"]) != len(k["children"]):
                print(f"{k['operation']} has generated children {len(k['children'])} which is not equal to {len(truth_node['children'])}")

            if len(truth_node["parents"]) != len(k["parents"]):
                print(f"{k['operation']} has generated {len(k['parents'])} parents which is not equal to {len(truth_node['parents'])}")
        else:
            parents = []
            for p in k["parents"]:
                if p not in id_map:
                    print(f"Could not find {p} in id_map for {k['operation']}")
                else:
                    parents.append(id_map[p])
            ref_node = get_tabla_item_by_parents(data, parents, k)
            id_map[k["id"]] = ref_node["id"]

def debug_print_tabla(data, gen_dfg, pm_graph):
    print(f"Printing opcounts for polymath graph:\n")
    print(op_counts(pm_graph))
    print(f"Printing opcounts for tabla IR generated graph with length {len(gen_dfg)}:\n")
    gen_op_count = print_tabla_op_counts(gen_dfg)
    _ = print_tabla_type_counts(gen_dfg)
    print(f"Printing opcounts for tabla IR original graph with length {len(data)}:\n")
    ref_op_count = print_tabla_op_counts(data)
    _ = print_tabla_type_counts(data)

def get_tabla_item_by_op(graph, op):
    for n in graph:
        if n["operation"] == op:
            return n
    raise KeyError(f"Could not find {op}")

def get_tabla_item_by_parents(graph, parents, node):
    op_name = node["operation"]
    parents = sorted(parents)
    for n in graph:

        if sorted(n["parents"]) == parents and op_name == n["operation"]:
            return n
    raise KeyError(f"Could not find {op_name} with parents {parents} for node:\n\t{node}")



def run_onnx_graph(model, input_vals, outputs):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model)
    inputs = {sess.get_inputs()[i].name: input_vals[i] for i in range(len(sess.get_inputs()))}
    result = sess.run(outputs, inputs)
    return result

def create_onnx_node_test(node, inputs, outputs, name):
    input_value_info = []
    output_value_info = []
    for name, value in inputs.items():
        ival = helper.make_tensor_value_info(name, mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype], value.shape)
        input_value_info.append(ival)

    for name, value in outputs.items():
        if isinstance(value, dict):
            oval = helper.make_tensor_value_info(name, mapping.NP_TYPE_TO_TENSOR_TYPE[value['dtype']], value['shape'])
        else:
            oval = helper.make_tensor_value_info(name, mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype], value.shape)
        output_value_info.append(oval)

    graph_def = helper.make_graph(
        [node],
        name,
        input_value_info,
        output_value_info,
    )
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    return model_def

def onnx_nms(boxes, scores, max_output_per_class=0, iou_threshold=0, score_threshold=0, center_point_box=0):
    input_names = ['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold']
    output_names = ['selected_indices']

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=input_names,
        outputs=output_names,
        center_point_box=center_point_box
    )
    max_output_boxes_per_class = np.array([max_output_per_class]).astype(np.int64)
    iou_threshold = np.array([iou_threshold]).astype(np.float32)
    score_threshold = np.array([score_threshold]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = [selected_indices]
    input_dict = {name: inputs[idx] for idx, name in enumerate(input_names)}
    output_dict = {name: {'dtype': outputs[idx].dtype, 'shape': ['w', 'h']} for idx, name in enumerate(output_names)}
    node_graph = create_onnx_node_test(node, input_dict, output_dict, "onnx_nms")
    onnx_path = 'nms_test.onnx'
    onnx.save_model(node_graph, onnx_path)
    onnx_res = run_onnx_graph(onnx_path, inputs, output_names)
    return onnx_res, outputs[0]

def np_nms(boxes, scores, max_output_per_class=0, iou_threshold=0, score_threshold=0, center_point_box=0):
    max_output_boxes_per_class = np.array([max_output_per_class]).astype(np.int64)
    iou_threshold = np.array([iou_threshold]).astype(np.float32)
    score_threshold = np.array([score_threshold]).astype(np.float32)
    boxes = boxes.squeeze(0)
    scores = scores.squeeze(0).squeeze(0)
    x1_t = boxes[:, 0]
    y1_t = boxes[:, 1]
    x2_t = boxes[:, 2]
    y2_t = boxes[:, 3]

    areas = (x2_t - x1_t) * (y2_t - y1_t)

    ndets = boxes.shape[0]
    # order = np.argsort(scores, axis=0)
    order = scores.argsort()[::-1]

    num_to_keep = 0
    supressed = np.zeros(ndets,  dtype=order.dtype)
    keep = np.zeros(ndets, dtype=order.dtype)

    for i_ in range(ndets):
        i = order[i_]
        if supressed[i] == 1:
            continue

        keep[num_to_keep] = i
        num_to_keep += 1
        ix1 = x1_t[i]
        iy1 = y1_t[i]
        ix2 = x2_t[i]
        iy2 = y2_t[i]
        iarea = areas[i]

        for j_ in range(i_+1, ndets):
            j = order[j_]
            if supressed[j] == 1:
                continue

            xx1 = max(ix1, x1_t[j])
            yy1 = max(iy1, y1_t[j])
            xx2 = min(ix2, x2_t[j])
            yy2 = min(iy2, y2_t[j])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr > iou_threshold:
                supressed[j] = 1
    print(num_to_keep)
    return keep[:num_to_keep]

def np_nms_preprocess(boxes, scores, max_output_per_class, iou_threshold, score_threshold, center_point_box):
    pass

def _torch_preprocess(boxes, scores):
    boxes = boxes.squeeze(0)
    scores = scores.squeeze(0).squeeze(0)

    return boxes, scores

def test_torch_nms(boxes, scores, overlap_threshold=0.5, min_mode=False):
    import torch
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    _keep = scores.new(scores.size(0)).zero_().long()

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(dim=0, descending=True)
    cnt = 0

    while order.size()[0] > 1 and cnt < _keep.shape[0]:
        _keep[cnt] = order[0]
        cnt += 1
        xx1 = torch.max(x1[order[0]], x1[order[1:]])
        yy1 = torch.max(y1[order[0]], y1[order[1:]])
        xx2 = torch.min(x2[order[0]], x2[order[1:]])
        yy2 = torch.min(y2[order[0]], y2[order[1:]])

        w = torch.clamp(xx2-xx1, min=0)
        h = torch.clamp(yy2-yy1, min=0)
        inter = w * h
        if min_mode:
            ovr = inter / torch.min(areas[order[0]], areas[order[1:]])
        else:
            ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = torch.nonzero(ovr <= overlap_threshold).squeeze()
        if inds.dim():
            order = order[inds + 1]
        else:
            break

    return _keep[:cnt]

def nms_cpu(boxes, scores, overlap_threshold=0.5, min_mode=False):
    boxes = boxes.cpu().numpy()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = scores.cpu().numpy()

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        keep.append(order[0])
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if min_mode:
            ovr = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    return keep

def torch_nms(boxes, scores, max_output_per_class=0, iou_threshold=0, score_threshold=0, center_point_box=0):
    from torchvision.ops import nms
    import torch
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(scores)
    boxes, scores = _torch_preprocess(boxes, scores)
    # np_boxes = boxes.numpy()
    # np_scores = scores.numpy()
    np_res = nms_cpu(boxes, scores)
    tres = test_torch_nms(boxes, scores)
    print(tres)
    print(np_res)
    # print(np_res)
    # print(tres)
    # res = nms(boxes, scores, iou_threshold)
    # print(res.unsqueeze(1).unsqueeze(1))

def op_counts(graph):
    counts = defaultdict(int)
    for k,v in graph.nodes.items():
        counts[v.op_name] += 1
    return counts

def print_tabla_op_counts(tb_ir):
    counts = defaultdict(int)
    for n in tb_ir:
        counts[n["operation"]] += 1
    print(f"---------------Op counts-------------\n\t"
          f"# Unique ops: {len(counts)}\n\t"
          f"Op Counts: ")
    pprint.pprint((counts))
    return counts

def print_tabla_type_counts(tb_ir):
    counts = defaultdict(int)
    for n in tb_ir:
        counts[n["dataType"]] += 1
    print(f"---------------Type counts-------------\n\t"
          f"# Unique types: {len(counts)}\n\t"
          f"Type Counts: ")
    pprint.pprint(list(counts))
    return counts

def np_linear(input_info):
    out_info = {}
    out_info["x*w"] = input_info["x"]*input_info["w"]
    out_info["h"] = np.sum(out_info["x*w"])
    out_info["d"] = out_info["h"] - input_info["y"]
    out_info["g"] = out_info["d"] * input_info["x"]
    out_info["mu*g"] = input_info["mu"] * out_info["g"]
    out_info["w"] = input_info["w"] - out_info["mu*g"]
    return out_info

def linear(m=3, coarse=False):
    with pm.Node(name="linear") as graph:
        m_ = pm.parameter("m")
        mu = pm.parameter(name="mu", default=1.0)
        x = pm.input("x", shape=(m_))
        y = pm.input("y")
        w = pm.state("w", shape=(m_))
        i = pm.index(0, (m_ - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]

    if coarse:
        in_info, keys, out_info = linear_data_gen(m=m)
        return graph, in_info, out_info, keys
    else:
        shape_val_pass = pm.NormalizeGraph({"m": m})
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = linear_data_gen(m=m, lowered=True)
        return new_graph, in_info, out_info, keys



def backprop_data_gen(l1=None, l2=None, l3=None, mu=1.0, lowered=False, debug=False):
    input_info = {}
    if debug:
        input_info["x"] = np.arange(-1*l1//2, l1//2).clip(-3,3).astype(np.float)

    else:
        input_info["x"] = np.random.randint(-3, 3, l1).astype(np.float)

    if debug:
        w1 = np.arange(-l2//2, l2//2)
        input_info["w1"] = np.repeat(w1, l1).reshape((l1,l2)).clip(-3,3).astype(np.float)
    else:
        input_info["w1"] = np.random.randint(-3, 3, (l1,l2)).astype(np.float)

    if debug:
        w2 = np.arange(-l3//2, l3//2)
        input_info["w2"] = np.repeat(w2, l2).reshape((l2,l3)).clip(-3,3).astype(np.float)
    else:
        input_info["w2"] = np.random.randint(-3, 3, (l2,l3)).astype(np.float)

    if debug:
        input_info["y"] = np.arange(-1*l3//2, l3//2).clip(-3,3).astype(np.float)
    else:
        input_info["y"] = np.random.randint(-3, 3, l3).astype(np.float)

    # if debug:
    #     input_info["b1"] = np.arange(-1*l2//2, l2//2).clip(-3,3)
    # else:
    #     input_info["b1"] = np.random.randint(-3, 3, l2)
    #
    #
    # if debug:
    #     input_info["b2"] = np.arange(-1*l2//2, l2//2).clip(-3,3)
    # else:
    #     input_info["b2"] = np.random.randint(-3, 3, l2)


    input_info["mu"] = mu
    out_info = np_backprop(input_info)
    if lowered:
        all_keys = []
        for k in list(input_info.keys()):
            if hasattr(input_info[k], "shape") and input_info[k].shape != (1,):
                pairs = list(product(*tuple([np.arange(i) for i in input_info[k].shape])))
                for p in pairs:
                    input_info[f"{k}/{k}{p}"] = input_info[k][p]
                    if k in ['w1','w2']:
                        all_keys.append(f"{k}/{k}{p}")
                input_info.pop(k)
    else:
        all_keys = ["w1","w2"]

    return input_info, all_keys, out_info


def np_backprop(input_info):
    out_info = {}
    out_info["a1"] = sigmoid(input_info["x"].dot(input_info["w1"]))
    out_info["a2"] = sigmoid(out_info["a1"].dot(input_info["w2"]))
    out_info["d3"] = out_info["a2"] - input_info["y"]

    out_info["d2"] = input_info["w2"].dot(out_info["d3"])*(out_info["a1"]*(1-out_info["a1"]))
    out_info["w1"] = input_info["w1"] - input_info["mu"]*(np.outer(input_info["x"], out_info["d2"]))
    out_info["w2"] = input_info["w2"] - input_info["mu"]*(np.outer(out_info["a1"], out_info["d3"]))

    return out_info

def backprop(l1=9, l2=10, l3=1, coarse=False, debug=False, pbar=False):
    with pm.Node(name="backprop") as graph:
        mu = pm.parameter("mu", default=1.0)
        l1_ = pm.parameter("l1")
        l2_ = pm.parameter("l2")
        l3_ = pm.parameter("l3")
        x = pm.input("x", shape=(l1_))
        y = pm.input("y", shape=(l3_))
        w1 = pm.state("w1", shape=(l1_, l2_))
        w2 = pm.state("w2", shape=(l2_, l3_))

        i1 = pm.index(0, (l1_ - 1), name="i1")
        i2 = pm.index(0, (l2_ - 1), name="i2")
        i3 = pm.index(0, (l3_ - 1), name="i3")

        a1 = pm.sigmoid(pm.sum([i1], x[i1]*(w1[i1, i2]).set_name("w1*x"),name="sum(w1*x)"),name="a1")
        a2 = pm.sigmoid(pm.sum([i2], (w2[i2, i3] * a1[i2]).set_name("w2*a1"), name="sum(w2*a1)"), name="a2")
        d3 = (a2[i3] - y[i3]).set_name("d3")
        d2 = pm.sum([i3], (w2[i2, i3]*d3[i3]), name="d2") * (a1[i2]*(mu - a1[i2]))
        g1 = (x[i1]*d2[i2]).set_name("g1")
        g2 = (a1[i2]*d3[i3]).set_name("g2")
        w1[i1, i2] = w1[i1, i2] - mu*g1[i1, i2]
        w2[i2, i3] = w2[i2, i3] - mu*g2[i2, i3]


    if coarse:
        in_info, keys, out_info = backprop_data_gen(l1, l2, l3, debug=debug)
        return graph, in_info, out_info, keys
    else:
        shape_val_pass = pm.NormalizeGraph({"l1": l1, "l2": l2, "l3": l3}, debug=pbar)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = backprop_data_gen(l1, l2, l3, lowered=True, debug=debug)
        return new_graph, in_info, out_info, keys

def linear_raw(m=3, coarse=False):
    with pm.Node(name="linear") as graph:
        m_ = pm.parameter("m")
        mu = pm.parameter(name="mu", default=1.0)
        x = pm.input("x", shape=(m_))
        y = pm.input("y")
        w = pm.state("w", shape=(m_))
        i = pm.index(0, (m_ - 1).set_name("m-1"), name="i")
        h = pm.sum([i], (x[i] * w[i]), name="h")
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]
        out = (w[i] + 5).set_name("w5")

    if coarse:
        in_info, keys, out_info = linear_data_gen(m=m)
        return graph, in_info, out_info, keys
    else:
        shape_val_pass = pm.NormalizeGraph({"m": m})
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = linear_data_gen(m=m, lowered=True)
        return new_graph, in_info, out_info, keys

def linear_data_gen(m=3, mu=1.0, lowered=False):
    input_info = {}
    input_info["x"] = np.random.randint(-3, 10, m)
    input_info["w"] = np.random.randint(-3, 10, m)
    input_info["y"] = np.random.randint(-3, 10, 1)[0]
    input_info["mu"] = mu
    out_info = np_linear(input_info)
    if lowered:
        all_keys = []
        for p in range(m):
            w_key = f"w/w({p},)"
            all_keys.append(w_key)
            input_info[w_key] = input_info["w"][p]
            input_info[f"x/x({p},)"] = input_info["x"][p]
        input_info.pop("w")
        input_info.pop("x")
    else:
        all_keys = "w"

    return input_info, all_keys, out_info

def sigmoid(value):
    return (1 / (1 + np.exp(-value)))

def np_logistic(input_info):
    out_info = {}
    out_info["x*w"] = (input_info["x"]*input_info["w"])
    out_info["reduce"] = np.sum(out_info["x*w"])
    out_info["h"] = sigmoid(out_info["reduce"])
    out_info["d"] = out_info["h"] - input_info["y"]
    out_info["g"] = out_info["d"] * input_info["x"]
    out_info["mu*g"] = input_info["mu"] * out_info["g"]
    out_info["w"] = input_info["w"] - out_info["mu*g"]
    return out_info

def logistic(m=3, coarse=False):
    with pm.Node(name="logistic") as graph:
        m_ = pm.parameter("m")
        mu = pm.parameter(name="mu", default=1)
        x = pm.input("x", shape=(m_))
        y = pm.input("y")
        w = pm.state("w", shape=(m_))
        i = pm.index(0, (m_ - 1).set_name("m-1"), name="i")
        h = pm.sigmoid(pm.sum([i], (x[i] * w[i]), name="h"))
        d = (h - y).set_name("h-y")
        g = (d * x[i]).set_name("d*x")
        w[i] = w[i] - mu * g[i]

    if coarse:
        in_info, keys, out_info = logistic_data_gen(m=m)
        return graph, in_info, out_info, keys
    else:
        shape_val_pass = pm.NormalizeGraph({"m": m})
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = logistic_data_gen(m=m, lowered=True)
        return new_graph, in_info, out_info, keys

def logistic_data_gen(m=3, mu=1.0, lowered=False):
    input_info = {}
    input_info["x"] = np.random.randint(-3, 3, m).astype(np.float)
    input_info["w"] = np.random.randint(-3, 3, m).astype(np.float)
    input_info["y"] = np.random.randint(-3, 3, 1)[0].astype(np.float)
    input_info["mu"] = mu
    out_info = np_logistic(input_info)
    if lowered:
        all_keys = []
        for p in range(m):
            w_key = f"w/w({p},)"
            all_keys.append(w_key)
            input_info[w_key] = input_info["w"][p]
            input_info[f"x/x({p},)"] = input_info["x"][p]
        input_info.pop("w")
        input_info.pop("x")
    else:
        all_keys = "w"

    return input_info, all_keys, out_info



def numpy_reco(input_dict):

    out_info = {}
    w1_x2 = np.zeros(input_dict["w1"].shape)
    w2_x1 = np.zeros(input_dict["w2"].shape)
    h1 = np.zeros(shape=(input_dict["m"]))
    h1_sum = np.zeros(shape=(input_dict["m"]))
    h2 = np.zeros(shape=(input_dict["n"]))
    h2_sum = np.zeros(shape=(input_dict["n"]))

    for i in range(input_dict["m"]):
        for l in range(input_dict["k"]):
            w1_x2[i][l] = input_dict["w1"][i][l] * input_dict["x2"][l]
            h1_sum[i] += w1_x2[i][l]
        h1[i] = h1_sum[i] * input_dict["r1"][i]
    out_info["h1"] = h1
    out_info["h1_sum"] = h1_sum
    out_info["w1_x2"] = w1_x2

    for j in range(input_dict["n"]):
        for l in range(input_dict["k"]):
            w2_x1[j][l] =input_dict["w2"][j][l] * input_dict["x1"][l]
            h2_sum[j] += w2_x1[j][l]
        h2[j] = h2_sum[j] * input_dict["r2"][j]
    out_info["h2"] = h2
    out_info["h2_sum"] = h2_sum
    out_info["w2_x1"] = w2_x1
    d1 = h1 - input_dict["y1"]
    out_info["d1"] = d1
    d2 = h2 - input_dict["y2"]
    out_info["d2"] = d2
    g1 = np.zeros(shape=(input_dict["m"], input_dict["k"]))
    g2 = np.zeros(shape=(input_dict["n"], input_dict["k"]))

    for i in range(input_dict["m"]):
        for l in range(input_dict["k"]):
            g1[i][l] = d1[i] * input_dict["x2"][l]
    out_info["g1"] = g1
    for j in range(input_dict["n"]):
        for l in range(input_dict["k"]):
            g2[j][l] = d2[j] * input_dict["x1"][l]
    out_info["g2"] = g2
    w1_out = input_dict["w1"] - g1
    w2_out = input_dict["w2"] - g2
    input_dict.pop("k")
    input_dict.pop("m")
    input_dict.pop("n")
    out_info["w1"] = w1_out
    out_info["w2"] = w2_out
    return out_info


def reco_data_gen(m_=3, n_=3, k_=2, mu=1.0, lowered=False):
    input_info = {}
    input_info["mu"] = mu

    input_info["m"] = m_
    input_info["n"] = n_
    input_info["k"] = k_
    input_info["w1"] = np.random.randint(-3, 5, m_ * k_).reshape(m_, k_)
    input_info["w2"] = np.random.randint(-3, 5, n_ * k_).reshape(n_, k_)
    input_info["x1"] = np.random.randint(-3, 5, k_)
    input_info["x2"] = np.random.randint(-3, 5, k_)

    input_info["r1"] = np.random.randint(0, 2, m_)
    input_info["y1"] = np.random.randint(0, 5, m_)
    input_info["r2"] = np.random.randint(0, 2, n_)
    input_info["y2"] = np.random.randint(0, 5, n_)
    out_info = numpy_reco(input_dict=input_info)
    if lowered:
        pairs_w1 = list(product(*tuple([np.arange(i) for i in input_info["w1"].shape])))
        pairs_w2 = list(product(*tuple([np.arange(i) for i in input_info["w2"].shape])))
        for p in pairs_w1:
            input_info[f"w1/w1({p[0]}, {p[1]})"] = input_info["w1"][p]
        input_info.pop("w1")

        for p in pairs_w2:
            input_info[f"w2/w2({p[0]}, {p[1]})"] = input_info["w2"][p]
        input_info.pop("w2")

        for p in range(k_):
            input_info[f"x1/x1({p},)"] = input_info["x1"][p]
            input_info[f"x2/x2({p},)"] = input_info["x2"][p]
        input_info.pop("x1")
        input_info.pop("x2")

        for p in range(m_):
            input_info[f"r1/r1({p},)"] = input_info["r1"][p]
            input_info[f"y1/y1({p},)"] = input_info["y1"][p]
        input_info.pop("r1")
        input_info.pop("y1")

        for p in range(n_):
            input_info[f"r2/r2({p},)"] = input_info["r2"][p]
            input_info[f"y2/y2({p},)"] = input_info["y2"][p]
        input_info.pop("r2")
        input_info.pop("y2")

        w1_keys = [f"w1/w1({p[0]}, {p[1]})" for p in pairs_w1]
        w2_keys = [f"w2/w2({p[0]}, {p[1]})" for p in pairs_w2]
        all_keys = w1_keys + w2_keys
    else:
        all_keys = ["w1", "w2"]
    return input_info, all_keys, out_info

def reco(m=3, n=3, k=3, coarse=False):
    with pm.Node(name="reco") as graph:
        m_ = pm.parameter("m")
        n_ = pm.parameter("n")
        k_ = pm.parameter("k")
        mu = pm.parameter("mu")
        x1 = pm.input("x1", shape=(k_))
        x2 = pm.input("x2", shape=(k_))

        r1 = pm.input("r1", shape=(m_))
        y1 = pm.input("y1", shape=(m_))

        r2 = pm.input("r2", shape=(n_))
        y2 = pm.input("y2", shape=(n_))

        w1 = pm.state("w1", shape=(m_, k_))
        w2 = pm.state("w2", shape=(n_, k_))
        i = pm.index(0, (m_ - 1).set_name("m-1"), name="i")
        j = pm.index(0, (n_ - 1).set_name("n-1"), name="j")
        l = pm.index(0, (k_ - 1).set_name("k-1"), name="l")

        h1_sum = pm.sum([l], (w1[i, l] * x2[l]).set_name("w1*x2")).set_name("h1_sum")
        h1 = (h1_sum[i] * r1[i]).set_name("h1")

        h2_sum = pm.sum([l], (w2[j, l] * x1[l]).set_name("w2*x1")).set_name("h2_sum")
        h2 = (h2_sum[j] * r2[j]).set_name("h2")

        d1 = (h1[i] - y1[i]).set_name("d1")
        d2 = (h2[j] - y2[j]).set_name("d2")
        g1 = (d1[i] * x2[l]).set_name("g1")
        g2 = (d2[j] * x1[l]).set_name("g2")
        w1[i, l] = (w1[i, l] - (mu*g1[i, l]).set_name("mu*g1")).set_name("w1-g1")
        w2[j, l] = (w2[j, l] - (mu*g2[j, l]).set_name("mu*g2")).set_name("w2-g2")

    if coarse:
        in_info, keys, out_info = reco_data_gen(m_=m,n_=n, k_=k)
        return graph, in_info, out_info, keys
    else:
        shape_val_pass = pm.NormalizeGraph({"m": m, "n": n, "k": k})
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = reco_data_gen(m_=m, n_=n, k_=k, lowered=True)
        return new_graph, in_info, out_info, keys

def debug_node_attr(node, tabs=None):
    tabs = "" if tabs is None else tabs
    added = "\t\t"
    return (f"{tabs}Node: {node}\n"
            f"{tabs}\tName: {node.name}\n"
            f"{tabs}\tGraph: {node.graph}\n"
            f"{tabs}\tOpname: {node.op_name}\n"
              f"{tabs}\tArgs: {node.args}\n"
              f"{tabs}\tKwargs: {node.kwargs}\n"
              f"{tabs}\tShape: {node.shape}\n"
              f"{tabs}\tNodes: \n{str([debug_node_attr(n, tabs=added) for _, n in node.nodes.items()])}\n")

def count_nodes(graph):
    counts = _count_nodes(graph, {"global": 0, "count": 0})
    return counts


def _count_nodes(graph, counts):
    for k, node in graph.nodes.items():
        if node.graph:
            if node.graph.name in counts.keys():
                counts[node.graph.name] += 1
            else:
                counts[node.graph.name] = 1
        else:
            counts["global"] += 1
        counts["count"] += 1
        _count_nodes(node, counts)
    return counts


def compare_nodes(node_x, node_y):
    if node_x.name != node_y.name:
        print(f"Unequal names: {node_x.name}\t{node_y.name}")
        return False
    elif node_x.op_name != node_y.op_name:
        print(f"Unequal ops: {node_x.op_name}\t{node_y.op_name}")
        return False
    if node_x.shape != node_y.shape:
        print(f"Unequal shape for {node_x.name} and {node_y.name}: {node_x.shape}\t{node_y.shape}")

        return False

    if len(node_x.args) != len(node_y.args):
        return False

    for arg_name, arg in enumerate(node_x.args):
        if isinstance(arg, Node) and arg.name != node_y.args[arg_name].name:
            print(f"Unequal args: {arg}\t{node_y.args[arg_name]} for "
                  f"{node_x.name}-{node_x.op_name} and {node_y.name}-{node_y.op_name}:\n"
                  f"x: {node_x.args}\ny: {node_y.args}")
            return False
        elif not isinstance(arg, Node) and arg != node_y.args[arg_name]:
            print(f"Unequal args: {arg}\t{node_y.args[arg_name]} for "
                  f"{node_x.name}-{node_x.op_name} and {node_y.name}-{node_y.op_name}:\n"
                  f"x: {node_x.args}\ny: {node_y.args}")
            return False

    for arg_key, arg in node_x.kwargs.items():
        if isinstance(arg, Node) and arg.name != node_y.args[arg_key].name:
            print(f"Unequal kwargs: {arg}\t{node_y.kwargs[arg_key]}")
            return False
        elif not isinstance(arg, Node) and arg != node_y.kwargs[arg_key]:
            print(f"Unequal kwargs: {arg}\t{node_y.kwargs[arg_key]}")
            return False

    if node_y.nodes.keys() != node_x.nodes.keys():
        print(f"Unequal node lists: {node_y.nodes.keys()}\t{node_y.kwargs[node_x.nodes.keys()]} for {node_x.name} and {node_y.name}")
        return False
    if node_x.nodes != node_y.nodes:
        print(f"Unequal node lists: {node_x.nodes.items()} \n{node_y.nodes.items()} for {node_x.name} and {node_y.name}")
        return False


    if node_x.graph is None and node_y.graph is not None:
        print(f"Unequal graphs: {node_x.graph} and {node_y.graph} for {node_x.name} and {node_y.name}")
        return False
    elif node_y.graph is None and node_x.graph is not None:
        print(f"Unequal graphs: {node_x.graph} and {node_y.graph} for {node_x.name} and {node_y.name}")
        return False
    elif node_x.graph and node_y.graph and node_x.graph.name != node_y.graph.name:
        print(f"Unequal graphs: {node_x.graph} and {node_y.graph} for {node_x.name} and {node_y.name}")
        return False
    return True

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) // stride + 1
  out_width = (W + 2 * padding - field_width) // stride + 1

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols

def conv_t(data, w, bias, conv_param):
    N, C, H, W = data.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'
    x_padded = np.pad(data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, int(out_height), int(out_width)), dtype=data.dtype)
    tout = np.zeros((N, num_filters, H + 2 * pad, W + 2 * pad, filter_height, filter_width, C))
    tout3 = np.zeros((N, C, filter_height, int(out_height), filter_width, int(out_width),  num_filters))
    vidx_pairs = np.zeros((N, C, filter_height, int(out_height), filter_width, int(out_width),  num_filters))
    tout32 = np.zeros((N, C, filter_height, int(out_height), filter_width, int(out_width),  num_filters))
    test_var_idx = np.zeros(x_padded.shape)
    p_map = {}
    c1 = []
    c2 = []
    all_pairs = {}

    tpairs = []
    for b in range(N):
        for k in range(C):
            for dy in range(filter_height):
                for y in range(out_height):
                    for dx in range(filter_width):
                        for x in range(out_width):
                            for c in range(num_filters):

                                c1.append((b,k,(dy+stride*y), (dx+stride*x)))
                                c2.append((c,k,(dy), (dx)))
                                all_pairs[((b, k, dy, y, dx, x, c))] = (c1[-1], c2[-1])
                                p_map[(b,c,dy,y,dx,x,k)] = (c1[-1], c2[-1])
                                tout32[b][k][dy][y][dx][x][c] = x_padded[b][k][dy + stride*y][dx + stride*x]
                                tout3[b][k][dy][y][dx][x][c] = x_padded[b][k][dy + stride*y][dx + stride*x]*w[c][k][dy][dx]
                                out[b][c][y][x] += tout3[b][k][dy][y][dx][x][c]
                                test_var_idx[b][k][dy + stride*y][dx + x*stride] = x_padded[b][k][dy + stride*y][dx + stride*x]
                            tpairs.append(c1[-1])

    tout4 = np.sum(tout3, axis=(2, 4, 6))
    pairs = [(c1[i], c2[i]) for i in range(len(c1))]
    return out, tout3, all_pairs

def conv3d(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) / stride + 1
    out_width = (W + 2 * pad - filter_width) / stride + 1
    out = np.zeros((N, num_filters, int(out_height), int(out_width)), dtype=x.dtype)
    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1,1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)
    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_forward_strides(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

    # Pad the input
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # Perform an im2col operation by picking clever strides
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)

    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                                               shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)

    # Now all our convolutions are a big matrix multiply
    # res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)
    res = w.reshape(F, -1).dot(x_cols)

    # Reshape the output
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)

    # Be nice and return a contiguous array
    # The old version of conv_forward_fast doesn't do this, so for a fair
    # comparison we won't either
    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache

def get_iris_data():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_test, y_test

def multinomial_log(x, w):
    numer = np.exp(x.dot(w))
    return numer/np.sum(numer)

def predictions(x, w):
    res = np.asarray([_predict_one(x[i], w) for i in range(x.shape[0])])
    return res

def _predict_one(x, w):
    if len(w.shape) > 1:
        res = np.asarray([x.dot(w[i]) for i in range(w.shape[0])])
        res = res / sum(res)
    else:
        res = x.dot(w)
    return res


def conv_data_gen(x_shape, w_shape, params, lowered=False, debug_matrix=False):
    input_info = {}

    input_info["pad"] = params["pad"]
    input_info["stride"] = params["stride"]
    if debug_matrix:
        input_info["data"] = np.arange(0, (np.prod(x_shape))).reshape(x_shape)
    else:
        input_info["data"] = np.random.randint(-5, 5, x_shape)
    input_info["data"] = input_info["data"].astype(np.float)

    if debug_matrix:
        input_info["w"] = np.arange(0, (np.prod(w_shape))).reshape(w_shape)
    else:
        input_info["w"] = np.random.randint(-5, 5, w_shape)
    input_info["w"] = input_info["w"].astype(np.float)

    if debug_matrix:
        input_info["bias"] = np.zeros((w_shape[0]))
    else:
        input_info["bias"] = np.random.randint(0, 10, (w_shape[0]))
    input_info["bias"] = input_info["bias"].astype(np.float)


    out = conv3d(input_info["data"], input_info["w"], input_info["bias"], params)
    out_info = {"out": out[0]}
    if lowered:
        pairs_w = list(product(*tuple([np.arange(i) for i in input_info["w"].shape])))
        pairs_data = list(product(*tuple([np.arange(i) for i in input_info["data"].shape])))
        for p in pairs_w:
            input_info[f"w/w({p[0]}, {p[1]}, {p[2]}, {p[3]})"] = input_info["w"][p]
        input_info.pop("w")

        for p in pairs_data:
            input_info[f"data/data({p[0]}, {p[1]}, {p[2]}, {p[3]})"] = input_info["data"][p]
        input_info.pop("data")

        for p in range(w_shape[0]):
            input_info[f"bias/bias({p},)"] = input_info["bias"][p]
        input_info.pop("bias")
        out_pairs = list(product(*tuple([np.arange(i) for i in out_info["out"].shape])))

        all_keys = [f"out/out({p[0]}, {p[1]}, {p[2]}, {p[3]})" for p in out_pairs]
    else:
        all_keys = "out"

    return input_info, all_keys, out_info

def conv(x_shape, w_shape, params, coarse=False, debug_matrix=False):
    with pm.Node(name="conv") as graph:
        n = pm.parameter(name="n")
        c = pm.parameter(name="ic")
        ih = pm.parameter(name="ih")
        iw = pm.parameter(name="iw")
        nf = pm.parameter(name="nf")
        kh = pm.parameter(name="kh")
        kw = pm.parameter(name="kw")
        x = pm.input(name="data", shape=(n, c, ih, iw))
        w = pm.state(name="w", shape=(nf, c, kh, kw))
        b = pm.state(name="bias", shape=(nf))
        stride = pm.parameter(name="stride")
        pad = pm.parameter(name="pad")
        out = pm.output(name="out")
        pm.conv_bias(x, w, b, out, stride, pad, name="conv_op")

    if coarse:
        in_info, keys, out_info = conv_data_gen(x_shape, w_shape, params, debug_matrix=debug_matrix)
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3],
                      "nf": w_shape[0], "kh": w_shape[2], "kw": w_shape[3],
                      "stride": params["stride"], "pad": params["pad"]}
        shape_val_pass = pm.NormalizeGraph(shape_dict)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = conv_data_gen(x_shape, w_shape, params, lowered=True, debug_matrix=debug_matrix)
        return new_graph, in_info, out_info, keys

def batchnorm(x_shape, coarse=False, debug_matrix=False):
    with pm.Node(name="conv") as graph:
        n = pm.parameter(name="n")
        c = pm.parameter(name="ic")
        ih = pm.parameter(name="ih")
        iw = pm.parameter(name="iw")
        x = pm.input(name="x", shape=(n, c, ih, iw))
        s = pm.input(name="s", shape=(c,))
        mean = pm.input(name="mean", shape=(c,))
        var = pm.input(name="var", shape=(c,))
        bias = pm.input(name="bias", shape=(c,))
        y = pm.output(name="y", shape=(n, c, ih, iw))
        pm.batch_norm(x, s, bias, mean, var, y, shape=x_shape)


    if coarse:
        in_info, keys, out_info = batchnorm_datagen(x_shape)
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3]}
        shape_val_pass = pm.NormalizeGraph(shape_dict)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = batchnorm_datagen(x_shape, lowered=True)
        return new_graph, in_info, out_info, keys

def batchnorm_datagen(x_shape, lowered=False):
    input_info = {}
    input_info['x'] = np.random.randint(-3,3, x_shape)
    input_info['s'] = np.random.randint(-3,3, x_shape[1])
    input_info['mean'] = np.random.randint(-3,3, x_shape[1])
    input_info['var'] = np.random.randint(-3,3, x_shape[1])
    input_info['bias'] = np.random.randint(-3,3, x_shape[1])
    out_info = {}
    out_info['y'] = np_batchnorm(**input_info)
    return input_info, ['y'], out_info

def np_batchnorm(x=None, s=None, bias=None, mean=None, var=None, epsilon=1e-5):
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)

    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

def global_avg_pool(x_shape, coarse=False, debug_matrix=False):
    with pm.Node(name="conv") as graph:
        n = pm.parameter(name="n")
        c = pm.parameter(name="ic")
        ih = pm.parameter(name="ih")
        iw = pm.parameter(name="iw")
        x = pm.input(name="x", shape=(n, c, ih, iw))
        y = pm.output(name="y", shape=(n, c, 1, 1))
        pm.global_avg_pool(x, y, shape=(x_shape[0], x_shape[1], 1,1))


    if coarse:
        in_info, keys, out_info = global_avg_pool_datagen(x_shape)
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3]}
        shape_val_pass = pm.NormalizeGraph(shape_dict)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = global_avg_pool_datagen(x_shape, lowered=True)
        return new_graph, in_info, out_info, keys

def global_avg_pool_datagen(x_shape, lowered=False):
    input_info = {}
    input_info['x'] = np.random.randint(-3,3, x_shape)

    out_info = {}
    out_info['y'] = np_global_avg_pool(**input_info)
    return input_info, ['y'], out_info

def np_global_avg_pool(x=None, epsilon=1e-5):
    spatial_shape = np.ndim(x) - 2
    y = np.average(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = np.expand_dims(y, -1)
    return y


def lrn(x_shape, alpha, beta, bias, nsize, coarse=False, debug_matrix=False):

    with pm.Node(name="lrn") as graph:
        n = pm.parameter(name="n")
        c = pm.parameter(name="ic")
        ih = pm.parameter(name="ih")
        iw = pm.parameter(name="iw")
        x = pm.input(name="x", shape=(n, c, ih, iw))
        y = pm.output(name="y", shape=(n, c, ih, iw))
        pm.lrn(x, y, alpha, beta, bias, nsize, shape=x_shape)

    if coarse:
        in_info, keys, out_info = lrn_datagen(x_shape, alpha, beta, bias, nsize)
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3]}
        shape_val_pass = pm.NormalizeGraph(shape_dict)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = lrn_datagen(x_shape, alpha, beta, bias, nsize, lowered=True)
        return new_graph, in_info, out_info, keys

def lrn_datagen(x_shape, alpha, beta, bias, nsize, lowered=False):
    input_info = {}
    input_info['x'] = np.random.randn(*x_shape).astype(np.int32)
    input_info['alpha'] = alpha
    input_info['beta'] = beta
    input_info['bias'] = bias
    input_info['nsize'] = nsize
    out_info = {}
    out_info['y'] = np_lrn(**input_info)
    input_info.pop('alpha')
    input_info.pop('beta')
    input_info.pop('bias')
    input_info.pop('nsize')
    return input_info, ['y'], out_info

def np_lrn(x=None, alpha=None, beta=None, bias=None, nsize=None):
    square_sum = np.zeros(x.shape)
    lradius = (nsize//2)
    uradius = (nsize//2)
    if len(x.shape) > 3:
        for n, c, h, w in np.ndindex(x.shape):
            square_sum[n, c, h, w] = np.sum(x[n,
                                         max(0, c - lradius):min(5, c + uradius + 1),
                                         h,
                                         w] ** 2)
    else:
        for c, h, w in np.ndindex(x.shape):
            square_sum[c, h, w] = np.sum(x[max(0, c - int(np.floor((nsize - 1) / 2))):min(5, c + int(
                                             np.ceil((nsize - 1) / 2)) + 1),
                                         h,
                                         w] ** 2)
    y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
    return y

def max_pool(x_shape, coarse=False, debug_matrix=False):
    with pm.Node(name="conv") as graph:
        n = pm.parameter(name="n")
        c = pm.parameter(name="ic")
        ih = pm.parameter(name="ih")
        iw = pm.parameter(name="iw")
        x = pm.input(name="x", shape=(n, c, ih, iw))
        y = pm.output(name="y", shape=(n, c, 1, 1))
        pm.max_pool(x, y, shape=(x_shape[0], x_shape[1], 1,1))


    if coarse:
        in_info, keys, out_info = global_avg_pool_datagen(x_shape)
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": x_shape[0], "ic": x_shape[1], "ih": x_shape[2], "iw": x_shape[3]}
        shape_val_pass = pm.NormalizeGraph(shape_dict)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = global_avg_pool_datagen(x_shape, lowered=True)
        return new_graph, in_info, out_info, keys

def max_pool_datagen(x_shape, lowered=False):
    input_info = {}
    input_info['x'] = np.random.randint(-3,3, x_shape)

    out_info = {}
    out_info['y'] = np_global_avg_pool(**input_info)
    return input_info, ['y'], out_info

def np_max_pool(x=None, epsilon=1e-5):
    spatial_shape = np.ndim(x) - 2
    y = np.average(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = np.expand_dims(y, -1)
    return y

def np_dense(x, w):
    return sigmoid(w.dot(x))

def dense_data_gen(x_shape, w_shape, lowered=False, debug_matrix=False):
    input_info = {}
    input_info["n"] = w_shape[0]
    input_info["m"] = w_shape[1]
    if debug_matrix:
        input_info["x"] = np.arange(0, (np.prod(x_shape))).reshape(x_shape)
    else:
        input_info["x"] = np.random.randint(0, 10, x_shape)

    if debug_matrix:
        input_info["w"] = np.arange(0, (np.prod(w_shape))).reshape(w_shape)
    else:
        input_info["w"] = np.random.randint(0, 10, w_shape)
    out = np_dense(input_info["x"], input_info["w"])
    if lowered:
        all_keys = []
        i = np.arange(0, input_info["m"])
        j = np.arange(0, input_info["n"])
        all_pairs = list(product(*(j,i)))

        for p in all_pairs:
            w_key = f"w/w{p}"
            all_keys.append(w_key)
            input_info[w_key] = input_info["w"][p]
            input_info[f"x/x({p[1]},)"] = input_info["x"][p[1]]
        input_info.pop("w")
        input_info.pop("x")
        all_keys = [f"y/y({p},)" for p in range(input_info["n"])]
    else:
        all_keys = "w"


    out_info = {"y": out}

    return input_info, all_keys, out_info

def multi_dense_data_gen(x1_shape, w1_shape, w2_shape, lowered=False, debug_matrix=False):
    input_info = {}
    input_info["n"] = w1_shape[0]
    input_info["m"] = w1_shape[1]
    input_info["p"] = w2_shape[0]
    if debug_matrix:
        input_info["x1"] = np.arange(0, (np.prod(x1_shape))).reshape(x1_shape)
    else:
        input_info["x1"] = np.random.randint(0, 10, x1_shape)

    if debug_matrix:
        input_info["w1"] = np.arange(0, (np.prod(w1_shape))).reshape(w1_shape)
    else:
        input_info["w1"] = np.random.randint(0, 10, w1_shape)

    if debug_matrix:
        input_info["w2"] = np.arange(0, (np.prod(w2_shape))).reshape(w2_shape)
    else:
        input_info["w2"] = np.random.randint(0, 10, w2_shape)
    out_info = {}
    out_info["y1"] = np_dense(input_info["x1"], input_info["w1"])
    out_info["y"] = np_dense(out_info["y1"], input_info["w2"])
    if lowered:
        i = np.arange(0, input_info["m"])
        j = np.arange(0, input_info["n"])
        k = np.arange(0, input_info["p"])
        y1_all_pairs = list(product(*(j, i)))
        y2_all_pairs = list(product(*(k, j)))

        for p in y1_all_pairs:
            w_key = f"w1/w1{p}"
            input_info[w_key] = input_info["w1"][p]
            input_info[f"x1/x1({p[1]},)"] = input_info["x1"][p[1]]
        for p in y2_all_pairs:
            w_key = f"w2/w2{p}"
            input_info[w_key] = input_info["w2"][p]
        input_info.pop("w1")
        input_info.pop("w2")
        input_info.pop("x1")
        all_keys = [f"y/y({p},)" for p in range(input_info["p"])]
    else:
        all_keys = "y"


    return input_info, all_keys, out_info

def two_layer_dense(x1_shape, w1_shape, w2_shape, coarse=False, debug_matrix=False):

    with pm.Node("multi_dim") as graph:
        m = pm.parameter("m")
        n = pm.parameter("n")
        p = pm.parameter("p")
        x1 = pm.input(name="x1", shape=(m))
        w1 = pm.state(name="w1", shape=(n, m))
        w2 = pm.state(name="w2", shape=(p, n))
        y = pm.output(name="y", shape=(p))
        y1 = pm.output(name="y1", shape=(n))
        pm.dense_sigmoid(x1, w1, y1)
        pm.dense_sigmoid(y1, w2, y)

    if coarse:
        in_info, keys, out_info = multi_dense_data_gen(x1_shape, w1_shape, w2_shape, debug_matrix=debug_matrix)
        return graph, in_info, out_info, keys
    else:
        shape_dict = {"m": w1_shape[1], "n": w1_shape[0], "p": w2_shape[0]}
        shape_val_pass = pm.NormalizeGraph(shape_dict)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = multi_dense_data_gen(x1_shape, w1_shape, w2_shape, lowered=True, debug_matrix=debug_matrix)
        return new_graph, in_info, out_info, keys

def dense(x_shape, w_shape, coarse=False, debug_matrix=False):
    m = pm.parameter("m")
    n = pm.parameter("n")
    x = pm.input(name="x", shape=(m))
    w = pm.state(name="w", shape=(n,m))
    y = pm.output(name="y", shape=(n))
    graph = pm.dense_sigmoid(x, w, y)
    if coarse:
        in_info, keys, out_info = dense_data_gen(x_shape, w_shape, debug_matrix=debug_matrix)
        return graph, in_info, out_info, keys
    else:
        shape_dict = {"m": w_shape[1], "n": w_shape[0]}
        shape_val_pass = pm.NormalizeGraph(shape_dict)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = dense_data_gen(x_shape, w_shape, lowered=True, debug_matrix=debug_matrix)
        return new_graph, in_info, out_info, keys

def tvm_lenet(num_classes=10, data_shape=(1, 1, 32, 32),
               dtype='float32', alpha=1.0, is_shallow=False):
    from tvm import relay
    from tvm.relay.testing import layers

    """Function to construct a Lenet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    conv1 = layers.conv2d(data=data,channels=6, kernel_size=(5,5), name='conv1')
    conv1 = relay.nn.relu(conv1)
    pool2 = relay.nn.avg_pool2d(conv1, pool_size=(2,2), strides=(2,2))
    conv3 = layers.conv2d(data=pool2, channels=16, kernel_size=(5,5), name='conv3')
    conv3 = relay.nn.relu(conv3)
    pool4 = relay.nn.avg_pool2d(conv3, pool_size=(2,2), strides=(2,2))
    flattened5 = relay.nn.batch_flatten(pool4)

    fcw5 = relay.var('fc5_weight')
    fc5 = relay.nn.dense(data=flattened5, weight=fcw5, units=120)
    fc5 = relay.nn.relu(fc5)

    fcw6 = relay.var('fc6_weight')
    fc6 = relay.nn.dense(data=fc5, weight=fcw6, units=84)
    fc6 = relay.nn.relu(fc6)

    fcw7 = relay.var('fc7_weight')
    fc7= relay.nn.dense(data=fc6, weight=fcw7, units=num_classes)
    fc7 = relay.nn.relu(fc7)

    softmax = relay.nn.softmax(data=fc7)
    fn = relay.Function(relay.analysis.free_vars(softmax), softmax)
    return fn

def np_relu(x):
    return x * (x > 0)

def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def np_lenet(lowered=False):
    input_info = {}
    input_info["data"] = np.random.randint(0, 5, (1, 1, 32, 32))
    input_info["w1"] = np.random.randint(0, 5, (6, 1, 5, 5))
    input_info["b1"] = np.random.randint(0, 5, 6)
    input_info["s1"] = 1
    input_info["p1"] = 0

    input_info["w2"] = np.random.randint(0, 5, (16, 6, 5, 5))
    input_info["b2"] = np.random.randint(0, 5, 16)
    input_info["s2"] = 1
    input_info["p2"] = 0

    input_info["w6"] = np.random.randint(0, 5, (120, 400))
    input_info["w7"] = np.random.randint(0, 5, (84, 120))
    input_info["w8"] = np.random.randint(0, 5, (10, 84))
    out_info = {}
    c1_params = {"stride": input_info["s1"], "pad": input_info["p1"]}
    out_info["c1"] = conv3d(input_info["data"], input_info["w1"], input_info["b1"], c1_params)[0]
    out_info["a1"] = np_relu(out_info["c1"])
    out_info["l1"] = pooling(out_info["a1"], 2, 2, 0, 2)

    c2_params = {"stride": input_info["s2"], "pad": input_info["p2"]}
    out_info["c2"] = conv3d(out_info["l1"], input_info["w2"], input_info["b2"], c2_params)[0]
    out_info["a2"] = np_relu(out_info["c2"])
    out_info["l2"] = pooling(out_info["a2"], 2, 2, 0, 2)

    out_info["f5"] = batch_flatten(out_info["l2"])
    out_info["f6"] = input_info["w6"].dot(out_info["f5"])
    out_info["a6"] = np_relu(out_info["f6"])

    out_info["f7"] = input_info["w7"].dot(out_info["a6"])
    out_info["a7"] = np_relu(out_info["f7"])

    out_info["f8"] = input_info["w8"].dot(out_info["a7"])
    out_info["a8"] = np_relu(out_info["f8"])
    out_info["sm"] = np_softmax(out_info["a8"])
    if lowered:
        for k in list(input_info.keys()):
            if hasattr(input_info[k], "shape"):
                pairs = list(product(*tuple([np.arange(i) for i in input_info[k].shape])))
                for p in pairs:
                    input_info[f"{k}/{k}{p}"] = input_info[k][p]
                input_info.pop(k)
        key = []
        for k in list(out_info.keys()):
            if hasattr(out_info[k], "shape"):
                pairs = list(product(*tuple([np.arange(i) for i in out_info[k].shape])))

                for p in pairs:
                    if k == "sm":
                        key.append(f"{k}/{k}{p}")
                    out_info[f"{k}/{k}{p}"] = out_info[k][p]

                if k != "sm":
                    out_info.pop(k)
    else:
        key = "sm"
    return input_info, key, out_info

def batch_flatten(x):
    return x.reshape(-1)

def lenet(lenet_type="lenet5", coarse=True, debug=False):

    with pm.Node(name="lenet") as graph:
        n = pm.parameter(name="n")
        c = pm.parameter(name="ic")
        ih = pm.parameter(name="ih")
        iw = pm.parameter(name="iw")
        nf1 = pm.parameter(name="nf1")
        kh1 = pm.parameter(name="kh1")
        kw1 = pm.parameter(name="kw1")
        data = pm.input(name="data", shape=(n, c, ih, iw))
        w1 = pm.state(name="w1", shape=(nf1, c, kh1, kw1))
        b1 = pm.state(name="b1", shape=(nf1))

        s1 = pm.parameter(name="s1")
        p1 = pm.parameter(name="p1")
        c1 = pm.output(name="c1")
        a1 = pm.output(name="a1")
        l1 = pm.output(name="l1")

        pm.conv_bias(data, w1, b1, c1, s1, p1)
        pm.relu(c1, a1)
        pm.avg_pool2d(a1, l1, 2, 2, 2, 0)

        nf2 = pm.parameter(name="nf2")
        kh2 = pm.parameter(name="kh2")
        kw2 = pm.parameter(name="kw2")
        w2 = pm.state(name="w2", shape=(nf2, nf1, kh2, kw2))
        b2 = pm.state(name="b2", shape=(nf2))
        s2 = pm.parameter(name="s2")
        p2 = pm.parameter(name="p2")
        c2 = pm.output(name="c2")
        a2 = pm.output(name="a2")
        l2 = pm.output(name="l2")

        pm.conv_bias(l1, w2, b2, c2, s2, p2)
        pm.relu(c2, a2)
        pm.avg_pool2d(a2, l2, 2, 2, 2, 0)

        f5 = pm.output(name="f5")
        pm.batch_flatten(l2, f5)

        f6 = pm.output(name="f6")
        m6 = pm.parameter(name="m6")
        n6 = pm.parameter(name="n6")
        w6 = pm.state(name="w6", shape=(n6, m6))
        a6 = pm.output(name="a6")
        pm.dense(f5, w6, f6)
        pm.relu1d(f6, a6)

        f7 = pm.output(name="f7")
        m7 = pm.parameter(name="m7")
        n7 = pm.parameter(name="n7")
        w7 = pm.state(name="w7", shape=(n7, m7))
        a7 = pm.output(name="a7")
        pm.dense(a6, w7, f7)
        pm.relu1d(f7, a7)

        f8 = pm.output(name="f8")
        m8 = pm.parameter(name="m8")
        n8 = pm.parameter(name="n8")
        w8 = pm.state(name="w8", shape=(n8, m8))
        a8 = pm.output(name="a8")
        pm.dense(a7, w8, f8)
        pm.relu1d(f8, a8)

        out = pm.output(name="sm")
        pm.softmax(a8, out)
    if coarse:
        in_info, keys, out_info = np_lenet()
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": 1, "ic": 1, "ih": 32, "iw": 32,
                      "nf1": 6, "kh1": 5, "kw1": 5, "s1": 1, "p1": 0,
                      "nf2": 16, "kh2": 5, "kw2": 5, "s2": 1, "p2": 0,
                      "m6": 400, "n6": 120, "m7": 120, "n7": 84, "m8": 84, "n8": 10
                      }
        shape_val_pass = pm.NormalizeGraph(shape_dict, debug=debug)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = np_lenet(lowered=True)
        return new_graph, in_info, out_info, keys

def fft(m_=3, coarse=False):
    with pm.Node(name="fft") as graph:
        m = pm.parameter("m")
        x = pm.input("x", shape=(m))
        i = pm.index(0, (m - 1).set_name("m-1"), name="i")

    if coarse:
        in_info, keys, out_info = fft_data_gen(m=m_)
        return graph, in_info, out_info, keys
    else:
        shape_val_pass = pm.NormalizeGraph({"m": m_})
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = fft_data_gen(m=m_, lowered=True)
        return new_graph, in_info, out_info, keys

def np_fft(input_info):
    out_info = {}
    out_info["fft_x"] = np.fft.fft(input_info["x"])
    return out_info

def fft_data_gen(m, lowered=False):
    input_info = {}
    input_info["x"] = np.random.randint(-5, 5, m)
    out_info = np_fft(input_info)
    if lowered:
        all_keys = []
        for p in range(m):
            input_info[f"x/x({p},)"] = input_info["x"][p]
        input_info.pop("x")
    else:
        all_keys = "x"

    return input_info, all_keys, out_info


def bit_reversal_indices(x):
    n = x.shape[0]
    log2n = np.int(np.log2(n))
    rev_ns = []
    kmat = np.array([[i>>j for j in range(log2n)] for i in range(n)])
    shifter = lambda a, b: (a<<1) | (b & 1)
    np_shifter = np.frompyfunc(shifter, 2, 1)
    test_out = np_shifter.reduce(kmat, axis=(1,), initial=0)
    x_out = np.empty(x.shape)
    for i in range(x.shape[0]):
        x_out[test_out[i]] = x[i]
    return x_out


def fft_parallelized():
    with pm.Node("fft") as graph:
        N = pm.parameter("N")
        x = pm.input("x", shape=(N,))
        n1 = pm.index(0, N-1, name="n1")
        n2 = pm.index(0, N-1, name="n2")

        X = pm.output("X", shape=(N,))

        M = pm.output("M", shape=(N,N))
        M[n1, n2] = (n1 * n2).set_name("test")
        M[n1, n2] = pm.exp(-2j * np.pi * M[n1,n2]/N)
        X[n1] = pm.sum([n2], M[n1, n2]* x[n2])
    return graph


def test_fft2(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] / 2]
        X_odd = X[:, X.shape[1] / 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()

def manual_fft(x):
    m = np.zeros((x.shape[0],x.shape[0]))
    for i, j in np.ndindex(m.shape):
        m[i][j] = i*j

    return m

def unwound_fft(x_):

    np_res = np.abs(np.fft.fft(x_))
    graph = fft_parallelized()
    pm_res = np.abs(graph("X", {"x": x_}))
    return pm_res, np_res

def GP_Model_BO(X, Y):
    import gpflow
    k1 = gpflow.kernels.Matern32(1, active_dims=[0], ard=True)
    m = gpflow.gpr.GPR(X, Y.T, k1)
    m.kern.lengthscales = np.std(X)
    m.kern.lengthscales = np.std(X)
    m.kern.variance = np.std(Y) / np.sqrt(2)
    m.likelihood.variance = np.std(Y) / np.sqrt(2)
    return m

def matern32(X, Y):
    import gpflow
    k1 = gpflow.kernels.Matern32(np.std(Y)/np.sqrt(2), active_dims=[0], lengthscales=[np.std(X)])
    ll_var = np.std(Y)/np.sqrt(2)
    m = gpflow.models.GPR(X, Y.T, k1, noise_variance=ll_var)
    return m




def yolo_convolution(tensor_in, filters=32, kernel_size=3,
        batch_normalize=True, act='leakyReLU',
        c_dtype=None, w_dtype=None,
        s_dtype=None, bn_dtype=None):
    from dnnweaver2.graph import Graph, get_default_graph
    from dnnweaver2.tensor_ops.cnn import conv2D, maxPool, batch_norm, leakyReLU
    import logging
    from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint

    from dnnweaver2 import get_tensor
    input_channels = tensor_in.shape[-1]

    weights = get_tensor(shape=(filters, kernel_size, kernel_size, input_channels),
                         name='weights',
                         dtype=w_dtype)
    biases = get_tensor(shape=(filters),
                         name='biases',
                         dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    conv = conv2D(tensor_in, weights, biases, pad='SAME', dtype=c_dtype)

    if batch_normalize:
        with get_default_graph().name_scope('batch_norm'):
            mean = get_tensor(shape=(filters), name='mean', dtype=FixedPoint(16,c_dtype.frac_bits))
            scale = get_tensor(shape=(filters), name='scale', dtype=s_dtype)
            bn = batch_norm(conv, mean=mean, scale=scale, dtype=bn_dtype)
    else:
        bn = conv

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(bn, dtype=bn.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = bn
    else:
        raise ValueError('Unknown activation type {}'.format(act))

    return act


def tiny_yolo(train=False, debug=False):
    from dnnweaver2.graph import Graph, get_default_graph
    from dnnweaver2.tensor_ops.cnn import conv2D, maxPool, batch_norm, leakyReLU
    import logging
    from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint

    from dnnweaver2 import get_tensor
    g = Graph('YOLOv2-Test: 16-bit', dataset='imagenet', log_level=logging.INFO)
    batch_size = 1
    with g.as_default():

        with g.name_scope('inputs'):
            i = get_tensor(shape=(batch_size,416,416,3), name='data', dtype=FQDtype.FXP16, trainable=False)

        with g.name_scope('conv0'):
            conv0 = yolo_convolution(i, filters=16, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,12),
                    s_dtype=FixedPoint(16,9), bn_dtype=FixedPoint(16,8))
        with g.name_scope('pool0'):
            pool0 = maxPool(conv0, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv1'):
            conv1 = yolo_convolution(pool0, filters=32, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,8),
                    s_dtype=FixedPoint(16,14), bn_dtype=FixedPoint(16,8))
        with g.name_scope('pool1'):
            pool1 = maxPool(conv1, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv2'):
            conv2 = yolo_convolution(pool1, filters=64, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    # batch_normalize=False, act='linear',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,10),
                    s_dtype=FixedPoint(16,13), bn_dtype=FixedPoint(16,9))
        with g.name_scope('pool2'):
            pool2 = maxPool(conv2, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv3'):
            conv3 = yolo_convolution(pool2, filters=128, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,10),
                    s_dtype=FixedPoint(16,13), bn_dtype=FixedPoint(16,10))
        with g.name_scope('pool3'):
            pool3 = maxPool(conv3, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv4'):
            conv4 = yolo_convolution(pool3, filters=256, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,11),
                    s_dtype=FixedPoint(16,13), bn_dtype=FixedPoint(16,10))
        with g.name_scope('pool4'):
            pool4 = maxPool(conv4, pooling_kernel=(1,2,2,1), stride=(1,2,2,1), pad='VALID')

        with g.name_scope('conv5'):
            conv5 = yolo_convolution(pool4, filters=512, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,12),
                    s_dtype=FixedPoint(16,13), bn_dtype=FixedPoint(16,11))
        with g.name_scope('pool5'):
            pool5 = maxPool(conv5, pooling_kernel=(1,2,2,1), stride=(1,1,1,1), pad=((0,0),(0,1),(0,1),(0,0)))

        with g.name_scope('conv6'):
            conv6 = yolo_convolution(pool5, filters=1024, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,12),
                    s_dtype=FixedPoint(16,11), bn_dtype=FixedPoint(16,9))

        with g.name_scope('conv7'):
            conv7 = yolo_convolution(conv6, filters=1024, kernel_size=3,
                    batch_normalize=True, act='leakyReLU',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,11),
                    s_dtype=FixedPoint(16,14), bn_dtype=FixedPoint(16,12))

        with g.name_scope('conv8'):
            conv8 = yolo_convolution(conv7, filters=125, kernel_size=1,
                    batch_normalize=False, act='linear',
                    w_dtype=FixedPoint(16,14), c_dtype=FixedPoint(16,11))

    return g



def pm_tiny_yolo(coarse=False):
    with pm.Node(name="tiny-yolo") as graph:
        n = pm.parameter(name="n")
        c = pm.parameter(name="ic")
        ih = pm.parameter(name="ih")
        iw = pm.parameter(name="iw")
        nf1 = pm.parameter(name="nf1")
        kh1 = pm.parameter(name="kh1")
        kw1 = pm.parameter(name="kw1")
        data = pm.input(name="data", shape=(n, c, ih, iw))
        w1 = pm.state(name="w1", shape=(nf1, c, kh1, kw1))
        b1 = pm.state(name="b1", shape=(nf1))

        s1 = pm.parameter(name="s1")
        p1 = pm.parameter(name="p1")
        c1 = pm.output(name="c1")
        a1 = pm.output(name="a1")
        l1 = pm.output(name="l1")

        pm.conv_bias(data, w1, b1, c1, s1, p1)
        pm.leaky_relu(c1, a1)
        pm.batch_norm(a1, l1, 2, 2, 2, 0)

        nf2 = pm.parameter(name="nf2")
        kh2 = pm.parameter(name="kh2")
        kw2 = pm.parameter(name="kw2")
        w2 = pm.state(name="w2", shape=(nf2, nf1, kh2, kw2))
        b2 = pm.state(name="b2", shape=(nf2))
        s2 = pm.parameter(name="s2")
        p2 = pm.parameter(name="p2")
        c2 = pm.output(name="c2")
        a2 = pm.output(name="a2")
        l2 = pm.output(name="l2")

        pm.conv_bias(l1, w2, b2, c2, s2, p2)
        pm.relu(c2, a2)
        pm.avg_pool2d(a2, l2, 2, 2, 2, 0)

        f5 = pm.output(name="f5")
        pm.batch_flatten(l2, f5)

        f6 = pm.output(name="f6")
        m6 = pm.parameter(name="m6")
        n6 = pm.parameter(name="n6")
        w6 = pm.state(name="w6", shape=(n6, m6))
        a6 = pm.output(name="a6")
        pm.dense(f5, w6, f6)
        pm.relu1d(f6, a6)

        f7 = pm.output(name="f7")
        m7 = pm.parameter(name="m7")
        n7 = pm.parameter(name="n7")
        w7 = pm.state(name="w7", shape=(n7, m7))
        a7 = pm.output(name="a7")
        pm.dense(a6, w7, f7)
        pm.relu1d(f7, a7)

        f8 = pm.output(name="f8")
        m8 = pm.parameter(name="m8")
        n8 = pm.parameter(name="n8")
        w8 = pm.state(name="w8", shape=(n8, m8))
        a8 = pm.output(name="a8")
        pm.dense(a7, w8, f8)
        pm.relu1d(f8, a8)

        out = pm.output(name="sm")
        pm.softmax(a8, out)
    if coarse:
        in_info, keys, out_info = np_lenet()
        return graph, in_info, out_info, keys
    else:

        shape_dict = {"n": 1, "ic": 1, "ih": 32, "iw": 32,
                      "nf1": 6, "kh1": 5, "kw1": 5, "s1": 1, "p1": 0,
                      "nf2": 16, "kh2": 5, "kw2": 5, "s2": 1, "p2": 0,
                      "m6": 400, "n6": 120, "m7": 120, "n7": 84, "m8": 84, "n8": 10
                      }
        shape_val_pass = pm.NormalizeGraph(shape_dict, debug=debug)
        new_graph = shape_val_pass(graph)
        in_info, keys, out_info = np_lenet(lowered=True)
        return new_graph, in_info, out_info, keys
