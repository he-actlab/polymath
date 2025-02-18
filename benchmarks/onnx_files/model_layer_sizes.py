from collections import defaultdict
import pathlib
import onnx
CWD = pathlib.Path(f"{__file__}").parent
import numpy as np

def load_model(model_name):
    path = f"{CWD}/{model_name}"
    return onnx.load(path)

def compute_size(n, shape_map, node_map, include_output=True):
    sizes = [shape_map[i] for i in n.input]
    if include_output:
        sizes += [shape_map[o] for o in n.output]
    return sizes

def get_node_shape(node):
    ret = []
    if isinstance(node, onnx.TensorProto):
        for i in node.dims:
            ret.append(i)
    else:
        assert isinstance(node, onnx.ValueInfoProto)
        for i, dim in enumerate(node.type.tensor_type.shape.dim):
            if hasattr(dim, 'dim_param') and dim.dim_param:
                d_val = dim.dim_param
            else:
                assert dim.dim_value > 0
                d_val = dim.dim_value
            ret.append(d_val)
    return tuple(ret)

def collect_uses(graph):
    output_uses = {}
    shapes = {}
    node_map = {}
    for i in graph.initializer:
        assert i.name not in output_uses
        shapes[i.name] = get_node_shape(i)
        output_uses[i.name] = 0

    for i in graph.input:
        if i.name in output_uses:
            continue
        shapes[i.name] = get_node_shape(i)
        output_uses[i.name] = 0

    for v in graph.value_info:
        if v.name in output_uses:
            continue
        shapes[v.name] = get_node_shape(v)
        output_uses[v.name] = 0

    for o in graph.output:
        if o.name in output_uses:
            continue
        shapes[o.name] = get_node_shape(o)
        output_uses[o.name] = 0

    for n in graph.node:
        for i in n.input:
            output_uses[i] += 1
        for o in n.output:
            output_uses[o] = 0
            node_map[o] = n

    return output_uses, node_map, shapes


def compute_singular_shape(size_list):
    total = 0
    for s in size_list:
        total += np.prod(s)
    return total

def get_max_res_path(res_info, shape_map, node_map):
    p1_size = 0
    idx = len(res_info["path1"]) - 1
    while res_info["path1"][idx] != res_info["path1"][0]:
        if res_info["path1"][idx] == res_info["output"]:
            idx-=1
            continue
        node = node_map[res_info["path1"][idx]]
        p1_size = max(p1_size, compute_singular_shape(compute_size(node, shape_map, node_map)))
        idx -= 1

    p2_size = 0
    idx = len(res_info["path2"]) - 1
    while res_info["path2"][idx] != res_info["path2"][0]:
        if res_info["path2"][idx] == res_info["output"]:
            idx-=1
            continue
        node = node_map[res_info["path2"][idx]]
        p2_size = max(p2_size, compute_singular_shape(compute_size(node, shape_map, node_map)))
        idx -= 1
    end_node = node_map[res_info["output"]]
    out_size = compute_singular_shape(compute_size(end_node, shape_map, node_map, include_output=False))
    return max(p1_size, p2_size) + out_size


def traverse_graph(onnx_model: onnx.ModelProto):
    graph = onnx_model.graph
    output_uses, node_map, shape_map = collect_uses(graph)
    layer_shapes = {n.name: compute_size(n, shape_map, node_map) for n in graph.node}
    layer_sizes_ordered = sorted([(name, compute_singular_shape(size)) for name, size in layer_shapes.items()],key=lambda pair: pair[1])
    res_layer_calc_sizes = {}
    residual_layer_info = {}
    current_res_name = None
    for n in graph.node:
        assert len(n.output) == 1
        if output_uses[n.output[0]] > 1:
            current_res_name = f"res{len(residual_layer_info)}"
            residual_layer_info[current_res_name] = {"path1": [n.output[0]], "path2": [n.output[0]]}
        elif current_res_name is None:
            continue
        elif n.op_type == "Add":
            if residual_layer_info[current_res_name]["path1"][-1] == n.input[0]:
                assert n.input[1] == residual_layer_info[current_res_name]["path2"][-1]
            else:
                assert residual_layer_info[current_res_name]["path1"][-1] == n.input[1]
                assert residual_layer_info[current_res_name]["path2"][-1] == n.input[0]
            residual_layer_info[current_res_name]["output"] = n.output[0]
            res_layer_calc_sizes[current_res_name] = get_max_res_path(residual_layer_info[current_res_name], shape_map, node_map)
            current_res_name = None
        elif any([i == residual_layer_info[current_res_name]["path1"][-1] for i in n.input]):
            residual_layer_info[current_res_name]["path1"].append(n.output[0])
        elif any([i == residual_layer_info[current_res_name]["path2"][-1] for i in n.input]):
            residual_layer_info[current_res_name]["path2"].append(n.output[0])

    sorted_sizes = sorted([(name, value) for name, value in res_layer_calc_sizes.items()], key=lambda x: x[1])
    print(sorted_sizes[-1])
    out_node = node_map[residual_layer_info[sorted_sizes[-1][0]]["output"]]
    out_size = compute_singular_shape(compute_size(out_node, shape_map, node_map, include_output=False))//2

    for p in residual_layer_info[sorted_sizes[-1][0]]["path1"]:
        node = node_map[p]
        res = compute_singular_shape(compute_size(node, shape_map, node_map))
        print(f"{node.name}/{p}: {res}\nTotal: {res + out_size}\n")

    for p in residual_layer_info[sorted_sizes[-1][0]]["path2"]:
        node = node_map[p]
        res = compute_singular_shape(compute_size(node, shape_map, node_map))
        print(f"{node.name}/{p}: {res}\nTotal: {res + out_size}\n")

    # print(node_map[residual_layer_info[sorted_sizes[-1][0]]['path1'][0]])
    print(layer_sizes_ordered[-1])
    # print(layer_sizes_ordered[-1])

def find_largest_layers(model):
    from pprint import pprint
    graph = model.graph
    output_uses, node_map, shape_map = collect_uses(graph)
    layer_shapes = {n.name: compute_size(n, shape_map, node_map) for n in graph.node}
    layer_sizes_ordered = sorted([(name, compute_singular_shape(size), size) for name, size in layer_shapes.items()],
                                 key=lambda pair: pair[1])
    layer_bank_sizes = {"IBUF_GMEM_BANK": [],
                   "OBUF_GMEM_BANK": [],
                   "WBUF_GMEM_BANK": [],
                   "BBUF_GMEM_BANK": [],
                   "VMEM_GMEM_BANK": [],
                   }
    operand_types = ["input", "weight"]
    for l in reversed(layer_sizes_ordered):

        name = l[0].split("_")[0]
        if name in ["Conv", "Gemm"]:
            layer_bank_sizes["IBUF_GMEM_BANK"].append((np.prod(l[2][0]), l))
            layer_bank_sizes["WBUF_GMEM_BANK"].append((np.prod(l[2][1]), l))
            layer_bank_sizes["BBUF_GMEM_BANK"].append((np.prod(l[2][2]), l))
            layer_bank_sizes["OBUF_GMEM_BANK"].append((np.prod(l[2][3]), l))
        else:
            layer_bank_sizes["VMEM_GMEM_BANK"].append((l[1], l))
    max_bank_layers = {}

    for k, v in layer_bank_sizes.items():
        max_layer = max(v, key=lambda pair: pair[0])
        max_bank_layers[k] = max_layer
        max_bank_layers[k] = {}
        max_bank_layers[k]['total_elements_required'] = max_layer[0]
        max_bank_layers[k]['layer_name'] = max_layer[1][0]
        max_bank_layers[k]['total_layer_size'] = max_layer[1][1]
        max_bank_layers[k]['all_operand_shapes'] = max_layer[1][2]

    pprint(max_bank_layers)


if __name__ == "__main__":
    MODEL_NAME = "resnet18.onnx"
    model = load_model(MODEL_NAME)

    find_largest_layers(model)
    # traverse_graph(model)