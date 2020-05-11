from onnx import load, numpy_helper, helper, shape_inference
import pathlib
import numpy as np
from .node_definitions import NODE_NAMES
import polymath as pm

def from_onnx(filepath, infer_shapes=True):
    onnx_proto, graph_name = load_onnx_proto(filepath)
    attr = get_model_attributes(onnx_proto)
    if infer_shapes:
        onnx_graph = shape_inference.infer_shapes(onnx_proto).graph
    else:
        onnx_graph = onnx_proto.graph
    for n in onnx_graph.node:
        if n.op_type not in NODE_NAMES and n.name not in NODE_NAMES:
            raise RuntimeError(f"Support for {n.op_type} or {n.name} is not currently included in PolyMath")

    domain = "nn"
    for a in attr["opset_import"]:
        if a.domain == "ai.onnx.ml":
            domain = "ml"
            break

    if domain == "ml":
        graph = generate_ml_mdfg(onnx_graph)
    else:
        graph = generate_nn_mdfg(onnx_graph)

    return graph

def load_onnx_proto(filepath):
    graph_name = pathlib.Path(filepath).stem
    return load(filepath), graph_name


def get_model_attributes(model):
    kwargs = {des.name: getattr(model, des.name) for des in model.DESCRIPTOR.fields if des.name != "graph"}
    return kwargs

def generate_ml_mdfg(onnx_graph):
    names = [des.name for des in onnx_graph.DESCRIPTOR.fields]
    graph_name = getattr(onnx_graph, "name")
    assert len(onnx_graph.input) == 1
    inp_shape = get_value_info_shape(onnx_graph.input[0])
    node_attr = get_attributes(onnx_graph.node[0])
    if onnx_graph.node[0].op_type == "LinearRegressor":
        graph = gen_from_shape("linear", inp_shape[0])
    elif onnx_graph.node[0].op_type == "LinearClassifier":
        assert node_attr["post_transform"] == "LOGISTIC"
        graph = gen_from_shape("logistic", inp_shape[0])
    elif onnx_graph.node[0].op_type == "SVMClassifier":
        graph = gen_from_shape("svm", inp_shape[0])
    else:
        raise ValueError(f"Unsupported graph type: {onnx_graph.node[0].op_type}")
    return graph

def get_initializers(initializers):
    init_dict = {}
    for i in initializers:
        val = numpy_helper.to_array(i)
        if len(val.shape) == 0:
            # val = np.asarray([np.int(val)])
            val = np.int(val)
        init_dict[i.name] = val
    return init_dict

def generate_nn_mdfg(onnx_graph):
    names = [des.name for des in onnx_graph.DESCRIPTOR.fields]
    graph_name = getattr(onnx_graph, "name")
    initializers = get_initializers(onnx_graph.initializer)
    mgdfg = pm.Node(name=graph_name)
    node_info = {}
    for i in onnx_graph.input:
        assert i.name not in node_info
        if i.name in initializers:
            if itercheck(initializers[i.name]):
                node_info[i.name] = pm.variable(initializers[i.name], name=i.name, shape=get_value_info_shape(i), graph=mgdfg)
            else:
                node_info[i.name] = pm.parameter(name=i.name, default=initializers[i.name], graph=mgdfg)
        else:
            node_info[i.name] = pm.input(name=i.name, shape=get_value_info_shape(i), graph=mgdfg)

    for o in onnx_graph.output:
        assert o.name not in node_info
        if o.name in initializers:
            node_info[o.name] = pm.variable(initializers[o.name], name=o.name, shape=get_value_info_shape(o), graph=mgdfg)
        else:
            node_info[o.name] = pm.output(name=o.name, shape=get_value_info_shape(o), graph=mgdfg)

    for v in onnx_graph.value_info:
        assert v.name not in node_info
        if v.name in initializers:
            node_info[v.name] = pm.variable(initializers[v.name], name=v.name, shape=get_value_info_shape(v), graph=mgdfg)
        else:
            node_info[v.name] = {"name": v.name, "shape": get_value_info_shape(v)}

    for n in onnx_graph.node:
        assert n.op_type in NODE_NAMES
        _ = convert_node(n, mgdfg, node_info)

    return mgdfg

def convert_node(onnx_node, mgdfg, node_info):
    name = onnx_node.name
    args = []

    for i in onnx_node.input:
        assert i in mgdfg.nodes
        args.append(mgdfg.nodes[i])

    assert len(onnx_node.output) == 1 and onnx_node.output[0] in node_info

    o_name = onnx_node.output[0]
    if isinstance(node_info[o_name], dict):
        o_shape = node_info[o_name]["shape"]
        attributes = get_attributes(onnx_node)
        args = tuple(args + list(attributes.values()) + list(o_shape))
        with mgdfg:
            new_node = NODE_NAMES[onnx_node.op_type](*args, name=o_name)
    else:
        o_shape = node_info[o_name].shape
        attributes = get_attributes(onnx_node)
        args = tuple(args + list(attributes.values()) + list(o_shape))
        indices = tuple([pm.index(0, s-1, name=f"{o_name}_i", graph=mgdfg) for s in o_shape])
        with mgdfg:
            new_node = NODE_NAMES[onnx_node.op_type](*args)
            node_info[o_name][indices] = new_node[indices]

    return mgdfg

def _print_proto_fields(pb):
    print(f"{pb} fields : {[n.name for n in pb.DESCRIPTOR.fields]}")

def get_value_info_shape(vi):
    ret = tuple([dim.dim_value for dim in vi.type.tensor_type.shape.dim if dim.dim_value > 0])
    return ret if len(ret) > 0 else (1,)

def get_attributes(node):
    attributes = {}
    for a in node.attribute:
        val = helper.get_attribute_value(a)
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        elif isinstance(val, list):
            if len(val) > 1:
                val = np.asarray(val)
            else:
                val = val[0]

        attributes[a.name] = val

    return attributes

def itercheck(obj):
    if isinstance(obj, np.ndarray):
        return len(obj.shape) > 0
    else:
        return hasattr(obj, '__iter__') and not isinstance(obj, str)

def gen_from_shape(graph_type, input_shape, params=None):
    if graph_type == "linear":
        x = pm.input(name="x", shape=input_shape)
        w = pm.state(name="w", shape=input_shape)
        y = pm.input(name="y")
        mu = pm.parameter(name="mu", default=1.0)
        m = pm.parameter(name="m", default=input_shape)
        return pm.linear_regressor_train(x, w, y, mu, m, name="linear_regressor")
    elif graph_type == "logistic":
        x = pm.input(name="x", shape=input_shape)
        w = pm.state(name="w", shape=input_shape)
        y = pm.input(name="y")
        mu = pm.parameter(name="mu", default=1.0)
        m = pm.parameter(name="m", default=input_shape)
        return pm.logistic_regressor_train(x, w, y, mu, m, name="logistic_regressor")
    elif graph_type == "svm":
        x = pm.input(name="x", shape=input_shape)
        w = pm.state(name="w", shape=input_shape)
        y = pm.input(name="y")
        mu = pm.parameter(name="mu", default=1.0)
        m = pm.parameter(name="m", default=input_shape)
        return pm.svm_classifier_train(x, w, y, mu, m, name="svm_classifier")

