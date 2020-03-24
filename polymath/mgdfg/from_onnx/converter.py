from onnx import load, numpy_helper, helper
import pathlib
import numpy as np
from .node_definitions import NODE_NAMES
import polymath as pm

def from_onnx(filepath):
    onnx_proto, graph_name = load_onnx_proto(filepath)
    attr = get_model_attributes(onnx_proto)

    for n in onnx_proto.graph.node:
        if n.op_type not in NODE_NAMES and n.name not in NODE_NAMES:
            raise RuntimeError(f"Support for {n.op_type} or {n.name} is not currently included in PolyMath")

    domain = "nn"
    for a in attr["opset_import"]:
        if a.domain == "ai.onnx.ml":
            domain = "ml"
            break

    if domain == "ml":
        graph = generate_ml_mdfg(onnx_proto.graph)
    else:
        graph = generate_nn_mdfg(onnx_proto.graph)

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

def generate_nn_mdfg(onnx_graph):
    names = [des.name for des in onnx_graph.DESCRIPTOR.fields]
    graph_name = getattr(onnx_graph, "name")
    initializers = {i.name: numpy_helper.to_array(i) for i in onnx_graph.initializer}
    mgdfg = pm.Node(name=graph_name)
    node_info = {}
    for i in onnx_graph.input:
        assert i.name not in node_info
        if i.name in initializers:
            node_info[i.name] = pm.variable(initializers[i.name], name=i.name, shape=get_value_info_shape(i), graph=mgdfg)
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
            node_info[v.name] = pm.placeholder(name=v.name, shape=get_value_info_shape(v), graph=mgdfg)

    return mgdfg

def _print_proto_fields(pb):
    print(f"{pb} fields : {[n.name for n in pb.DESCRIPTOR.fields]}")

def get_value_info_shape(vi):
    return [dim.dim_value for dim in vi.type.tensor_type.shape.dim if dim.dim_value > 0]

def get_attributes(node):
    attributes = {}
    for a in node.attribute:
        val = helper.get_attribute_value(a)
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        elif isinstance(val, list):
            val = np.asarray(val)
        attributes[a.name] = val

    return attributes


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

