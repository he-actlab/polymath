from onnx import load, numpy_helper, helper, shape_inference
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import onnx
import pathlib
import numpy as np
from polymath.srdfg.templates.onnx_conversion import NODE_NAMES
import polymath as pm

# TODO: Dynamically create this list to make sure it is valid
PYTORCH_OP_NAMES = ['max_pool', 'lrn', 'conv', 'conv_bias', 'global_avg_pool', 'dropout', 'elem_tanh',
                'softmax', 'elem_cast', 'elem_sigmoid', 'batch_norm', 'batch_flatten', 'avg_pool2d',
                'leaky_relu', 'relu', 'dense_sigmoid', 'dense', 'avg_pool', 'gemm', 'gemm_no_bias', 'elem_add', 'elem_sub',
                 'elem_mul', 'dropout', 'coarse_flatten', 'cross_entropy_loss', 'reduce_sum']

def update_pytorch_graph_names(graph):
    names = {}
    for n in onnx_graph.node:
        new_inputs = []
        for i in n.input:
            if i in names:
                new_inputs.append(f"{names[i]}")
            elif i.isdigit():
                names[i] = f"{n.name}_{i}"
                new_inputs.append(names[i])
            else:
                names[i] = i
                new_inputs.append(i)
        n.input = new_inputs

        new_outputs = []
        for o in n.output:
            if o in names:
                new_outputs.append(f"{names[o]}")
            elif o.isdigit():
                names[o] = f"{n.name}_{o}"
                new_outputs.append(names[o])
            else:
                names[o] = o
                new_outputs.append(o)
        n.output = new_outputs

def from_pytorch(filepath, infer_shapes=True, use_filename=True, lower=False):
    onnx_proto, graph_name = load_onnx_proto(filepath)
    attr = get_model_attributes(onnx_proto)
    if infer_shapes:
        onnx_graph = shape_inference.infer_shapes(onnx_proto).graph
    else:
        onnx_graph = onnx_proto.graph
    for n in onnx_graph.node:
        if n.op_type not in NODE_NAMES and n.name not in NODE_NAMES:
            raise RuntimeError(f"Support for {n.op_type} or {n.name} is not currently included in PolyMath")

    graph = generate_srdfg(onnx_graph)
    if use_filename:
        graph_name = filepath.split("/")[-1].split(".")[0]
        graph.set_name(graph_name)

    if lower:
        lower_pass = pm.Lower(ONNX_OP_NAMES)
        graph = lower_pass(graph)

    return graph

def load_onnx_proto(filepath):
    graph_name = pathlib.Path(filepath).stem
    return load(filepath), graph_name

def get_model_attributes(model):
    kwargs = {des.name: getattr(model, des.name) for des in model.DESCRIPTOR.fields if des.name != "graph"}
    return kwargs

def get_initializers(initializers):
    init_dict = {}
    for i in initializers:
        val = numpy_helper.to_array(i)
        if len(val.shape) == 0:
            val = np.int(val)
        init_dict[i.name] = val
    return init_dict

def get_states_by_gradient(onnx_graph):
    state_vars = {}
    input_names = [i.name for i in onnx_graph.input]
    output_names = [o.name for o in onnx_graph.output]
    for n in onnx_graph.node:
        if n.output[0] in output_names and n.op_type == "Sub":
            for i in n.input:
                if i in input_names:
                    state_vars[n.output[0]] = i
                    break

    return state_vars

def generate_srdfg(onnx_graph):
    names = [des.name for des in onnx_graph.DESCRIPTOR.fields]
    graph_name = getattr(onnx_graph, "name")
    initializers = get_initializers(onnx_graph.initializer)
    mgdfg = pm.Node(name=graph_name)
    # TODO: This is a hotfix for identifying gradient updates, but weights should have initializers
    state_variables = get_states_by_gradient(onnx_graph)
    node_info = {}
    # TODO: If a value has an initializer, set the initializer value as the value for the node
    for o in onnx_graph.output:

        assert o.name not in node_info

        if o.name in state_variables:
            node_info[o.name] = pm.state(name=state_variables[o.name], shape=get_value_info_shape(o, mgdfg), graph=mgdfg)
            node_info[state_variables[o.name]] = node_info[o.name]
        else:
            node_info[o.name] = pm.output(name=o.name, shape=get_value_info_shape(o, mgdfg), graph=mgdfg)


    for i in onnx_graph.input:
        if i.name in state_variables.values():
            assert i.name in node_info
            continue
        assert i.name not in node_info
        if i.name in state_variables:
            node_info[i.name] = pm.state(name=state_variables[i.name], shape=get_value_info_shape(i, mgdfg), graph=mgdfg)
            node_info[state_variables[i.name]] = node_info[i.name]
        elif i.name in initializers and not itercheck(initializers[i.name]):
            node_info[i.name] = pm.parameter(name=i.name, default=initializers[i.name], graph=mgdfg)
        elif i.name in initializers:
            node_info[i.name] = pm.state(name=i.name, shape=get_value_info_shape(i, mgdfg), graph=mgdfg)
        else:
            node_info[i.name] = pm.input(name=i.name, shape=get_value_info_shape(i, mgdfg), graph=mgdfg)

    for v in onnx_graph.value_info:
        if v.name in node_info:
            continue
        elif v.name in initializers:
            node_info[v.name] = pm.variable(initializers[v.name], name=v.name, shape=get_value_info_shape(v, mgdfg), graph=mgdfg)
        else:

            node_info[v.name] = {"name": v.name, "shape": get_value_info_shape(v, mgdfg)}

    for k, v in initializers.items():
        if k not in node_info:
            # TODO: Need to set the value here
            node_info[k] = pm.state(name=k, shape=get_value_info_shape(v, mgdfg), graph=mgdfg)
            state_variables[k] = node_info[k]

    for k, v in mgdfg.nodes.items():
        if isinstance(v, pm.parameter) and k not in node_info:
            node_info[k] = v

    for n in onnx_graph.node:
        assert n.op_type in NODE_NAMES
        _ = convert_node(n, mgdfg, node_info, state_variables)

    return mgdfg

def convert_node(onnx_node, mgdfg, node_info, state_vars):
    name = onnx_node.name
    args = []
    # TODO: check if node name is already in the graph

    for i in onnx_node.input:
        if i not in mgdfg.nodes:
            raise KeyError(f"Input node {i} for {name} not in graph nodes:\n"
                           f"Nodes: {list(mgdfg.nodes.keys())}")

        args.append(mgdfg.nodes[i])
    num_outputs = 0
    outnode = None
    for o in onnx_node.output:
        if o in node_info:
            num_outputs += 1
            outnode = o
    if num_outputs != 1:
        raise RuntimeError(f"Length of outputs for node {onnx_node.name} is not equal to 1:\n"
                           f"Output: {onnx_node.output}")
    # elif onnx_node.output[0] not in node_info:
    #     raise RuntimeError(f"Could not find output for {onnx_node.name} in node info:\n"
    #                        f"Output: {onnx_node.output}")
    assert outnode is not None
    o_name = state_vars[outnode] if outnode in state_vars else outnode
    # o_name = state_vars[onnx_node.output[0]] if onnx_node.output[0] in state_vars else onnx_node.output[0]
    if isinstance(node_info[o_name], dict):


        o_shape = node_info[o_name]["shape"]

        attributes = get_attributes(onnx_node)
        args = tuple(args)
        kwargs = attributes
        kwargs['shape'] = tuple(list(o_shape))

        print(onnx_node.op_type)
        print(onnx_node.name)
        print()
        with mgdfg:
            new_node = NODE_NAMES[onnx_node.op_type](*args, name=o_name, **kwargs)

        if id(new_node.graph) != id(mgdfg):
            new_node.graph = mgdfg
            new_node.set_name(o_name)

        if o_name not in mgdfg.nodes:
            raise KeyError(f"Newly created node {new_node} with graph {new_node.graph} not added to the graph:\n"
                           f"\t{list(mgdfg.nodes.keys())}")

        if not new_node.is_shape_finalized():
            new_node._shape = o_shape
    else:

        o_shape = node_info[o_name].shape
        attributes = get_attributes(onnx_node)
        args = tuple(args)
        kwargs = attributes
        kwargs['shape'] = tuple(list(o_shape))
        kwargs['out'] = node_info[o_name]
        with mgdfg:
            new_node = NODE_NAMES[onnx_node.op_type](*args, **kwargs)

    return mgdfg

def _print_proto_fields(pb):
    print(f"{pb} fields : {[n.name for n in pb.DESCRIPTOR.fields]}")

def get_value_info_shape(vi, mgdfg):
    if isinstance(vi, np.ndarray):
        ret = vi.shape
    else:
        ret = []
        for i, dim in enumerate(vi.type.tensor_type.shape.dim):
            if hasattr(dim, 'dim_param') and dim.dim_param:
                if dim.dim_param in mgdfg.nodes:
                    shape_node = mgdfg.nodes[dim.dim_param]
                else:
                    shape_node = pm.parameter(name=dim.dim_param, graph=mgdfg)
                d_val = shape_node

            elif not dim.dim_value:
                shape_node = pm.parameter(name=f"{vi.name}_dim_{i}", graph=mgdfg)
                d_val = shape_node
            elif dim.dim_value > 0:
                d_val = dim.dim_value
            else:
                continue


            ret.append(d_val)
        ret = tuple(ret)
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
        if a.name in ["from","to"]:
            val = TENSOR_TYPE_TO_NP_TYPE[val]
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

def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)


