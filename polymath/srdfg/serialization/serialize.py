import polymath.srdfg.serialization.srdfgv3_pb2 as pb
from polymath import Node, Domain, Template, output
import polymath as pm
from numproto import ndarray_to_proto, proto_to_ndarray
from typing import Iterable, Union, Mapping
from numbers import Integral
import numpy as np
CONTEXT_TEMPLATE_TYPES = (pm.state, pm.output, pm.temp)

def pb_store(node, file_path, outname=None):
    if outname:
        file_path = f"{file_path}/{outname}"
    else:
        file_path = f"{file_path}/{node.name}.srdfg"

    count_before = len(node.nodes.keys())
    with open(file_path, "wb") as program_file:
        program_file.write(_serialize_node(node).SerializeToString())
    count_after = len(node.nodes.keys())

def pb_load(file_path, verbose=False):
    new_program = pb.Node()
    with open(file_path, "rb") as program_file:
        new_program.ParseFromString(program_file.read())
    deserialization_info = {'write_resets': [],
                            'uuid_map': {}
                            }
    deser_node = _deserialize_node(new_program, deserialization_info, verbose=verbose)
    return deser_node


def _to_bytes_or_false(val: (Union[str, bytes])) -> Union[bytes, bool]:
    if isinstance(val, bytes):
        return val
    else:
        try:
            return val.encode('utf-8')
        except AttributeError:
            return False

def _serialize_domain(dom, pb_dom):
    for d in dom:
        new_dom = pb_dom.domains.add()
        if isinstance(d, Node):
            new_dom.type = pb.Attribute.Type.NODE
            new_dom.s = _to_bytes_or_false(d.name)
        elif isinstance(d, np.ndarray):
            new_dom.type = pb.Attribute.Type.NDARRAY
            new_dom.nda.CopyFrom(ndarray_to_proto(d))
        elif isinstance(d, Integral):
            new_dom.type = pb.Attribute.Type.INT32
            new_dom.i32 = d
        elif isinstance(d, float):
            new_dom.type = pb.Attribute.Type.DOUBLE
            new_dom.d = d
        elif isinstance(d, str):
            new_dom.type = pb.Attribute.Type.STRING
            new_dom.s = _to_bytes_or_false(d)
        elif isinstance(d, bool):
            new_dom.type = pb.Attribute.Type.BOOL
            new_dom.b = d
        elif isinstance(d, Iterable):
            if all(isinstance(a, Node) for a in d):
                new_dom.type = pb.Attribute.Type.NODES
                new_dom.ss.extend([_to_bytes_or_false(a.name) for a in d])
            elif all(isinstance(a, list) for a in d):
                new_dom.type = pb.Attribute.Type.NDARRAYS
                np_arr = [ndarray_to_proto(np.asarray(a)) for a in d]
                new_dom.ndas.extend(np_arr)
            elif all(isinstance(a, np.ndarray) for a in d):
                new_dom.type = pb.Attribute.Type.NDARRAYS
                new_dom.ndas.extend(ndarray_to_proto(a) for a in d)
            elif all(isinstance(a, Integral) for a in d):
                new_dom.type = pb.Attribute.Type.INT32S
                new_dom.i32s.extend(d)
            elif all(isinstance(a, float) for a in d):
                new_dom.type = pb.Attribute.Type.DOUBLES
                new_dom.ds.extend(d)
            elif all(map(lambda bytes_or_false: bytes_or_false is not False, [_to_bytes_or_false(a) for a in d])):
                new_dom.type = pb.Attribute.Type.STRINGS
                new_dom.ss.extend([_to_bytes_or_false(a) for a in d])
            elif all(isinstance(a, bool) for a in d):
                new_dom.type = pb.Attribute.Type.BOOLS
                new_dom.bs.extend(d)
            else:
                raise TypeError(f"Cannot find serializable method for argument {d} with "
                                f"type {type(d)} in domain {dom.names}")

        else:
            raise TypeError(f"Cannot find serializable method for domain {d} with type {type(d)}")

def _deserialize_domain(pb_dom, graph, node_name, info, write_graph=None):
    doms = []

    for d in pb_dom.dom.domains:
        if d.type == pb.Attribute.Type.NODE:
            if d.s.decode("utf-8") != node_name:
                d_name = d.s.decode("utf-8")
                if d_name in graph.nodes:
                    arg_node = graph.nodes[d_name]
                elif write_graph and d_name in write_graph.nodes:
                    arg_node = write_graph.nodes[d_name]
                else:
                    all_graphs = []
                    g = graph

                    while g is not None:
                        all_graphs.append(g.name)
                        g = g.graph
                    arg_node = None
                    for k, v in info['uuid_map'].items():
                        if d_name == v.name:
                            arg_node = v
                            break
                    if arg_node is None:
                        raise KeyError(f"Unable to find node in graph {d_name} for {node_name}: "
                                       f"{graph.name}. All Graphs: {all_graphs}\n"
                                       f"Write graph: {write_graph}")
                doms.append(arg_node)

        elif d.type == pb.Attribute.Type.NDARRAY:
            doms.append(proto_to_ndarray(d.nda))
        elif d.type == pb.Attribute.Type.INT32:
            doms.append(d.i32)
        elif d.type == pb.Attribute.Type.DOUBLE:
            doms.append(d.d)
        elif d.type == pb.Attribute.Type.STRING:
            doms.append(d.s.decode("utf-8"))
        elif d.type == pb.Attribute.Type.BOOL:
            doms.append(d.b)
        elif d.type == pb.Attribute.Type.NODES:
            arg_node = []
            for a in d.ss:
                if a.decode("utf-8") in graph.nodes:
                    anode = graph.nodes[a.decode("utf-8")]
                elif write_graph and a.decode("utf-8") in write_graph.nodes:
                    anode = write_graph.nodes[a.decode("utf-8")]
                else:
                    raise KeyError(f"Unable to find node in graph {a.decode('utf-8')}")
                arg_node.append(anode)
            doms.append(arg_node)
        elif d.type == pb.Attribute.Type.NDARRAYS:
            doms.append([proto_to_ndarray(a) for a in d.ndas])
        elif d.type == pb.Attribute.Type.INT32S:
            doms.append(list(d.i32s))
        elif d.type == pb.Attribute.Type.DOUBLES:
            doms.append(list(d.ds))
        elif d.type == pb.Attribute.Type.STRINGS:
            doms.append([a.decode("utf-8") for a in d.ss])
        elif d.type == pb.Attribute.Type.BOOLS:
            doms.append(list(d.b))
        else:
            raise TypeError(f"Cannot find deserializeable method for argument {d} with type {d.type}")
    return Domain(tuple(doms))


def _deserialize_node(pb_node, deserialization_info, graph=None, verbose=False):
    set_fields = pb_node.DESCRIPTOR.fields_by_name
    kwargs = {}
    kwargs["name"] = pb_node.name
    kwargs["op_name"] = pb_node.op_name
    kwargs["dependencies"] = [dep for dep in pb_node.dependencies]
    write_graph = None

    if kwargs["op_name"] == "write":
        wg = pb_node.kwargs["write_graph"]
        kwargs["write_graph"] = [a.decode("utf-8") for a in wg.ss]
        curr_g = graph
        while curr_g:
            if len(kwargs["write_graph"]) > 0 and kwargs["write_graph"][-1] in curr_g.nodes:
                write_graph = curr_g.nodes[kwargs["write_graph"][-1]]
                break
            curr_g = curr_g.graph

    args = []
    for name in pb_node.kwargs:
        arg = pb_node.kwargs[name]
        if arg.type == pb.Attribute.Type.DOM:
            kwargs[name] = _deserialize_domain(arg, graph, pb_node.name, deserialization_info, write_graph=write_graph)
        elif arg.type == pb.Attribute.Type.NODE:
            if arg.s.decode("utf-8") in graph.nodes:
                arg_node = graph.nodes[arg.s.decode("utf-8")]
            elif write_graph and arg.decode("utf-8") in write_graph.nodes:
                arg_node = write_graph.nodes[arg.s.decode("utf-8")]
            else:
                raise KeyError(f"Unable to find node in graph {arg.decode('utf-8')}")
            kwargs[name] = arg_node
        elif arg.type == pb.Attribute.Type.NDARRAY:
            kwargs[name] = proto_to_ndarray(arg.nda)
        elif arg.type == pb.Attribute.Type.INT32:
            kwargs[name] = arg.i32
        elif arg.type == pb.Attribute.Type.DOUBLE:
            kwargs[name] = arg.d
        elif arg.type == pb.Attribute.Type.STRING:
            kwargs[name] = arg.s.decode("utf-8")
        elif arg.type == pb.Attribute.Type.BOOL:
            kwargs[name] = arg.b
        elif arg.type == pb.Attribute.Type.NODES:
            arg_node = []
            for a in arg.ss:
                if a.decode("utf-8") in graph.nodes:
                    anode = graph.nodes[a.decode("utf-8")]
                elif write_graph and a.decode("utf-8") in write_graph.nodes:
                    anode = write_graph.nodes[a.decode("utf-8")]
                else:
                    raise KeyError(f"Unable to find node in graph {a.decode('utf-8')}")
                arg_node.append(anode)
            kwargs[name] = arg_node
        elif arg.type == pb.Attribute.Type.NDARRAYS:
            kwargs[name] = [proto_to_ndarray(a) for a in arg.ndas]
        elif arg.type == pb.Attribute.Type.INT32S:
            kwargs[name] = list(arg.i32s)
        elif arg.type == pb.Attribute.Type.DOUBLES:
            kwargs[name] = list(arg.ds)
        elif arg.type == pb.Attribute.Type.STRINGS:
            kwargs[name] = [a.decode("utf-8") for a in arg.ss]
        elif arg.type == pb.Attribute.Type.BOOLS:
            kwargs[name] = list(arg.b)
        else:
            raise TypeError(f"Cannot find deserializeable method for argument {name} with type {arg.type}")


    for i, arg in enumerate(pb_node.args):
        if arg.type == pb.Attribute.Type.NODE:
            arg_str = arg.s.decode("utf-8")
            if arg_str in graph.nodes:
                arg_node = graph.nodes[arg_str]
            elif write_graph and arg_str in write_graph.nodes:
                arg_node = write_graph.nodes[arg_str]
            else:
                if verbose:
                    err_str = f"Could not find {arg_str} in nodes for {graph.name} - {graph}\n" \
                              f"Node name: {pb_node.name} - {pb_node.op_name}:\n" \
                                   f"Keys: {list(graph.nodes.keys())}\n"
                else:
                    err_str = f"Could not find {arg_str} in nodes for {graph.name} - {graph}\n" \
                              f"Node name: {pb_node.name} - {pb_node.op_name}:\n"
                raise RuntimeError(err_str)
            args.append(arg_node)
        elif arg.type == pb.Attribute.Type.NDARRAY:
            args.append(proto_to_ndarray(arg.nda))
        elif arg.type == pb.Attribute.Type.INT32:
            args.append(arg.i32)
        elif arg.type == pb.Attribute.Type.DOUBLE:
            args.append(arg.d)
        elif arg.type == pb.Attribute.Type.STRING:
            args.append(arg.s.decode("utf-8"))
        elif arg.type == pb.Attribute.Type.BOOL:
            args.append(arg.b)
        elif arg.type == pb.Attribute.Type.MAP:
            mapping = {}
            for name in arg.mapping:
                mapped_arg = arg.mapping[name]
                mapping[name] = to_polymath_arg(mapped_arg, graph, write_graph, pb_node, verbose=verbose)
            args.append(mapping)
        elif arg.type == pb.Attribute.Type.NODES:
            arg_node = []
            for a in arg.ss:
                if a.decode("utf-8") in graph.nodes:
                    anode = graph.nodes[a.decode("utf-8")]
                elif write_graph and a.decode("utf-8") in write_graph.nodes:
                    anode = write_graph.nodes[a.decode("utf-8")]
                else:
                    raise KeyError(f"Unable to find node in graph {a.decode('utf-8')}")
                arg_node.append(anode)
            args.append(arg_node)
        elif arg.type == pb.Attribute.Type.NDARRAYS:
            args.append([proto_to_ndarray(a) for a in arg.ndas])
        elif arg.type == pb.Attribute.Type.INT32S:
            args.append(list(arg.i32s))
        elif arg.type == pb.Attribute.Type.DOUBLES:
            args.append(list(arg.ds))
        elif arg.type == pb.Attribute.Type.STRINGS:
            args.append([a.decode("utf-8") for a in arg.ss])
        elif arg.type == pb.Attribute.Type.BOOLS:
            args.append(list(arg.b))
        else:
            raise TypeError(f"Cannot find deserializeable method for argument {arg} with type {arg.type}")

    args = tuple(args)

    mod_name, cls_name = pb_node.module.rsplit(".", 1)
    mod = __import__(mod_name, fromlist=[cls_name])

    if "target" in kwargs:
        func_mod_name, func_name = kwargs["target"].rsplit(".", 1)
        func_mod = __import__(func_mod_name, fromlist=[func_name])
        target = getattr(func_mod, func_name)
        kwargs.pop("target")

        if cls_name in ["func_op", "slice_op", "index_op"]:
            node = getattr(mod, cls_name)(target, *args, graph=graph, **kwargs)
        elif cls_name in ["cast", "clip"]:
            init_keys = ["np_dtype", "minval", "maxval"]
            new_args = []
            for k in list(kwargs.keys()):
                if k in init_keys:
                    new_args.append(kwargs.pop(k))
            new_args += list(args)
            new_args = tuple(new_args)
            node = getattr(mod, cls_name)(*new_args, graph=graph, **kwargs)
        else:
            node = getattr(mod, cls_name)(*args, graph=graph, **kwargs)
    else:
        template_subclass_names = [c.__name__ for c in Template.__subclasses__()]

        if cls_name in template_subclass_names:
            kwargs.pop("op_name")
            kwargs['skip_definition'] = True
            for a in args:
                if isinstance(a, CONTEXT_TEMPLATE_TYPES) and a.name not in deserialization_info['write_resets']:
                    a.write_count = 0
        node = getattr(mod, cls_name)(*args, graph=graph, **kwargs)
    if pb_node.graph_id >= 0 and pb_node.graph_id not in deserialization_info['uuid_map']:
        print(f"Cannot find {pb_node.name} with graph {node.graph.name}")
    deserialization_info['uuid_map'][pb_node.uuid] = node

    for pb_n in pb_node.nodes:

        if pb_n.name in node.nodes:
            continue
        node.nodes[pb_n.name] = _deserialize_node(pb_n, deserialization_info, graph=node, verbose=verbose)

    shape_list = []
    for shape in pb_node.shape:
        val_type = shape.WhichOneof("value")
        if val_type == "shape_const":
            shape_list.append(shape.shape_const)
        else:
            if shape.shape_id not in graph.nodes:
                shape_list.append(node.nodes[shape.shape_id])
            else:
                shape_list.append(graph.nodes[shape.shape_id])
    node._shape = tuple(shape_list)


    return node

def from_polymath_arg(serialized_arg, arg):
    if isinstance(arg, Node):
        serialized_arg.type = pb.Attribute.Type.NODE
        serialized_arg.s = _to_bytes_or_false(arg.name)
    elif isinstance(arg, np.ndarray):
        serialized_arg.type = pb.Attribute.Type.NDARRAY
        serialized_arg.nda.CopyFrom(ndarray_to_proto(arg))
    elif isinstance(arg, Integral):
        serialized_arg.type = pb.Attribute.Type.INT32
        serialized_arg.i32 = arg
    elif isinstance(arg, float):
        serialized_arg.type = pb.Attribute.Type.DOUBLE
        serialized_arg.d = arg
    elif isinstance(arg, str):
        serialized_arg.type = pb.Attribute.Type.STRING
        serialized_arg.s = _to_bytes_or_false(arg)
    elif isinstance(arg, bool):
        serialized_arg.type = pb.Attribute.Type.BOOL
        serialized_arg.b = arg
    else:
        raise RuntimeError(f"Unable to find valid type for arg {arg}")

def to_polymath_arg(arg, graph, write_graph, pb_node, verbose):
    if arg.type == pb.Attribute.Type.NODE:
        arg_str = arg.s.decode("utf-8")
        if arg_str in graph.nodes:
            arg_node = graph.nodes[arg_str]
        elif write_graph and arg_str in write_graph.nodes:
            arg_node = write_graph.nodes[arg_str]
        else:
            if verbose:
                err_str = f"Could not find {arg_str} in nodes for {graph.name} - {graph}\n" \
                          f"Node name: {pb_node.name} - {pb_node.op_name}:\n" \
                          f"Keys: {list(graph.nodes.keys())}\n"
            else:
                err_str = f"Could not find {arg_str} in nodes for {graph.name} - {graph}\n" \
                          f"Node name: {pb_node.name} - {pb_node.op_name}:\n"
            raise RuntimeError(err_str)
        return arg_node
    elif arg.type == pb.Attribute.Type.NDARRAY:
        return proto_to_ndarray(arg.nda)
    elif arg.type == pb.Attribute.Type.INT32:
        return arg.i32
    elif arg.type == pb.Attribute.Type.DOUBLE:
        return arg.d
    elif arg.type == pb.Attribute.Type.STRING:
        return arg.s.decode("utf-8")
    elif arg.type == pb.Attribute.Type.BOOL:
        return arg.b
    else:
        raise RuntimeError

def _serialize_node(node_instance):
    graph_id = -1 if node_instance.graph is None else id(node_instance.graph)
    pb_node = pb.Node(name=node_instance.name, op_name=node_instance.op_name, uuid=id(node_instance),
                      module=f"{node_instance.__class__.__module__}.{node_instance.__class__.__name__}",
                      graph_id=graph_id)

    for shape in node_instance.shape:
        pb_shape = pb_node.shape.add()
        if isinstance(shape, Node):
            pb_shape.shape_id = shape.name
        elif not isinstance(shape, Integral):
            raise TypeError(f"Invalid type for shape {shape} - {type(shape)}")
        else:
            pb_shape.shape_const = shape
    pb_node.dependencies.extend(node_instance.dependencies)

    for arg in node_instance.args:
        new_arg = pb_node.args.add()
        if isinstance(arg, Node):
            new_arg.type = pb.Attribute.Type.NODE
            new_arg.s = _to_bytes_or_false(arg.name)
        elif isinstance(arg, np.ndarray):
            new_arg.type = pb.Attribute.Type.NDARRAY
            new_arg.nda.CopyFrom(ndarray_to_proto(arg))
        elif isinstance(arg, Integral):
            new_arg.type = pb.Attribute.Type.INT32
            new_arg.i32 = arg
        elif isinstance(arg, float):
            new_arg.type = pb.Attribute.Type.DOUBLE
            new_arg.d = arg
        elif isinstance(arg, str):
            new_arg.type = pb.Attribute.Type.STRING
            new_arg.s = _to_bytes_or_false(arg)
        elif isinstance(arg, bool):
            new_arg.type = pb.Attribute.Type.BOOL
            new_arg.b = arg
        elif isinstance(arg, Mapping):
            new_arg.type = pb.Attribute.Type.MAP
            for name, value in arg.items():
                mapped_arg = new_arg.mapping[name]
                from_polymath_arg(mapped_arg, value)
        elif isinstance(arg, Iterable):

            # TODO: Fix this to be more generic

            if isinstance(arg, list) and len(arg) == 1 and isinstance(arg[0], tuple):
                arg = arg[0]
            if all(isinstance(a, Node) for a in arg):
                new_arg.type = pb.Attribute.Type.NODES
                new_arg.ss.extend([_to_bytes_or_false(a.name) for a in arg])
            elif all(isinstance(a, list) for a in arg):
                new_arg.type = pb.Attribute.Type.NDARRAYS
                np_arr = [ndarray_to_proto(np.asarray(a)) for a in arg]
                new_arg.ndas.extend(np_arr)
            elif all(isinstance(a, np.ndarray) for a in arg):
                new_arg.type = pb.Attribute.Type.NDARRAYS
                new_arg.ndas.extend(ndarray_to_proto(a) for a in arg)
            elif all(isinstance(a, Integral) for a in arg):
                new_arg.type = pb.Attribute.Type.INT32S
                new_arg.i32s.extend(arg)
            elif all(isinstance(a, float) for a in arg):
                new_arg.type = pb.Attribute.Type.DOUBLES
                new_arg.ds.extend(arg)
            elif all(map(lambda bytes_or_false: bytes_or_false is not False, [_to_bytes_or_false(a) for a in arg])):
                new_arg.type = pb.Attribute.Type.STRINGS
                new_arg.ss.extend([_to_bytes_or_false(a) for a in arg])
            elif all(isinstance(a, bool) for a in arg):
                new_arg.type = pb.Attribute.Type.BOOLS
                new_arg.bs.extend(arg)
            else:
                raise TypeError(f"Cannot find serializable method for argument {arg} with "
                                f"type {type(arg)} in node {node_instance.name} - {node_instance.op_name}\n"
                                f"All args: {node_instance.args}")

        else:
            raise TypeError(f"Cannot find serializable method for argument {arg} with type {type(arg)} in node {node_instance.name} - {node_instance.op_name}\n")

    for name, arg in node_instance.kwargs.items():
        if arg is None:
            continue
        new_arg = pb_node.kwargs[name]
        if isinstance(arg, Domain):
            _serialize_domain(arg, new_arg.dom)
            new_arg.type = pb.Attribute.Type.DOM
        elif isinstance(arg, Node):
            new_arg.type = pb.Attribute.Type.NODE
            new_arg.s = _to_bytes_or_false(arg.name)
        elif isinstance(arg, np.ndarray):
            new_arg.type = pb.Attribute.Type.NDARRAY
            new_arg.nda.CopyFrom(ndarray_to_proto(arg))
        elif isinstance(arg, Integral):
            new_arg.type = pb.Attribute.Type.INT32
            new_arg.i32 = arg
        elif isinstance(arg, float):
            new_arg.type = pb.Attribute.Type.DOUBLE
            new_arg.d = arg
        elif isinstance(arg, str):
            new_arg.type = pb.Attribute.Type.STRING
            new_arg.s = _to_bytes_or_false(arg)
        elif isinstance(arg, bool):
            new_arg.type = pb.Attribute.Type.BOOL
            new_arg.b = _to_bytes_or_false(arg)
        elif isinstance(arg, Iterable):
            if all(isinstance(a, Node) for a in arg):
                new_arg.type = pb.Attribute.Type.NODES
                new_arg.ss.extend([_to_bytes_or_false(a.name) for a in arg])
            elif all(isinstance(a, np.ndarray) for a in arg):
                new_arg.type = pb.Attribute.Type.NDARRAYS
                new_arg.ndas.extend(ndarray_to_proto(a) for a in arg)
            elif all(isinstance(a, Integral) for a in arg):
                new_arg.type = pb.Attribute.Type.INT32S
                new_arg.i32s.extend(arg)
            elif all(isinstance(a, float) for a in arg):
                new_arg.type = pb.Attribute.Type.DOUBLES
                new_arg.ds.extend(arg)
            elif all(map(lambda bytes_or_false: bytes_or_false is not False, [_to_bytes_or_false(a) for a in arg])):
                new_arg.type = pb.Attribute.Type.STRINGS
                new_arg.ss.extend([_to_bytes_or_false(a) for a in arg])
            elif all(isinstance(a, bool) for a in arg):
                new_arg.type = pb.Attribute.Type.BOOLS
                new_arg.bs.extend(arg)
        else:
            raise TypeError(f"Cannot find serializable method for argument {name}={arg} with type {type(arg)} in {node_instance}")
    serialized = []

    for k, node in node_instance.nodes.items():
        if not isinstance(node, Node):
            raise RuntimeError(f"Non-node object included in graph for {node_instance.name} with name {k}.")
        elif node.name != node_instance.name:
            serialized.append(_serialize_node(node))
    pb_node.nodes.extend(serialized)

    return pb_node