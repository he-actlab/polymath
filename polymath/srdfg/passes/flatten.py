from polymath.srdfg.serialization import srdfg_utils
from polymath.srdfg.serialization.mgdfg_pb2 import Component, ValueInfo
import logging
from polymath.srdfg.passes.pass_utils import is_literal, is_number

def flatten_graph(output_graph: Component, graph, templates, context,edge_node_ids, arg_map):

    components = {}
    for e in graph.edge_info:
        copy_edge = ValueInfo()

        if is_literal(e):

            uid = str(e)
            copy_edge.CopyFrom(graph.edge_info[e])
            copy_edge.name = uid
        elif e in arg_map.keys():
            uid = context + e
            copy_edge.CopyFrom(arg_map[e])
            copy_edge.attributes['alias'].CopyFrom(srdfg_utils.make_attribute('alias', arg_map[e].name))
        else:
            uid = context + e
            copy_edge.CopyFrom(graph.edge_info[e])
            copy_edge.name = uid

        if e in graph.input and e not in output_graph.input:
            output_graph.input.extend([uid])
        elif e in graph.state and e not in output_graph.state:
            output_graph.state.extend([uid])
        elif e in graph.output and e not in output_graph.output:
            output_graph.output.extend([uid])
        elif e in graph.parameters and e not in output_graph.parameters:
            output_graph.parameters.extend([uid])

        if graph.name != 'main':
            ordered_args = srdfg_utils.get_attribute_value(graph.attributes['ordered_args'])
        else:
            ordered_args = []

        if 'dimensions' in list(copy_edge.attributes):
            dims = srdfg_utils.get_attribute_value(copy_edge.attributes['dimensions'])
            new_dims = []
            for d in dims:
                if d in arg_map.keys():
                    new_dims.append(arg_map[d].name)
                else:
                    new_dims.append(d)
            copy_edge.attributes['dimensions'].CopyFrom(srdfg_utils.make_attribute('dimensions', new_dims))

        if uid not in edge_node_ids['edges'].keys():
            edge_node_ids['edges'][uid] = str(len(edge_node_ids['edges'].keys()))
            output_graph.edge_info[uid].CopyFrom(copy_edge)
            if e not in arg_map.keys():
                output_graph.edge_info[uid].gid = int(edge_node_ids['edges'][uid])
                output_graph.edge_info[uid].attributes['component_type'].CopyFrom(
                    srdfg_utils.make_attribute('component_type', graph.op_type))


    for n in graph.sub_graph:
        op_cat = n.op_cat
        if op_cat == 'component':

            if n.op_type in components.keys():
                components[n.op_type] += 1
                new_context = context + n.op_type + str(components[n.op_type]) + '/'
            else:
                components[n.op_type] = 0
                new_context = context + n.op_type + str(components[n.op_type]) + '/'

            instance_args = srdfg_utils.get_attribute_value(n.attributes['ordered_args'])
            ctemplate = templates[n.op_type]
            signature_args = srdfg_utils.get_attribute_value(ctemplate.attributes['ordered_args'])
            carg_map = create_map(instance_args, signature_args,graph.edge_info, ctemplate.edge_info, templates[n.op_type])
            update_statement_graphs(ctemplate, output_graph, new_context)

            flatten_graph(output_graph, ctemplate, templates, new_context , edge_node_ids, carg_map)

        else:
            new = update_node(n, context, arg_map)

            if new.name not in edge_node_ids['nodes'].keys():
                edge_node_ids['nodes'][new.name] = str(len(edge_node_ids['nodes'].keys()))
            new.gid = int(edge_node_ids['nodes'][new.name])
            output_graph.sub_graph.extend([new])

def update_statement_graphs(template, output_graph, context):
    for s in template.statement_graphs:
        statement_nodes = s.statement_node
        new_graph = output_graph.statement_graphs.add()
        nodes = []
        for n in statement_nodes:
            nodes.append(context + n)
        new_graph.statement_node.extend(nodes)


def create_map(instance_args, signature_args, instance_edges, signature_edges, op=None):
    carg_map = {}

    for i in range(len(instance_args)):
        iarg = instance_args[i]
        sarg = signature_args[i]
        if is_number(iarg):
            iarg = str(iarg)
        carg_map[sarg] = instance_edges[iarg]

        carg_map[sarg].name = iarg


        idims = srdfg_utils.get_attribute_value(instance_edges[iarg].attributes['dimensions'])
        iid_literal = False
        if instance_edges[iarg].iid:
            inst_iid = instance_edges[iarg].iid
            iid_literal = is_literal(inst_iid)

        sdims = srdfg_utils.get_attribute_value(signature_edges[sarg].attributes['dimensions'])

        if len(idims) != len(sdims) and not iid_literal:
            logging.error("Error! Dimensions between edges connecting components do not match:{} versus {} for {} and {}".format(idims, sdims, iarg, sarg))
        elif not iid_literal:
            for d in range(len(idims)):

                inst_dim = idims[d]
                sig_dim = sdims[d]
                if is_number(inst_dim):
                    inst_dim = str(inst_dim)
                carg_map[sig_dim] = instance_edges[inst_dim]
                carg_map[sig_dim].name = inst_dim
                carg_map[sig_dim].attributes['vtype'].CopyFrom(srdfg_utils.make_attribute('vtype', 'scalar'))



    if len(signature_args) > len(instance_args):
        start = len(instance_args)
        for default in signature_args[start:]:
            sig_attr = list(signature_edges[default].attributes)

            if 'default' not in sig_attr:
                logging.error(
                    "Error! No default value for unspecified arg: {}".format(default))
            else:
                def_val = srdfg_utils.get_attribute_value(signature_edges[default].attributes['default'])
                carg_map[default] = signature_edges[default]
                carg_map[default].attributes['value'].CopyFrom(srdfg_utils.make_attribute('value', def_val))
                if is_number(def_val):
                    def_val = str(def_val)
                carg_map[default].name = def_val
                carg_map[default].attributes['vtype'].CopyFrom(srdfg_utils.make_attribute('vtype', 'scalar'))
    for e in op.edge_info:
        vcat = srdfg_utils.get_attribute_value(op.edge_info[e].attributes['vcat'])
        if vcat == 'declaration':
            dims = srdfg_utils.get_attribute_value(op.edge_info[e].attributes['dimensions'])
            sig_name = op.edge_info[e].name.rsplit("/", 1)[-1]


    return carg_map

def update_node(node, context, carg_map):
    new = Component(name=context  + node.name)
    inputs = []
    outputs = []
    states = []
    parameters = []
    for inp in node.input:
        if is_number(inp):
            i = str(inp)
        else:
            i = inp
        if is_literal(i):
            inputs.append(i)
        elif i in carg_map.keys():
            # inputs.append(carg_map[i])
            inputs.append(carg_map[i].name)
        else:
            inputs.append(context  + i)
    new.input.extend(inputs)

    for o in node.output:
        if is_number(o):
            out = str(o)
        else:
            out = o

        if is_literal(out):
            outputs.append(out)
        elif out in carg_map.keys():
            # outputs.append(carg_map[out])
            outputs.append(carg_map[out].name)
        else:
            outputs.append(context + out)

    new.output.extend(outputs)

    for st in node.state:
        if is_number(st):
            s = str(st)
        else:
            s = st

        if is_literal(s):
            states.append(s)
        elif s in carg_map.keys():
            # states.append(carg_map[s])
            states.append(carg_map[s].name)
        else:
            states.append(context + s)
    new.state.extend(states)


    for para in node.parameters:
        if is_number(para):
            p = str(para)
        else:
            p = para

        if is_literal(p):
            parameters.append(p)
        elif p in carg_map.keys():
            # parameters.append(carg_map[p])
            parameters.append(carg_map[p].name)
        else:
            parameters.append(context + p)

    new.parameters.extend(parameters)

    for attr in node.attributes:
        new.attributes[attr].CopyFrom(node.attributes[attr])
    new.op_type = node.op_type

    return new


