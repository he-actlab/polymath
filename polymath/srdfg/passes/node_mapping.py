from . import is_literal, is_number
from polymath.srdfg.serialization.mgdfg_pb2 import Component


def map_nodes(graph, templates, mapped_components, config_map):

    for n in graph.sub_graph:
        op_cat = n.op_cat
        if op_cat == 'component':
            if n.op_type in config_map['ops'].keys():
                n.op_cat = 'mapped_node'
                mapped_components.append(n.op_type)
            else:
                map_nodes(templates[n.op_type], templates, mapped_components, config_map)
        elif n.op_type in mapped_components:
            n.op_cat = 'mapped_node'


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