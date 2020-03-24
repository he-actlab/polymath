import logging
logger = logging.getLogger(__name__)


def check(expression):

    open_tup = tuple('(')
    close_tup = tuple(')')
    map = dict(zip(open_tup, close_tup))
    queue = []

    for i in expression:
        if i in open_tup:
            queue.append(map[i])
        elif i in close_tup:
            if not queue or i != queue.pop():
                return False
    return True

def get_text(context):
    if context.getText()[0] == '(' and context.getText()[-1] == ')' and check(context.getText()[1:-1]):

        return context.getText()[1:-1]
    else:
        return context.getText()

def infer_type(types, inputs, output):

    valid = [('int', 'float'), ('float', 'int')]
    type = types[0]
    for t in range(len(types)):
        if type != types[t]:
            if (types[t],type) not in valid:
                logging.warning("Error! Type mismatch for {} in {}. {} and {}".format(inputs, output, types[t], type))
            elif types[t] == 'float':
                type = types[t]

    return type

def infer_dtype(curr_type: str, new_type: str):
    if curr_type not in TYPES_PRECEDENCE.keys():
        raise TypeError(f"Invalid type for inference: {curr_type}")

    if new_type not in TYPES_PRECEDENCE.keys():
        raise TypeError(f"Invalid type for inference: {new_type}")

    if TYPES_PRECEDENCE[curr_type] < TYPES_PRECEDENCE[new_type]:
        return curr_type
    else:
        return new_type

def infer_dims(dims, inputs, output):
    return dims[0]

def infer_vtype(vtypes, inputs, output):
    vtype_vals = {'scalar': 0,
              'var' : 1,
              'index' : 2}
    vtype = 'scalar'
    for v in vtypes:
        if vtype_vals[v] > vtype_vals[vtype]:
            vtype = v
    return vtype

def check_edge_nodes(node_list, nodes, edges,cname):
    edge_cats = ['declaration', 'argument', 'literal', 'assign', 'assign_declaration']
    node_cats = ['argument', 'component', ]
    for id, edge in edges.items():
        if id not in node_list and edges[id]['vcat'] not in edge_cats:
            logging.warning("{} -- ID: {} not in node list".format(cname, id))
        if len(edge['src']) == 0 and edge['vcat'] not in ['argument','literal']:
            logging.warning("{} -- ID: {} does not have source node".format(cname, id))
        elif len(edge['src']) > 1:
            logging.warning("{} -- ID: {} has multiple source node"
                  ": {} -- {}".format(cname, id, edge['src'], edge['vcat']))

    for node in node_list:
        for i in nodes[node]['inputs']:
            if len(edges[i]['src']) == 0:
                logging.warning("{} -- ID: {} has no sources".format(cname, i))
        if node not in edges.keys() and nodes[node]['op_cat'] not in node_cats:
            logging.warning("{} -- ID: {} not in edge list".format(cname, node))


GROUP_FUNCTIONS = ['argmax', 'argmin', 'min', 'max', 'sum', 'prod']
FUNCTIONS = ['pi', 'e', 'log2', 'log', 'log10', 'floor', 'ceiling', 'sin', 'cos',
             'ln', 'fread', 'fwrite', 'sigmoid']

FLOAT_FUNCS = ['pi', 'e', 'log2', 'log', 'log10', 'sin', 'cos', 'sigmoid', 'ln']
INT_FUNCS = ['floor', 'ceiling']

BINARY_OPS = ['+', '-', '*', '/', '^', '<', '>', '==', '<=', '>=', '!=', '%']

DATATYPE_SPECIFIERS = ['int', 'float', 'str', 'bool', 'complex']

STRING_FUNCTION_TO_STRING_TYPE = {"pi": 'float',
                   "log": 'float',
                   "log2": 'float',
                   "float": 'float',
                   "int": 'int',
                   "bool": 'bool',
                   "complex": 'complex',
                   "bin": 'int',
                   "random": 'float',
                   "ceiling": 'float',
                   "floor": 'float',
                   "e": 'float',
                   "fread": 'str',
                   "fwrite": 'str',
                  "sigmoid" : "float"}

STRING_TEXT_TO_BINEXP = {"*": "mul",
                         "/": "div",
                         "+": "add",
                         "-": "sub",
                         "<": "tlt",
                         ">": "tgt",
                         "<=": "tlte",
                         ">=": "tgte",
                         "==": "teq",
                         "!=": "tne",
                          "^": "exp",
                         "%": "mod"
                         }

STRING_TEXT_TO_UNEXP = {"+" : "mov",
                        "-": "neg"}

STRING_TEXT_TO_FUNCTION = {"pi": "mov",
             "log": "log",
             "log2": "log2",
             "float": "cast",
             "int": "cast",
             "bin": "cast",
             "ceiling": "ceil",
             "floor": "floor",
             "e": "mov",
             "fread": "fread",
             "fwrite": "fwrite",
               "sigmoid" : "sigmoid"}
INSTRUCTION_OPS = ["index", "pred_store"]
NODE_OPS = ["src", "sink"]
NON_COMPONENT_NAMES = BINARY_OPS + DATATYPE_SPECIFIERS + GROUP_FUNCTIONS + FUNCTIONS + INSTRUCTION_OPS + NODE_OPS

TYPES_PRECEDENCE = {"bool": 0, "float": 1, "int" : 2, "str": 3}