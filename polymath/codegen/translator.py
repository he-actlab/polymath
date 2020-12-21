from polymath.srdfg.passes.flatten import flatten_graph, is_literal, is_number
from polymath.srdfg.passes.node_mapping import map_nodes
from polymath.codegen.codegen_utils import CMLANG_CAST_MAP, get_func

from polymath.srdfg.serialization.mgdfg_pb2 import Component
from polymath.srdfg.visualize import *
import json
import logging
from typing import Callable, List



class Translator(object):

    def __init__(self, onnx_proto: str, config_path: str, passes: List[str], run_async=False):
        self.input_proto = onnx_proto
        self.output_dir, self.output_file = os.path.split(self.input_proto)
        self.proto_name = self.output_file.split('.')[0]
        self.program = load_store.load_program(self.input_proto)
        self.graph = self.program.graph
        self.templates = self.program.templates

        self.config_path = config_path
        self.passes = passes
        self.load_config()
        self.perform_passes()
        self.graph_variables = {}
        if 'initialize_graph' in self.translator_config.keys():
            self.target_graph = self.initialize_graph()
        else:
            self.target_graph = None

        self.create_translated_graph()
        self.execute_graph()


    def load_config(self):
        with open(self.config_path, 'r') as config_file:
            config_data = config_file.read()

        self.translator_config = json.loads(config_data)

    def execute_graph(self):
        get_func(self.translator_config['graph_execution']['func_name'])(self.target_graph)

    def perform_passes(self):
        assert self.passes

        self.translated_graph = Component(name="translated_" + str(self.proto_name))
        edge_node_ids = {'edges': {},
                         'nodes': {}}
        self.translated_graph.statement_graphs.extend([])
        for p in self.passes:
            if p == 'flatten':
                flatten_graph(self.translated_graph, self.graph, self.templates, '', edge_node_ids, {})
                # flattened_graph_attr = hdfgutils.make_attribute('flattened', self.translated_graph)
                # self.program.attributes['flattened_graph'].CopyFrom(flattened_graph_attr)
            elif p == 'map_nodes':
                map_nodes(self.graph, self.templates, [], self.translator_config)
            else:
                logging.error(f"Invalid pass {p}. Exiting.")
                exit(1)

    def inititalize_input(self):
        for i in list(self.translated_graph.input):
            self.graph_variables[i] = self.init_var(self.translator_config['input_init'], i)

    def initialize_graph(self) -> Component:
        init_info = self.translator_config['initialize_graph']
        args = []
        for i in init_info['func_args']:
            args.append(self.get_arg_attribute(i,self.proto_name))

        kwargs = {}
        for k in init_info['func_kwargs'].keys():
            kwargs[k] = self.get_arg_attribute(init_info['func_kwargs'][k], self.proto_name)

        result = get_func(init_info['func_name'])(*args, **kwargs)
        return result

    def create_translated_graph(self):
        output_id = None
        assert len(self.translated_graph.input) == 1
        self.scope = ''
        self.inititalize_input()

        for n in self.translated_graph.sub_graph:
            op_cat = n.op_cat
            if op_cat == 'mapped_node':
                op_context = str(n.name).rsplit("/", 1)
                self.op_name = n.op_type
                if len(op_context) > 1:
                    self.scope = str(n.name).split("/")[-2]
                else:
                    self.scope = ''

                if len(op_context) > 1 and op_context[0] != 'main':
                    scope = op_context[0] + '/'
                else:
                    scope = ''
                op_config = self.translator_config['ops'][n.op_type]
                op_func = get_func(op_config['op_name'])
                args, kwargs, output_id = self.create_op_args(n.op_type, n, self.templates[n.op_type], scope)

                if len(output_id) == 1:
                    self.graph_variables[output_id[0]] = op_func(*args, **kwargs)

                    if output_id[0] in list(self.translated_graph.edge_info):
                        iedge = self.translated_graph.edge_info[output_id[0]]
                        if iedge.name != output_id[0]:
                            self.graph_variables[str(iedge.name)] = self.graph_variables[output_id[0]]
                else:
                    temp = op_func(*args, **kwargs)
                    if not hasattr(temp, '__len__'):
                        logging.error(f"Size mismatch between output of {n.op_type} which has length 1 output"
                                      f"Supplied config outputs:  {output_id}")
                        exit(1)
                    elif len(temp) != len(output_id):
                        logging.error(f"Size mismatch between output of {n.op_type} which has length {len(temp)} output"
                                      f"Supplied config outputs:  {output_id}")
                        exit(1)

                    for i in range(len(temp)):
                        self.graph_variables[output_id[i]] = temp[i]
                        if output_id[i] in list(self.translated_graph.edge_info):
                            iedge = self.translated_graph.edge_info[output_id[i]]
                            if iedge.name != output_id[i]:
                                self.graph_variables[str(iedge.name)] = self.graph_variables[output_id[i]]

        if not output_id:
            logging.error(f"No nodes mapped for graph_name")
            exit(1)
        elif len(output_id) != 1:
            logging.error(f"More than one output supplied for graph_name: {output_id}")
            exit(1)

        self.output_id = output_id[0]

        if 'output_wrapper' in self.translator_config.keys():
            self.target_graph = self.finalize_graph()


    def finalize_graph(self) -> Callable:
        output_info = self.translator_config['output_wrapper']
        args = []
        for i in output_info['func_args']:
            args.append(self.get_arg_attribute(i,self.output_id))

        result = get_func(output_info['func_name'])(*args)

        return result

    def create_op_args(self, op_name: str, node: Component, node_signature: Component, scope: str):
        op_config = self.translator_config['ops'][op_name]
        instance_args = mgdfg_utils.get_attribute_value(node.attributes['ordered_args'])
        signature_args = mgdfg_utils.get_attribute_value(node_signature.attributes['ordered_args'])
        default_map = self.create_default_map(self.templates[op_name])

        for i in range(len(instance_args)):
            instance_args[i] = scope + instance_args[i]


        args = self.get_ordered_args(op_config, signature_args, instance_args, default_map, op_name, scope)
        kwargs = self.get_kwargs(op_config, signature_args, instance_args,default_map, op_name, scope)
        output_keys = self.get_output_keys(op_config, signature_args, instance_args, op_name, scope)
        return args, kwargs, output_keys

    def get_ordered_args(self, op_config: dict, signature_args: List[str], instance_args: List[str], default_map,  op: str, scope):
        args = []
        for a in op_config['positional_arguments']:

            if a not in op_config['arg_map'].keys():
                if a == 'scope':
                    args.append(self.scope)
                    continue
                elif a == 'graph_name':
                    args.append(self.target_graph)
                    continue
                elif a == 'key':
                    args.append(op)
                else:
                    logging.error(f"{a} not found in argument map for op {op}. Please check config")
                    exit(1)
            arg = op_config['arg_map'][a]['key']
            if arg not in signature_args:
                logging.error(f"Argument {arg} not found in signature list {signature_args} for op {op}")
                exit(1)
            idx = signature_args.index(arg)

            if idx >= len(instance_args):
                if default_map[signature_args[idx]] is None:
                    logging.error(f"Error! No default argument for unspecified parameter {arg} in {op}, name: {signature_args[idx]}")
                    exit(1)
                if op_config['arg_map'][a]['init_func']:
                    var = self.init_var(op_config['arg_map'][a], default_map[signature_args[idx]], literal=True)
                elif op_config['arg_map'][a]['type'] in CMLANG_CAST_MAP.keys():
                    var = default_map[signature_args[idx]]
                else:
                    logging.error(f"Unable to resolve argument {default_map[signature_args[idx]]} for keyword {a}={signature_args[arg]}")
                    var = None
                    exit(1)

            else:
                instance_arg = instance_args[idx]
                if instance_arg in list(self.translated_graph.edge_info):
                    edge = self.translated_graph.edge_info[instance_arg]
                    ename = edge.name
                else:
                    ename = instance_arg

                if ename in self.graph_variables.keys() and instance_arg not in self.graph_variables.keys():
                    var = self.graph_variables[ename]
                elif instance_arg not in self.graph_variables.keys():
                    if op_config['arg_map'][a]['init_func']:
                        var = self.init_var(op_config['arg_map'][a], instance_arg)
                        if op_config['arg_map'][a]['arg_type'] != 'parameter':
                            self.graph_variables[instance_arg] = var

                    elif op_config['arg_map'][a]['type'] in CMLANG_CAST_MAP.keys():
                        var = CMLANG_CAST_MAP[op_config['arg_map'][a]['type']](instance_arg)
                    else:
                        logging.error(f"Unable to resolve argument {instance_arg} for keyword {a}={signature_args[arg]}")
                        var = None
                        exit(1)
                else:
                    var = self.graph_variables[instance_arg]

            args.append(var)
        return args

    def get_kwargs(self, op_config: dict, signature_args: List[str], instance_args: List[str], default_map, op, scope):
        kwargs = {}
        for k in op_config['keyword_arguments'].keys():
            if op_config['keyword_arguments'][k] not in op_config['arg_map'].keys():
                logging.error(f"Key id {k} with value {op_config['keyword_arguments'][k]} not found in argument map for op {op}."
                              f" Please check config")
                exit(1)
            id = op_config['keyword_arguments'][k]
            arg = op_config['arg_map'][id]['key']
            if arg not in signature_args:
                logging.error(f"Argument {arg} not found in signature list {signature_args} for op {op}")
                exit(1)
            idx = signature_args.index(arg)
            if idx >= len(instance_args):
                if default_map[signature_args[idx]] is None:
                    logging.error(f"Error! No default argument for unspecified parameter {arg} in {op}, name: {signature_args[idx]}")
                    exit(1)
                if op_config['arg_map'][id]['init_func']:
                    var = self.init_var(op_config['arg_map'][id], default_map[signature_args[idx]], literal=True)
                elif op_config['arg_map'][id]['type'] in CMLANG_CAST_MAP.keys():
                    var = default_map[signature_args[idx]]
                else:
                    logging.error(f"Unable to resolve argument {default_map[signature_args[idx]]} for keyword {id}={signature_args[arg]}")
                    var = None
                    exit(1)

            else:
                instance_arg = instance_args[idx]
                if instance_arg in list(self.translated_graph.edge_info):
                    edge = self.translated_graph.edge_info[instance_arg]
                    ename = edge.name
                else:
                    ename = instance_arg

                if ename in self.graph_variables.keys() and instance_arg not in self.graph_variables.keys():
                    var = self.graph_variables[ename]
                elif instance_arg not in self.graph_variables.keys():
                    if op_config['arg_map'][id]['init_func']:
                        var = self.init_var(op_config['arg_map'][id], instance_arg)
                        if op_config['arg_map'][id]['arg_type'] != 'parameter':
                            self.graph_variables[instance_arg] = var
                    elif op_config['arg_map'][id]['type'] in CMLANG_CAST_MAP.keys():
                        var = CMLANG_CAST_MAP[op_config['arg_map'][id]['type']](instance_arg)
                    else:
                        logging.error(f"Unable to resolve argument {instance_arg} for keyword {id}={signature_args[arg]}")
                        exit(1)
                else:
                    var = self.graph_variables[instance_arg]

            kwargs[k] = var
        return kwargs

    def get_output_keys(self, op_config, signature_args, instance_args, op, scope):
        output_keys = []
        for o in op_config['op_output']:
            if o not in op_config['arg_map'].keys():
                logging.error(f"Key id {o} with value {op_config['keyword_arguments'][o]} not found in argument map for op {op}."
                              f" Please check config")
                exit(1)
            arg = op_config['arg_map'][o]['key']
            if arg not in signature_args:
                logging.error(f"Argument {arg} not found in signature list {signature_args} for op {op}")
                exit(1)
            idx = signature_args.index(arg)
            if idx >= len(instance_args):
                logging.error(f"Error! Cannot assign output {o} to unspecified parameter {signature_args[idx]}")
                exit(1)
            output_keys.append(instance_args[idx])

        return output_keys

    def create_default_map(self, template):
        default_map = {}
        ordered_args = mgdfg_utils.get_attribute_value(template.attributes['ordered_args'])

        for a in ordered_args:
            if a not in list(template.edge_info):
                logging.error(f"Argument {a} not found in edges for {template.op_type}")

            edge = template.edge_info[a]
            if 'default' in list(edge.attributes):
                dtype = mgdfg_utils.get_attribute_value(edge.attributes['type'])
                default_map[a] = CMLANG_CAST_MAP[dtype](mgdfg_utils.get_attribute_value(edge.attributes['default']))
            else:
                default_map[a] = None
        return default_map

    def init_var(self, var, instance_name, literal=False):
        args = []
        kwargs = {}
        arg_type = var['arg_type']
        if isinstance(instance_name, str):

            id = instance_name.rsplit('/', 1)
            if len(id) > 1:
                scope = id[0]
                id = id[-1]
            else:
                scope = 'main'
                id = id[0]
        else:
            id = str(instance_name).rsplit('/', 1)
            if len(id) > 1:
                scope = id[0]
                id = id[-1]
            else:
                scope = 'main'
                id = id[0]
        if arg_type == 'parameter' and not literal and not is_literal(id):
            if instance_name not in list(self.translated_graph.edge_info):
                logging.error(f"Unable to get value for parameter {instance_name}")
                exit(1)
            edge = self.translated_graph.edge_info[instance_name]
            if 'value' not in list(edge.attributes):
                logging.error(f"Could not find literal for parameter argument {instance_name}.\n"
                              f"Possible attributes: {list(edge.attributes)}")
                exit(1)
            value = mgdfg_utils.get_attribute_value(edge.attributes['value'])
        elif is_literal(id) and isinstance(instance_name, str):
            if id in list(self.translated_graph.edge_info):
                edge = self.translated_graph.edge_info[id]
                value = mgdfg_utils.get_attribute_value(edge.attributes['value'])
            elif instance_name in list(self.translated_graph.edge_info):
                edge = self.translated_graph.edge_info[instance_name]
                value = mgdfg_utils.get_attribute_value(edge.attributes['value'])
            else:
                logging.error(f"Could not find literal for parameter argument {instance_name} with id {id}.\n"
                              f"var: {var['key']}")
                exit(1)

        else:
            value = instance_name

        for a in var['init_func_args']:
            arg_result = self.get_arg_attribute(a, value, literal=literal)
            args.append(arg_result)

        for k in var['init_func_kw'].keys():
            kwargs[k] = self.get_arg_attribute(var['init_func_kw'][k], value, literal=literal)
        if len(kwargs.keys()) == 0:

            var = get_func(var['init_func'])(*args)
        else:
            var = get_func(var['init_func'])(*args, **kwargs)
        return var

    def get_arg_attribute(self, key, instance_name, literal=False):
        if isinstance(key, list):
            arg = []
            for k in key:
                arg.append(self.get_arg_attribute(k, instance_name, literal=literal))
            return arg
        elif isinstance(key, dict):
            assert 'func_name' in key.keys() and 'func_args' in key.keys()
            return get_func(key['func_name'])(self.get_arg_attribute(key['func_args'], instance_name))
        elif isinstance(key, str):
            if key == 'name':
                return instance_name
            elif key == 'input_op':
                return self.op_name
            elif key == 'shape':
                if literal:
                    logging.error(f"Cannot get shape for literal value {instance_name} as attribute")
                    exit(1)
                edge = self.translated_graph.edge_info[instance_name]
                if 'dimensions' not in list(edge.attributes):
                    logging.error(f"No dimensions for edge {instance_name}")
                    tuple_dims = ()
                else:
                    dimensions = mgdfg_utils.get_attribute_value(edge.attributes['dimensions'])
                    tuple_dims = tuple(int(d) if is_number(d) else d for d in dimensions)

                return tuple_dims
            elif key == 'type':
                if literal:
                    return type(instance_name).__name__

                edge = self.translated_graph.edge_info[instance_name]
                if 'type' not in list(edge.attributes):
                    logging.error(f"No type for edge {instance_name}")
                    dtype = 'float32'
                else:
                    dtype = mgdfg_utils.get_attribute_value(edge.attributes['type'])
                return dtype
            elif key == "graph_name":
                return self.target_graph
            elif key == "scope":
                return self.scope
            elif key == "relative_scope":
                if len(self.scope.split("/")) > 1:
                    return self.scope.split("/")[-2]
                else:
                    return ''
            elif instance_name in self.graph_variables.keys():
                return self.graph_variables[instance_name]
            else:
                return key
        elif isinstance(key, bool) or isinstance(key, int):
            return key
        else:
            logging.error(f"Could not evaluate argument {key} for {instance_name}")

    def get_args(self, names, vars):
        args = []
        for n in names:
            if n not in vars.keys():
                logging.error(f"Operation argument {n} not in created variables: {vars.keys()}")
            else:
                args.append(vars[n])
        return args

    def arg_conversion(self, instance_arg, target_arg):
        if isinstance(target_arg, tuple):
            result = tuple(instance_arg for _ in range(len(target_arg)))
            return result
        else:
            return instance_arg











