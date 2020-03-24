
import re
import codegen as c
import hdfg.hdfgutils as utils
from codegen.c.component_init import ComponentInit
from codegen.c.component_instance import InstanceData
from codegen.c.component_execution import ComponentExecution
import codegen.c.cgen_utils as cutils


class Component(object):

    queue_map = {'output' : 'OUTPUT_QUEUE({qname})',
                 'input' : 'INPUT_QUEUE({qname})',
                 'component' : '{qname}'}
    def __init__(self, template, run_async):
        self.template = template
        if template.name == 'main':
            self.name = 'main_component'
        else:
            self.name = template.name
        self.graph = template.sub_graph
        self.edges = template.edge_info
        self.input = list(template.input)
        self.output = list(template.output)
        self.state = list(template.state)
        self.queues = self.input + self.output + self.state
        self.parameters = template.parameters
        self.op_type = template.op_type
        self.statements = template.statement_graphs
        self.order = utils.get_attribute_value(template.attributes['ordered_args'])
        self.vars = {}
        self.struct_members = []
        self.init_args = []
        self.assignments = []
        self.var_names = []
        self.node_map = {}
        self.components = {}

        self.create_node_map()
        self.init = ComponentInit(self.name, self.vars, self.var_names, self.op_type, self.order)
        self.instance = InstanceData(self.name, self.vars, self.var_names)
        self.execution = ComponentExecution(self.name, self.vars, self.var_names, self.statements,self.node_map,self.edges, run_async, self.op_type, template.statements)

    def add_var(self, dvar):
        if dvar['name'] not in self.var_names:
            self.var_names.append(dvar['name'])
            self.vars[dvar['name']] = dvar

    def create_node_map(self):
        for n in self.graph:
            name = n.name
            self.node_map[name] = n
            op_cat = n.op_cat

            if op_cat == 'component':
                if n.op_type not in self.components.keys():
                    self.components[n.op_type] = 0
                name = n.op_type + '_' + str(self.components[n.op_type])
                self.components[n.op_type] += 1
                init_arg = c.Pointer(cutils.create_value('_' + n.op_type, name))
                ordered_args = utils.get_attribute_value(n.attributes['ordered_args'])

                var = {'vtype': 'component',
                        'name': name,
                        'pointers' : 1,
                        'struct_dtype': '_' + n.op_type,
                        'dtype': '_' + n.op_type,
                        'dims': [],
                        'init_type' : name,
                        'init_arg_type': init_arg,
                        'cname' : n.op_type,
                        'args' : ordered_args,
                        'input' : list(n.input)
                        }
                self.add_var(var)

            elif op_cat == 'assign' or op_cat == 'argument':
                assignee = self.edges[n.output[0]]
                vid = assignee.vid
                dtype = utils.get_attribute_value(assignee.attributes['type'])
                dims = cutils.get_dims(assignee, self.edges)
                pointers = len(dims)
                default = None
                if vid in self.input:
                    struct_dtype = 'input'
                    vtype = 'flow'
                    init_type = self.queue_map['input'].format(qname=vid)
                    init_arg_type = cutils.create_value('flow', vid)
                elif vid in self.output:
                    struct_dtype = 'output'
                    vtype = 'flow'
                    init_type = self.queue_map['output'].format(qname=vid)
                    init_arg_type = cutils.create_value('flow', vid)
                elif vid in self.state:
                    struct_dtype = 'state'
                    vtype = 'flow'
                    init_type = self.queue_map['input'].format(qname=vid)
                    init_arg_type = cutils.create_value('flow', vid)
                elif vid in self.parameters:
                    struct_dtype = dtype
                    vtype = 'param'
                    init_arg_type = cutils.create_value(dtype, vid)
                    for _ in range(pointers):
                        init_arg_type = c.Pointer(init_arg_type)
                    init_type = vid
                    attributes = list(self.edges[vid].attributes)
                    if 'default' in attributes:
                        default = utils.get_attribute_value(self.edges[vid].attributes['default'])
                else:
                    vtype = 'dynamic_variable'
                    struct_dtype = None
                    init_arg_type = None
                    init_type = None


                if vid in self.queues or vid in self.parameters:
                    for d in dims:
                        dimvar = {'name' : d,
                                  'vtype' : 'dim',
                                    'dims' : [],
                                    'pointers' : 0,
                                    'struct_dtype' : 'int',
                                    'dtype' : 'int',
                                    'init_type' : d,
                                    'init_arg_type' : cutils.create_value('int', d)
                                }
                        self.add_var(dimvar)

                var = {'name' : vid,
                       'vtype' : vtype,
                        'dims' : dims,
                        'pointers' : pointers,
                        'struct_dtype' : struct_dtype,
                        'dtype' : dtype,
                        'init_type' : init_type,
                        'init_arg_type' : init_arg_type,
                       'default' : default
                        }
                self.add_var(var)
            elif op_cat == 'declaration':
                dtype = utils.get_attribute_value(self.edges[n.output[0]].attributes['type'])
                var = {'name' : n.output[0],
                       'vtype' : 'flow_declaration',
                        'dims' : list(n.input),
                        'pointers' : 0,
                        'struct_dtype' : None,
                        'dtype' : dtype,
                        'init_type' : None,
                        'init_arg_type' : None
                        }
                self.add_var(var)
            elif op_cat == 'index':
                index = self.edges[n.output[0]]
                dims =  utils.get_attribute_value(index.attributes['dimensions'])
                if len(dims) == 0:
                    dims = '(' + '(' + n.input[1] + ')' + '-' + n.input[0] + ')' + '+1'
                    vtype = 'index_declaration'
                    low = n.input[0]
                    upper = n.input[1]
                else:
                    vtype = 'multi_index'
                    low = None
                    upper = None
                var = {'name' : n.output[0],
                       'vtype' : vtype,
                        'dims' : dims,
                        'pointers' : 0,
                        'struct_dtype' : None,
                        'dtype' : 'int',
                       'init_type': None,
                       'init_arg_type': None,
                       'lower': low,
                       'upper': upper
                        }
                self.add_var(var)


    def print_struct(self):
        print(self.instance.struct)

    def print_init(self):
        print(self.init.init_func)

    def print_exec(self):
        print(self.execution.function_code)

    def gen_init(self):
        self.init.gen_init()

    def gen_instance(self):
        self.instance.gen_struct()

    def gen_exec(self, signature_map):
        self.execution.gen_exec(signature_map)
        # self.print_exec()

