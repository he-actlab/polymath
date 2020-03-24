import codegen as c
import hdfg.hdfgutils as utils
from codegen.c.statement import Statement
from codegen.c.index import Index
import codegen.c.cgen_utils as cutils
import logging

class ComponentExecution(object):

    def __init__(self, name, vars, var_names, statements,nodes,edges,run_async, op_type, statement_nodes):
        self.name = name
        self.vars = vars
        self.run_async = run_async
        self.var_names = var_names
        self.statements = statements
        self.statement_nodes = statement_nodes
        self.loop_statements = []
        self.component_instances = []
        self.flows = {
                        'input': [],
                        'output': [],
                        'state': [],
                        'flow' : []
                      }
        self.nodes = nodes
        self.free_vars = []
        self.function_body = []
        self.done = False
        self.op_type = op_type
        self.buffers = []
        self.read_func = []
        self.read_func_sizes = []
        self.read_assignees = []
        self.read_paths = []
        self.defined = []
        self.loop_start = 0
        self.instance_names = []
        self.dynamic_allocations = []
        self.edges = edges

    def gen_exec(self, signature_map):
        if self.done:
            print("Already generated code\n")
        else:
            self.signature_map = signature_map
            exec_arg = [c.Pointer(c.Value('_' + self.name, 'self'))]
            signature = c.FunctionDeclaration(c.Value('component', self.name), exec_arg)
            self.create_preamble()
            if self.run_async:
                self.function_body += self.component_instances

            if len(self.flows['input']) > 0 or self.op_type == 'spring':
                self.create_loop()

            if not self.run_async:
                self.function_body += self.component_instances

            self.free_memory()
            self.signature = signature
            self.function_code = c.FunctionBody(signature, c.Block(self.function_body))
            self.done = True

    def create_loop_body(self):
        # print(f"Component {self.name}")
        # sub_g_lengths = []
        # for n in self.statement_nodes:
        #     print(f"{n.name}\t{n.op_type}\t{n.input}\t{n.output}\t{len(n.sub_graph)}")

        for s in self.statement_nodes:
            if s.op_type == 'declaration' or s.sub_graph[0].op_cat == 'component':
                continue

            statement = Statement(s, self.edges, self.vars, self.var_names, self.defined)
            if statement.new_variable:
                self.function_body.append(statement.new_variable)
                self.free_vars.append(statement.defined[-1])
            if statement.read_func:
                self.read_func.append(statement.read_func)
                self.read_assignees.append(statement.assignee_var)
                self.read_paths.append(statement.path)
                self.read_func_sizes.append(statement.read_func_size)

            if statement.write_func:
                pass
            if len(statement.statements) > 0:
                self.loop_statements += statement.statements

    def free_memory(self):
        free_temp = 'free({name})'
        for dmem in self.dynamic_allocations:
            self.function_body.append(c.Statement(free_temp.format(name=dmem)))

        input_free_template = 'FREE_INPUT_QUEUE(self->{queue})'
        output_free_template = 'FREE_OUTPUT_QUEUE(self->{queue})'

        for v in self.free_vars:
            statement = c.Statement(free_temp.format(name=v))
            self.function_body.append(statement)

        for i in self.flows['input']:
            statement = c.Statement(input_free_template.format(queue=i))
            self.function_body.append(statement)

        for s in self.flows['state']:
            statement_input = c.Statement(input_free_template.format(queue=s))
            statement_output = c.Statement(output_free_template.format(queue=s + '_oq'))
            self.function_body.append(statement_input)
            self.function_body.append(statement_output)

        for o in self.flows['output']:
            statement = c.Statement(output_free_template.format(queue=o))
            self.function_body.append(statement)

        for cinst in self.instance_names:
            self.function_body.append(c.Statement('free(self->{instance})'.format(instance=cinst)))




    def create_preamble(self):
        for k in self.var_names:
            v = self.vars[k]

            if v['struct_dtype']:

                if v['struct_dtype'] in ['input', 'output', 'state']:
                    if len(v['dims']) == 0:
                        statement_init = c.Value(v['dtype'], v['name'])
                        buffer_var = c.Pointer(c.Value(v['dtype'],  v['name'] + '_buff'))
                        buffer_init = c.Assign(buffer_var, 'malloc(sizeof({dtype})*1)'.format(dtype=v['dtype']))
                        self.buffers.append(v['name'])
                        self.function_body.append(buffer_init)
                        self.dynamic_allocations.append(v['name'] + '_buff')

                    else:
                        self.dynamic_allocations.append(v['name'])
                        statement = c.Value(v['dtype'], '(*' + v['name'] + ')')
                        size = 'sizeof({dtype}'.format(dtype=v['dtype'])
                        for di in range(len(v['dims'])):
                            d=v['dims'][di]
                            if di > 0:
                                statement = c.ArrayOf(statement, d)
                            size += '[{dim}]'.format(dim=d)
                        statement_init= c.Assign(statement, 'malloc({size}))'.format(size=size))

                    self.function_body.append(statement_init)
                    self.defined.append(v['name'])
                    self.flows[v['struct_dtype']].append(v['name'])

                elif v['vtype'] == 'component':
                    instance_arg = 'self->{cinstance}'.format(cinstance=v['name'])
                    self.instance_names.append(v['name'])

                    args = [instance_arg] + self.create_component_instance_signature(v['args'], self.signature_map[v['cname']])

                    if (len(args)) != (len(self.signature_map[v['cname']]['order']) + 1):
                        print("Args: {}, _signature: {}".format(args,self.signature_map[v['cname']]['order'] ))

                    init_statement = c.Statement('init_{cname}({args})'.format(cname=v['cname'], args=", ".join(args)))
                    self.function_body.append(init_statement)
                    exec_statement = c.Statement('{cname}({args})'.format(cname=v['cname'], args=instance_arg))
                    self.component_instances.append(exec_statement)
                else:
                    assignee = v['init_arg_type']
                    assignment = c.Assign(assignee, 'self->{}'.format(v['name']))
                    self.defined.append(v['name'])
                    self.function_body.insert(0, assignment)


            elif v['vtype'] == 'flow_declaration':

                self.flows['flow'].append(v['name'])
                assignee = c.Value('flow', v['name'])
                if len(v['dims']) > 0:
                    size = 'sizeof({}[{}])'.format(v['dtype'], "][".join(v['dims']))
                else:
                    size = 'sizeof({})'.format(v['dtype'])
                assignment = c.Assign(assignee, 'QUEUE({size})'.format(size=size))
                self.defined.append(v['name'])

                # Need to put these executions within the while loop
                self.function_body.append(assignment)

        for f in self.flows['flow']:

            statement = c.Statement('FREE_QUEUE({queue})'.format(queue=f))
            self.function_body.append(statement)

    def create_component_instance_signature(self, instance_arguments, signature):
        instance_call_arguments = []
        argument_index = 0
        signature_variable_map = {}

        for k in range(len(signature['order'])):
            i = signature['order'][k]

            vtype = signature['vars'][i]['vtype']
            if vtype == 'dim':
                continue

            if (argument_index+1) == len(signature['order']):
                logging.error("Too many arguments to component: {} in component: {}".format(signature['name'], self.name))
                exit(0)
            elif len(instance_arguments) <= argument_index:
                defaults = signature['order'][k:]

                for d in defaults:

                    if signature['vars'][d]['default'] is None:
                        logging.error(
                            "No default value for unspecified argument in component: {} in component: {}, argument: {}".format(signature['name'], self.name,d))
                        exit(0)
                    argument_default = signature['vars'][d]['default']
                    instance_call_arguments.append(str(argument_default))
                break

            variable_dimensions = signature['vars'][i]['dims']
            instance_argument = instance_arguments[argument_index]

            if len(variable_dimensions) > 0:
                instance_dimensions = self.vars[instance_argument]['dims']

                if len(variable_dimensions) != len(instance_dimensions):
                    logging.error(
                        "Input flow {} has incorrect dimensions in component {}, instance {},  argument {}".format(instance_argument, self.name,signature['name'], i))
                    exit(0)

                for d in range(len(instance_dimensions)):
                    dimension_variable = variable_dimensions[d]

                    if dimension_variable not in signature_variable_map.keys():
                        instance_call_arguments.append(instance_dimensions[d])
                        signature_variable_map[dimension_variable] = instance_dimensions[d]

            signature_variable_map[i] = instance_argument
            if instance_argument in self.vars.keys() and self.vars[instance_argument]['struct_dtype'] in ['input', 'output']:
                instance_call_arguments.append('FLOW(self->{})'.format(instance_argument))
            else:
                instance_call_arguments.append(instance_argument)
            argument_index += 1

        return instance_call_arguments

    def create_loop(self):
        conditions = []

        read_template = 'READ(self->{queue}, {queue})'
        read_template_1d = 'READ(self->{queue}, {queue}_buff)'

        write_template = 'WRITE(self->{queue}, {queue})'
        write_template_1d = 'WRITE(self->{queue}, {queue}_buff)'

        write_state_template = 'WRITE(self->{queue}_oq, {queue})'
        write_state_template_1d = 'WRITE(self->{queue}_oq, {queue}_buff)'

        self.loop_statements.append(c.Comment('Read from inputs continually'))

        for i in self.flows['input']:
            if i in self.buffers:
                conditions.append(read_template_1d.format(queue=i))
                self.loop_statements.append(c.Assign(i, i + '_buff[0]'))
            else:
                conditions.append(read_template.format(queue=i))

        self.loop_statements.append(c.Comment('Read from states'))
        for s in self.flows['state']:
            if s in self.buffers:
                statement = c.Statement(read_template_1d.format(queue=s))
                self.loop_statements.append(c.Assign(s, s + '_buff[0]'))
            else:
                statement = c.Statement(read_template.format(queue=s))
            self.loop_statements.append(statement)

        self.create_loop_body()

        self.loop_statements.append(c.Comment('Write to outputs and states'))
        for o in self.flows['output']:
            if o in self.buffers:
                statement = c.Statement(write_template_1d.format(queue=o))
                self.loop_statements.append(c.Assign(o + '_buff[0]', o))
            else:
                statement = c.Statement(write_template.format(queue=o))

            self.loop_statements.append(statement)

        for s in self.flows['state']:
            if s in self.buffers:
                statement = c.Statement(write_state_template_1d.format(queue=s))
                self.loop_statements.append(c.Assign(s + '_buff[0]', s))
            else:
                statement = c.Statement(write_state_template.format(queue=s))

            self.loop_statements.append(statement)


        if self.op_type == 'spring':
            if len(self.read_func) == 0:
                logging.error('Spring component does not generate data')
                exit(0)
            else:
                for r in range(len(self.read_func)):
                    name = self.read_assignees[r].name
                    path = self.read_paths[r]
                    input_fp = name + '_fp'
                    self.dynamic_allocations.append(name)
                    size = self.read_func_sizes[r]
                    func = self.read_func[r]
                    bounds_dict = {}
                    if len(size) > 1:
                        declaration = c.Value('char*', '(*' + name + ')[{}]'.format(size[-1]))
                        bounds_dict['i'] = ['0', str(size[0]) + '-1']
                        bounds_dict['j'] = ['0', str(size[1]) + '-1']
                        bounds = ['i', 'j']
                        mem_free_body = c.Statement('free({}[i][j])'.format(name))
                        mem_free = Index([mem_free_body], bounds_dict, bounds)
                        self.loop_statements.append(mem_free.loop)
                    else:
                    #     bounds = ['i']
                        declaration = c.Value('char*', '(*' + name + ')')
                    #     mem_free_body = c.Statement('free({}[i])'.format(name))

                    size_text = "[" + "][".join(size) + "]"
                    declaration_init = c.Assign(declaration, 'malloc(sizeof(char *{}))'.format(size_text))

                    self.defined.append(name)
                    self.function_body.append(declaration_init)
                    self.function_body.append(c.Comment('Spring component, read lines from file'))

                    input_file = c.Pointer(c.Value('FILE', input_fp))
                    input_file_open = c.Assign(input_file, 'fopen({}, "r")'.format(path))
                    file_exists = c.If('{} == NULL'.format(input_fp), c.Statement('exit(1)'))
                    self.function_body.append(input_file_open)
                    self.function_body.append(file_exists)


                    conditions.append(func)

        loop = c.While(" && ".join(conditions), c.Block(self.loop_statements))

        self.function_body.append(loop)

        if self.op_type == 'spring':
            for r in range(len(self.read_func)):
                input_fp = self.read_assignees[r].name + '_fp'
                self.function_body.append(c.Statement('fclose({})'.format(input_fp)))







