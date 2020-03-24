import codegen as c
import codegen.c.cgen_utils as cutils

class ComponentInit(object):

    def __init__(self,name, vars, var_names, op_type, order):
        self.vars = vars
        self.name=name
        self.var_names = var_names
        self.ordered_flows = order
        self.order = []
        self.init_args = []
        self.op_type = op_type

    def gen_order(self):
        for f in self.ordered_flows:
            v = self.vars[f]
            if len(v['dims']) > 0:
                for d in v['dims']:
                    if d not in self.order:
                        self.order.append(d)
                        self.init_args.append(self.vars[d]['init_arg_type'])

            self.init_args.append(v['init_arg_type'])
            self.order.append(f)

    def gen_init(self):
        self.init_args.append(c.Pointer(c.Value('_' + self.name, 'self')))
        self.gen_order()
        init_statements = []
        for k in range(len(self.var_names)):
            v_name = self.var_names[k]
            v = self.vars[v_name]

            if v['init_arg_type']:
                if v['vtype'] == 'component':
                    mem_alloc = 'malloc(sizeof({struct}))'.format(struct=v['struct_dtype'])
                    instance_assign = c.Assign('self->{instance}'.format(instance=v['name']), mem_alloc)

                    init_statements.append(instance_assign)
                else:
                    vstatement = c.Assign('self->{vname}'.format(vname=v['name']), v['init_type'])
                    init_statements.append(vstatement)

                    if v['struct_dtype'] == 'state':
                        value = c.Value(v['dtype'], v['name'] + '_temp')

                        for d in v['dims']:
                            value = c.ArrayOf(value, d)

                        output_queue = c.Assign('self->{vname}_oq'.format(vname=v['name']), 'OUTPUT_QUEUE({qname})'.format(qname=v['name']))
                        init_statements.append(output_queue)

                        init_value = c.Statement('memset({temp}, 0, sizeof({temp}))'.format(temp=v['name'] + '_temp'))
                        init_statements.append(value)
                        init_statements.append(init_value)
                        write_queue = c.Statement('WRITE(self->{queue}_oq, {temp_var})'.format(queue=v['name'], temp_var=v['name'] + '_temp'))
                        init_statements.append(write_queue)
        self.signature = c.FunctionDeclaration(c.Value('component', 'init_' + self.name), self.init_args)
        declaration_block = c.Block(init_statements)

        self.function_code = c.FunctionBody(self.signature, declaration_block)
