import hdfg.hdfgutils as utils
from codegen.c.group_function import GroupFunction
from codegen.c.index import Index
import codegen.c.cgen_utils as cutils
from polymath.pmlang.antlr_generator.graphutils import *
import logging

import codegen as c

class Statement(object):
    dtype_map = {
                "float" : "%f",
                "int" : "%d",
                "str" : "%s"
                 }
    def __init__(self,statement, edges, vars, var_names, defined):
        self.statement_node = statement
        self.stype = statement.op_type
        self.nodes = statement.sub_graph
        self.vars = vars
        self.var_names = var_names
        self.line = utils.get_attribute_value(self.nodes[0].attributes['line'])
        self.line = self.line.replace(";", "")
        self.node_cats = [n.op_cat for n in self.nodes]
        self.edges = edges
        self.read_func = None
        self.write_func = None
        self.path = None
        self.read_func_size = None
        self.group_vals = 0
        self.statements = []
        self.new_variable = None
        self.defined = defined
        self.create_statement()

    def create_statement(self):

        if self.stype == 'expression':

            fname = self.nodes[0].op_type
            args = self.nodes[0].input
            statement = self.handle_function(fname, args)

            if statement:
                self.statements = [statement]
        else:
            self.statements += self.handle_assignment_statement()


    def handle_function(self, fname, args, assignment=None):
        fn = None
        if fname == 'fread':
            self.new_variable = None
            if len(args) != 4:
                logging.error('Incorrect amount of arguments supplied to function fread')
                exit(0)
            elif not assignment:
                logging.error('Assignment value not supplied to fread')
                exit(0)
            else:
                #TODO: Change this to work for different file types
                # while (FREAD2D(m + 1, 2, lines, infile))
                # lines = fread(path, type, m + 1, 1);
                self.path = args[0]
                if args[3] == str(1):
                    read_func = 'FREAD1D({args})'
                    fargs = [args[2], self.assignee.name, self.assignee.name + '_fp']
                    self.read_func_size = [str(args[2])]
                else:
                    read_func = 'FREAD2D({args})'
                    fargs = [args[2], args[3], self.assignee.name, self.assignee.name + '_fp']
                    self.read_func_size = [str(args[3]), str(args[2])]
                read_statement = read_func.format(args=", ".join(fargs))
                self.read_func = read_statement
            return None
        elif fname == 'fwrite':
            self.new_variable = None
            if len(args) != 3:
                logging.error('Incorrect amount of arguments supplied to function fwrite')
                exit(0)
            else:
                self.path = args[1]
                self.new_variable = None
                var_name = args[0]
                var_name_write = var_name
                if var_name not in self.vars.keys():
                    logging.error("Variable {} does not exist in scope for writing. Exiting.".format(var_name))
                    exit(1)
                var = self.vars[var_name]
                indices = ['i_', 'j_', 'k_', 'l_', 'm_']
                bounds_dict = {}
                bounds = []
                for i in range(len(var['dims'])):
                    bounds_dict[indices[i]] = [str(0), '(' + str(var['dims'][i]) + ')' + '-1']
                    bounds.append(indices[i])
                    var_name_write += '[' + indices[i] + ']'
                dtype_spec = self.dtype_map[var['dtype']]
                write_func_body = c.Statement('fprintf({},"{},", {}'.format(var_name + '_fp',dtype_spec, var_name_write))
                self.write_func = Index([write_func_body], bounds_dict, bounds)
            return None
        elif len(args) == 0:
            fn = cutils.FUNCTION_MACRO_MAP[fname]
        elif len(args) == 1:
            fn = cutils.FUNCTION_MACRO_MAP[fname].format(val=args[0])
        else:
            fn = None
        return fn



    def handle_assignment_statement(self):
        assign = []

        if self.node_cats[-1] == 'assign':
            self.assignee = self.edges[self.nodes[-1].output[0]]
        else:
            self.assignee = self.edges[self.nodes[-2].output[0]]

        assignee_attr = list(self.assignee.attributes)
        vid = self.assignee.vid
        vid_type = utils.get_attribute_value(self.edges[vid].attributes['vtype'])
        if vid_type not in ['index', 'scalar'] and vid not in self.defined:
            new =  self.vars[vid]
            self.new_variable = c.Value(new['dtype'], '(*' + vid + ')')
            size_text = 'sizeof({dtype}'.format(dtype=new['dtype'])
            for di in range(len(new['dims'])):
                d = new['dims'][di]
                if di > 0:
                    self.new_variable = c.ArrayOf(self.new_variable, d)
                size_text += '[{dim}]'.format(dim=d)
            self.new_variable = c.Assign(self.new_variable, 'malloc({size}))'.format(size=size_text))
            self.defined.append(vid)

        if self.assignee.iid != '':
            iid = self.assignee.iid
            vtype = utils.get_attribute_value(self.edges[iid].attributes['vtype'])
            if vtype == 'index':
                loop_statement = True
            else:
                loop_statement = False
        else:
            loop_statement = False

        orig_line = self.line
        # text_map = {"e()" : "M_E"}
        text_map = {}
        nnames = [n.name for n in self.nodes]
        # print(f"Original line: {self.line}\t {nnames}")
        for n in list(self.nodes)[::-1]:

            op = n.name
            op_cat = n.op_cat
            op_type  = n.op_type
            inputs = n.input
            if op_cat == 'group':
                index_id = inputs[-1]
                bounds, bounds_dict = cutils.get_index_dims(index_id, self.edges)
                temp_var = 'group_' + str(self.group_vals)
                self.group_vals +=1
                group = GroupFunction(n, bounds, bounds_dict, temp_var)
                group.gen_code()
                assign += group.function_code

                if op in text_map.keys():
                    op = text_map[op]

                text_map[op] = temp_var
                self.line = self.line.replace(op, temp_var)

            elif op_cat == 'function':
                fn = self.handle_function(op_type,n.input, assignment=self.assignee.name)
                if fn:
                    if n.output[0] in text_map.keys():
                        op = text_map[n.output[0]]
                    else:
                        op = n.output[0]

                    text_map[op] = fn
                    self.line = self.line.replace(op, fn)

            elif op_type == 'exp':
                if inputs[0] in text_map.keys():
                    op1 = text_map[inputs[0]]
                else:
                    op1 = inputs[0]

                if inputs[1] in text_map.keys():
                    op2 = text_map[inputs[1]]
                else:
                    op2 = inputs[1]



                if op in text_map.keys():
                    op = text_map[op]

                # if op in self.line:
                #     print(f"Test 1 {op} in {self.line}")
                # elif '(' + op + ')' in self.line:
                #     print(f"Test 2 {'(' + op + ')'} in {self.line}")
                # else:
                #     temp_op1 = '(' + op1 + ')'
                #     temp_op2 = '(' + op2 + ')'
                #     test1 = temp_op1 + '^' + temp_op2
                #     test2 = temp_op1 + '^' + op2
                #     test3 = op1 + '^' + temp_op2
                #     if test1 in self.line:
                #         print(f"test 3: {test1} in {self.line}")
                #     elif test2 in self.line:
                #         print(f"test 4: {test2} in {self.line}")
                #     elif test3 in self.line:
                #         print(f"test 5: {test3} in {self.line}")
                #     else:
                #         print(f"No replacements for {self.line} with {op}")

                new_pow = 'pow({op1},{op2})'.format(op1=op1,op2=op2)
                text_map[op] = new_pow
                self.line = self.line.replace(op,new_pow)

        # print(f"Final line: {self.line}")
        # print("\n")
        assign.append(c.Statement(self.line))

        if loop_statement:
            bounds, bounds_dict = cutils.get_index_dims(iid, self.edges)
            idx = Index(assign, bounds_dict, bounds)
            return [idx.loop]
        elif self.read_func:
            return []
        else:
            return assign




