import codegen as c
from codegen.c.index import Index
import hdfg.hdfgutils as utils


class GroupFunction(object):
    limit_map = {
        'float' : 'FLT',
        'int' : 'INT',
        'str' : 'CHAR'
    }
    def __init__(self,node, bounds,bounds_dict, group_id):
        self.function_code = []
        self.expr_type = utils.get_attribute_value(node.attributes['edge_type'])
        self.bounds = bounds
        self.bounds_dict = bounds_dict
        self.expr_arg = node.input[0]
        self.function = node.op_type
        self.group_id = group_id


    def gen_code(self):

        if self.function == 'sum':
            self.sum_func()
        elif self.function == 'argmin':
            self.argmin_func()
        elif self.function == 'argmax':
            self.argmax_func()
        elif self.function == 'max':
            self.max_func()
        elif self.function == 'min':
            self.min_func()
        elif self.function == 'prod':
            self.prod_func()


# Notes:
#
    def sum_func(self):
        group_decl = c.POD(self.expr_type, self.group_id)
        init_sum = c.Assign(group_decl, 0)
        self.loop_body = c.Assign(self.group_id, '{} + {}'.format(self.group_id, self.expr_arg))
        self.function_code.append(init_sum)
        assert len(self.bounds) > 0
        func_loop = Index([self.loop_body], self.bounds_dict, self.bounds)
        self.function_code.append(func_loop.loop)

    def prod_func(self):
        group_decl = c.POD(self.expr_type, self.group_id)
        init_prod = c.Assign(group_decl, 0)
        self.loop_body = c.Assign(self.group_id, '{} * {}'.format(self.group_id, self.expr_arg))
        self.function_code.append(init_prod)
        assert len(self.bounds) > 0
        func_loop = Index([self.loop_body], self.bounds_dict, self.bounds)
        self.function_code.append(func_loop.loop)

    def min_func(self):
        group_decl = c.POD(self.expr_type, self.group_id)
        init_sum = c.Assign(group_decl, self.limit_map[self.expr_type] + '_MAX')
        self.loop_body = c.If('{} > {}'.format(self.group_id, self.expr_arg),
                              c.Assign(self.group_id, self.expr_arg))
        self.function_code.append(group_decl)
        self.function_code.append(init_sum)
        assert len(self.bounds) > 0
        func_loop = Index([self.loop_body], self.bounds_dict, self.bounds)
        self.function_code.append(func_loop.loop)

    def max_func(self):
        group_decl = c.POD(self.expr_type, 'max')
        init_sum = c.Assign('max', self.limit_map[self.expr_type] + '_MIN')
        self.loop_body = c.If('max < {}'.format(self.expr_arg),
                              c.Assign('max', self.expr_arg))
        self.function_code.append(group_decl)
        self.function_code.append(init_sum)
        assert len(self.bounds) > 0
        func_loop = Index([self.loop_body], self.bounds_dict, self.bounds)
        self.function_code.append(func_loop.loop)

    def argmin_func(self):
        self.function_code.append(c.POD(self.expr_type, 'min'))
        self.function_code.append(c.Assign('min', self.limit_map[self.expr_type] + '_MAX'))
        self.function_code.append(c.ArrayInitializer(c.POD('int32', 'amin[{len}]'.format(len=len(self.bounds))), [0 for i in self.bounds]))
        assert len(self.bounds) > 0
        block_code = []
        block_code.append(c.Assign('min', self.expr_arg))
        for b in range(len(self.bounds)):
            block_code.append(c.Assign('amin[{idx}]'.format(idx=b), '{idx}'.format(idx=self.bounds[b])))

        self.loop_body = c.If('min > {}'.format(self.expr_arg),
                              c.Block(block_code))
        func_loop = Index([self.loop_body], self.bounds_dict, self.bounds)
        self.function_code.append(func_loop.loop)

    def argmax_func(self):
        self.function_code.append(c.POD(self.expr_type, 'max'))
        self.function_code.append(c.Assign('max', self.limit_map[self.expr_type] + '_MIN'))
        self.function_code.append(c.ArrayInitializer(c.POD('int32', 'amax[{len}]'.format(len=len(self.bounds))), [0 for i in self.bounds]))
        assert len(self.bounds) > 0
        block_code = []
        block_code.append(c.Assign('max', self.expr_arg))
        for b in range(len(self.bounds)):
            block_code.append(c.Assign('amax[{idx}]'.format(idx=b), '{idx}'.format(idx=self.bounds[b])))

        self.loop_body = c.If('max < {}'.format(self.expr_arg),
                              c.Block(block_code))
        func_loop = Index([self.loop_body], self.bounds_dict, self.bounds)
        self.function_code.append(func_loop.loop)

    def get_code(self):
        return self.function_code






