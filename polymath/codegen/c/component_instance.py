import codegen.c.cgen_utils as cutils
import codegen as c


class InstanceData(object):

    def __init__(self,name,vars, var_names):
        self.vars = vars
        self.var_names = var_names
        self.name = name

    def gen_struct(self):
        struct_vars = []
        for k in self.var_names:
            v = self.vars[k]

            if not v['struct_dtype']:
                continue

            dtype = v['struct_dtype'] if v['struct_dtype'] != 'state' else 'input'
            svar = cutils.create_value(dtype, v['name'])
            if dtype not in ['input', 'output', 'state']:
                for _ in range(v['pointers']):
                    svar = c.Pointer(svar)

            struct_vars.append(svar)
            if  v['struct_dtype'] == 'state':
                state_output= cutils.create_value('output', v['name']+'_oq')
                struct_vars.append(state_output)
        self.typedef = c.Typedef(c.Struct("_" + self.name, [], "_" + self.name))
        self.struct_body = c.Struct("_" + self.name, struct_vars)
