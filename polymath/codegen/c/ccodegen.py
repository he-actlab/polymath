import os
import onnx
from shutil import copyfile
from hdfg import hdfgutils
from hdfg import load_store
import codegen as c
from .component import Component
from .includes import Includes

class CCodeGen(object):
    copy_files = ['pipe.c', 'pipe.h', 'utils.c', 'utils.h', 'csv.h', 'csv.c']
    def __init__(self, onnx_proto, run_async=False):
        self.input_proto = onnx_proto
        self.output_dir, self.output_file = os.path.split(self.input_proto)
        self.proto_name = self.output_file.split('.')[0]
        program = load_store.load_program(self.input_proto)
        self.graph = program.graph
        self.templates = program.templates
        self.components = {}
        self.includes = []
        self.functions = []
        self.structs = []
        self.signature_map = {}
        self.initializer = None
        self.header = []
        self.exec = []
        self.run_async = run_async
        self.component_gen()
        self.create_code()


    def component_gen(self):
        for t in self.templates:
            comp_code = Component(self.templates[t], run_async=self.run_async)
            comp_code.gen_init()
            comp_code.gen_instance()
            self.signature_map[t] = {'order' :comp_code.init.order,
                                     'vars' : comp_code.vars,
                                     'name' : t
                                     }
            self.components[t] = comp_code

        for t in self.templates:
            self.components[t].gen_exec(self.signature_map)

        self.add_header()
        self.add_exec()

    def add_header(self):
        for t in self.templates:
            self.header.append(self.components[t].instance.typedef)
        for t in self.templates:
            self.exec.append(self.components[t].instance.struct_body)
            self.header.append(self.components[t].execution.signature)
            self.header.append(self.components[t].init.signature)

    def add_exec(self):
        for t in self.templates:
            self.exec.append(self.components[t].execution.function_code)
            self.exec.append(self.components[t].init.function_code)

    def create_main(self):
        body = []
        main_params = self.templates['main'].parameters
        args = []
        init_args = 'main_0'
        if len(main_params) > 0:
            assert main_params[0] == 'argv'
            arg_c = c.Value('int', 'argc')
            arg_v = c.Pointer(c.ArrayOf(c.Value('char', 'argv')))
            args = [arg_c, arg_v]
            init_args += ', argc, argv'

        signature = c.FunctionDeclaration(c.Value('int', 'main'), args)
        main_struct = c.Pointer(c.Value('_main_component', 'main_0'))
        struct_assign = c.Assign(main_struct, 'malloc(sizeof(_main_component))')
        body.append(struct_assign)
        init_main = c.Statement('init_main_component({init_args})'.format(init_args=init_args))
        body.append(init_main)
        exec_main = c.Statement('main_component(main_0)')
        body.append(exec_main)
        free_main = c.Statement('free(main_0)')
        body.append(free_main)


        main_func = c.FunctionBody(signature, c.Block(body))

        return main_func

    def create_code(self):
        self.create_main()
        output_dir = self.output_dir + '/{}_ccode'.format(self.proto_name)
        header_file = output_dir + '/' + '{}.h'.format(self.proto_name)
        exec_file = output_dir + '/' + '{}.c'.format(self.proto_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for f in self.copy_files:
            if not os.path.exists(os.getcwd() + '/' + f):
                copyfile(os.getcwd() + '/codegen/c/libs/' + f, output_dir + '/' + f)

        with open(os.getcwd() + '/codegen/c/libs/Makefile', 'r') as original:
            makefile = original.read()

        with open(output_dir + '/Makefile', 'w') as modified:
            modified.write("NAME={}\n".format(self.proto_name) + makefile)



        with open(header_file, 'w') as header:
            utils_include = c.Include('utils.h', system=False)
            header.write(utils_include.__str__()+'\n\n\n')
            for h in self.header:
                header.write(h.__str__() + '\n\n')
        main_func = self.create_main()
        with open(exec_file, 'w') as exec:
            header_include = c.Include(self.proto_name + '.h', system=False)
            math_include = c.Include('tgmath.h', system=True)
            exec.write(header_include.__str__() + '\n')
            exec.write(math_include.__str__() + '\n\n\n')
            for e in self.exec:
                exec.write(e.__str__() + '\n\n')
            exec.write(main_func.__str__() + '\n\n')

