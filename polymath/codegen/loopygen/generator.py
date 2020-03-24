import os
import polymath.mgdfg.serialization.mgdfg_utils as hu
from polymath.mgdfg.serialization import load_store
import loopy as lp
import numpy as np
import re

INSN_RE = re.compile(
        r"^"
        r"\s*"
        r"(?P<lhs>[^{]+?)"
        r"\s*(?<!\:)=\s*"
        r"(?P<rhs>.+?)"
        r"\s*?"
        r"(?:\{(?P<options>.+)\}\s*)?$")

class LoopyGen(object):
    def __init__(self, cd_proto,
                 target=lp.CTarget(),
                 test_instructions=False,
                 print_domains=False,
                 print_instructions=False,
                 print_variables=False,
                 print_assumptions=False,
                 print_codegen=True
                 ):
        # Verbose debugging options
        self.test_instructions = test_instructions
        self.print_domains = print_domains
        self.print_instructions = print_instructions
        self.print_variables = print_variables
        self.print_assumptions = print_assumptions
        self.print_codegen = print_codegen

        # Target output
        self.target = target

        # Reading from proto
        self.input_proto = cd_proto
        self.output_dir, self.output_file = os.path.split(self.input_proto)
        self.proto_name = self.output_file.split('.')[0]
        program = load_store.load_program(self.input_proto)
        self.graph = program.graph
        self.templates = program.templates
        self.kernels = {}
        self.program = None

        self.function_gen()



    def function_gen(self):
        for t in self.templates:

            domains = self.get_domains(t)
            instr = self.get_instr(t)
            vars = self.get_vars(t)

            assumptions = self.templates[t].assumptions
            is_callee = True if t != 'main' else False
            if not is_callee:
                continue

            self.debug_printing(domains, instr, vars, assumptions, t)
            knl = lp.make_kernel(domains,
                                 instr,
                                 vars,
                                 name=t,
                                 seq_dependencies=True,
                                 assumptions=assumptions,
                                 is_callee_kernel=is_callee,
                                 target=self.target)
            knl = self.make_inames_unique(knl)
            print(knl)

            if self.print_codegen:
                self.debug_codegen(knl)


    def make_inames_unique(self, knl):

        for i in knl.instructions:
            print(type(i.expression))
            inames = self.recurse_expr(i.expression)
            if len(inames) > 0:
                iid = i.id
                iname_str = ",".join(inames)
                knl = lp.duplicate_inames(knl, iname_str, within=f"id:{iid}")

        return knl

    def recurse_expr(self, expr):

        if 'operation' in dir(expr) and isinstance(expr, lp.symbolic.Reduction):
            return list(expr.inames)
        elif 'children' in dir(expr):
            inames=[]
            for c in expr.children:
                inames += self.recurse_expr(c)
        else:
            return []
        return inames

    def debug_codegen(self, knl):
        print(lp.generate_code_v2(knl).all_code())


    def debug_printing(self, domains, instr, vars, asmp, cname):

        if self.print_domains:
            self.show_instructions(domains, cname)

        if self.print_instructions:
            self.show_instructions(instr, cname)

        if self.print_variables:
            self.show_instructions(vars, cname)

        if self.print_assumptions:
            print(f"----------Printing Assumptions for {cname}:--------")
            print(f"\t{asmp}")
            print("-------------------------------------------------------------")
            print("\n")




    def show_vars(self, vars, cname):
        print(f"----------Printing Variables for {cname}:--------")
        for v in vars:
            print(f"\t{v}")
        print("-------------------------------------------------------------")
        print("\n")

    def show_domains(self, domains, cname):
        print(f"----------Printing Domains for {cname}:--------")
        for d in domains:
            print(f"\t{d}")
        print("-------------------------------------------------------------")
        print("\n")


    def show_instructions(self, insns, cname):
        print(f"----------Printing Instructions for {cname}:--------")
        for i in insns:
            print(f"\t{i}")
        print("-------------------------------------------------------------")
        print("\n")


    def get_vars(self, cname):
        vars = []
        comp = self.templates[cname]

        # Parameters are constant arguments
        for p in list(comp.parameters):
            p_edge = comp.edge_info[p]
            dtype = hu.get_attribute_value(p_edge.attributes['type'])
            dims = self.get_edge_dims(p_edge)
            param_var = lp.ConstantArg(p,dtype,shape=dims)
            vars.append(param_var)

        # Need to add read only option for inputs
        # Inputs, outputs, and states represent
        global_args = list(comp.input) + list(comp.output) + list(comp.state)
        dim_vars = []
        for g in global_args:
            g_edge = comp.edge_info[g]
            dtype = hu.get_attribute_value(g_edge.attributes['type'])
            dims = self.get_edge_dims(g_edge)
            for d in dims:
                if d not in dim_vars:
                    dim_vars.append(d)
                    dim_var = lp.ValueArg(d, dtype=np.int32)
                    vars.append(dim_var)
            if len(dims) == 0:
                g_var = lp.ValueArg(g,dtype=dtype)
            else:
                g_var = lp.GlobalArg(g,dtype=dtype, shape=dims)
            vars.append(g_var)

        # Each flow declaration is a temporary variable declared in the comp scope
        for s in comp.statements:

            if s.op_type == 'declaration' and s.op_cat == 'declaration':
                d_vars = list(s.output)
                for d in d_vars:
                    edge_decl = comp.edge_info[d]
                    dims = self.get_edge_dims(edge_decl)
                    dtype = hu.get_attribute_value(edge_decl.attributes['type'])
                    t_var = lp.TemporaryVariable(d, dtype=dtype, shape=dims)
                    vars.append(t_var)

        return vars


    def get_edge_dims(self, edge):

        if 'dimensions' in list(edge.attributes):
            dims = tuple(hu.get_attribute_value(edge.attributes['dimensions']))
        else:
            dims = tuple()

        return dims

    def get_instr(self, cname):
        instr = list(self.templates[cname].instructions)

        # self.test_parse_insn(instr)
        if len(instr) == 0:
            print(f"{cname} includes no instructions")
            return []
        elif self.test_instructions:
            for i in instr:
                assign_instr = INSN_RE.match(i)
                if assign_instr is not None:
                    assign_instr = assign_instr.groupdict()
                    print(f"lhs: {assign_instr['lhs']}, rhs: {assign_instr['rhs']}")

        return instr

    # def test_parse_insn(self, instrs):
    #     from loopy.kernel.creation import parse_insn, get_default_insn_options_dict
    #     insn_opts = [get_default_insn_options_dict()]
    #     for insn in instrs:
    #         print(f"Testing instruction: {insn}")
    #         insn_match = INSN_RE.match(insn)
    #
    #         if insn_match is not None:
    #             insn, insn_inames_to_dup = parse_insn(
    #                 insn_match.groupdict(), insn_opts[-1])
    #             continue



    def get_domains(self, cname):

        domains = list(self.templates[cname].domains)
        if len(domains) == 0:
            return ['{:}']
        else:
            return domains







