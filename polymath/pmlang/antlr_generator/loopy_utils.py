import loopy as lp

import numpy as np
from polymath.pmlang.antlr_generator.parser import PMLangParser
from typing import Optional, Tuple, List

def create_loop_kernel(component_name, domains, instructions, edges, signature):

    domains, assumption_string = create_domain_string(domains, edges)
    globals = create_globals(signature, edges)


    knl = lp.make_function(
        domains,
        instructions,
        globals + ["..."],
        name=component_name,
        assumptions=assumption_string,
        target=lp.CTarget()
    )
    knl = add_instruction_deps(knl)
    if component_name == 'main':
        print(lp.generate_code_v2(knl).device_code())
    return knl

def add_instruction_deps(knl):
    assignees = {}

    for i in knl.instructions:
        assignee = i.assignee_name
        if assignee not in assignees.keys():
            assignees[assignee] = i.id
        else:
            knl = lp.add_dependency(knl, f"id:{i.id}", f"id:{assignees[assignee]}")
            assignees[assignee] = i.id
    return knl


def combine_domains(knl_domains):

    if isinstance(knl_domains, list):
        return _combine_domains_from_list(knl_domains)
    else:
        return _combine_domains_from_kernel(knl_domains)

def _combine_domains_from_list(domain_strs):
    from loopy.kernel.creation import parse_domains
    import islpy as isl
    domains = parse_domains(domain_strs, {})
    result = None
    for dom in domains:
        if result is None:
            result = dom
        else:
            aligned_dom, aligned_result = isl.align_two(
                dom, result, across_dim_types=True)
            result = aligned_result & aligned_dom
    return result

def _combine_domains_from_kernel(knl):
    import islpy as isl
    result = None
    for dom in knl.domains:
        if result is None:
            result = dom
        else:
            aligned_dom, aligned_result = isl.align_two(
                    dom, result, across_dim_types=True)
            result = aligned_result & aligned_dom

    return result

def create_domain_string(domains, edges):
    assumptions = []
    d_str = ["{" + f"[{k}] : {domains[k]}" + "}" for k in domains.keys()]
    for k in domains.keys():
        if 'lower' in edges[k].keys():
            a = f"{edges[k]['lower']} >= 0"
            if a not in assumptions and not edges[k]['lower'].isdigit():
                assumptions.append(a)
    assumption_str = " and ".join(assumptions)
    return d_str, assumption_str

def create_globals(signature, edges):

    globals = []
    dimension_vars = []
    for s in signature['args']:
        if s in signature['param']:
            g = lp.ValueArg(s, dtype=edges[s]['type'])
            globals.append(g)
        else:
            dims = edges[s]['dimensions']
            for d in dims:
                if d in dimension_vars:
                    continue
                g = lp.ValueArg(d,dtype=np.int32)
                globals.append(g)
                dimension_vars.append(d)
            g = lp.GlobalArg(s, shape=tuple(dims), dtype=edges[s]['type'])
            globals.append(g)


    return globals

def make_component(cname, args, edges):

    dim_vals = []
    for a in args:
        dims = edges[a]['dimensions']
        for d in dims:
            if d in dim_vals:
                continue
            dim_vals.append(d)
    dim_vals = dim_vals + args
    instruction_str = f"{cname}({', '.join(dim_vals)})"
    return instruction_str

def parse_insn(instruction_text):
    import loopy as kc

    inst,_,_ = kc.parse_instructions(instruction_text, {})

    return inst

def add_temporary(stmt):
    return '<>' + stmt

def process_pow(stmt):
    return stmt.replace("^", "**")

def process_if_stmt(prefix, predicate, true_stmt, false_stmt):

    cond_stmt = f"{prefix} = if({predicate}, {true_stmt}, {false_stmt})"
    return cond_stmt

def process_constant(stmt, constant):
    if constant == 'e':
        val = np.e
    elif constant == 'pi':
        val = np.pi
    else:
        val = None
    return stmt.replace(f"{constant}()", str(val))

def process_cdlang_stmt(stmt):

    new_stmt = stmt.replace('][', ',')
    new_stmt = new_stmt.replace(";", "")
    return new_stmt

def make_group_func(lhs, reduce_func, group_expr):
    import re
    group_expr = process_cdlang_stmt(group_expr)

    if 'arg' in reduce_func:
        new_arr_expr = reduce_func.replace("][", ",").replace("[", "((").replace("]", "), ", 1)
        indices = re.search('\(\((.*)\)', new_arr_expr).group(1)
        new_arr_expr = new_arr_expr.replace(f"({indices})", f"({indices}),({indices})")
        rhs = f"{new_arr_expr}{group_expr})"
        stmt = f"{lhs}={rhs}"
    else:
        new_arr_expr = reduce_func.replace("][", ",").replace("[", "((").replace("]", "), ", 1)
        rhs = f"{new_arr_expr}{group_expr})"
        stmt = f"{lhs}={rhs}"

    return stmt

def make_lex_schedule(knl):
    import islpy as isl
    from polymath.loopy.loopy.schedule import RunInstruction
    ctx = knl.isl_context

    for i in knl.instructions:
        insn_id = i.id
        # print(i.depends_on)
    #
    for s in knl.schedule:
        # if isinstance(s, RunInstruction):
        print(s.hash_fields)
        if 'iname' in s.hash_fields:
            print(s.iname)
        # print("\n".join(dir(s)))

def generate_loopy_expr(expr_str: str, expr: PMLangParser.ExpressionContext) -> Tuple[str,PMLangParser.ExpressionContext]:

    if not isinstance(expr, PMLangParser.ExpressionContext):
        raise TypeError(f"Invalid type for loopy expression gen: {expr.getText()}")

    if expr.group_expression():
        func_name = expr.group_expression().GROUP_FUNCTION().getText()
        index_list = []

        for index_expr in expr.group_expression().index_value_list().children:
            if index_expr.IDENTIFIER():
                index_expr_text = index_expr.IDENTIFIER().getText()
            else:
                index_expr_text = index_expr.NUMBER().getText()
            index_list.append(index_expr_text)
        func_arg_str, _ = generate_loopy_expr("", expr.group_expression().expression())
        expr_text = f"{func_name}(({', '.join(index_list)}), {func_arg_str})"
    else:
        expr_text = expr.getText()

    return expr_text, expr

def make_domain(prog):

    knl_code = lp.generate_code_v2(prog)

    for id, v in knl_code.implemented_domains.items():
        # print(v[0].compute_schedule())
        print(v[0].to_str())
        print(dir(v[0]))
        # print(f"{id}: {v}")
