# Generated from /Users/seankinzer/ACTLab/pmlang/pmlang.code/cmstack/PMLang/PMLang.g4 by ANTLR 4.7.2
from collections import deque

from antlr4 import *
from polymath.pmlang.antlr_generator.graphutils import *
from polymath.pmlang.antlr_generator.parser import PMLangParser
# from polymath.mgdfg.template import Template, Index, Variable, Expression
from typing import List, Union, Dict

logger = logging.getLogger(__name__)


# This class defines a complete listener for a parse tree produced by PMLangParser.
class PMLangListener(ParseTreeListener):

    def __init__(self, filename,  draw=False):
        self.graph_name = filename
        self.scope = deque()
        self.components: Dict[str, Template] = {}
        self.draw = draw

    def enterComponent_definition(self, ctx: PMLangParser.Component_definitionContext):
        self.current_component: Template = Template(ctx.IDENTIFIER().getText())
        self.fname = deque()
        self.inputs = deque()

    def exitComponent_definition(self, ctx: PMLangParser.Component_definitionContext):
        self.components[self.current_component.name] = self.current_component
        self.current_component = None

    def exitPmlang(self, ctx:PMLangParser.PmlangContext):
        for component_name, component in self.components.items():
            for node in component.nodes:
                if node.is_template_node:
                    ref_component = self.components[node.op_name]
                    component.connect_component(node, ref_component.signature_vars)

            component.connect_outputs()

    def exitFlow_declaration(self, ctx: PMLangParser.Flow_declarationContext):
        new_var_name = ctx.IDENTIFIER().getText()
        new_var = Variable(new_var_name, self.type)
        new_var.set_type_modifier("declaration")
        if len(self.domain_strs) > 0:
            new_var.add_dim_list(self.get_domain_str(ctx))
        self.current_component.add_var_symbol(new_var)

    # Enter a parse tree produced by PMLangParser#flow.
    def enterFlow(self, ctx: PMLangParser.FlowContext):
        self.default = None

        self.flow_name = ctx.flow_expression().IDENTIFIER().getText()
        dtype = ctx.dtype_specifier().getText()
        type_modifier = ctx.flow_type().getText()
        flow_var = Variable(self.flow_name, dtype)
        flow_var.set_type_modifier(type_modifier)
        self.current_component.add_argument(flow_var)

    # Enter a parse tree produced by PMLangParser#flow_expression.
    def enterFlow_expression(self, ctx: PMLangParser.Flow_expressionContext):
        self.scope.append(ctx.IDENTIFIER().getText())

        if ctx.literal():

            if ctx.literal().STRING_LITERAL():
                default_val = ctx.getText()
            elif ctx.literal().number():
                lit = ctx.literal().number()
                if lit.integer():
                    val = int(ctx.literal().getText())
                elif lit.FLOAT_NUMBER():
                    val = float(ctx.literal().getText())
                else:
                    assert lit.IMAG_NUMBER(), "Error , unknown literal type"
                    num = complex(ctx.literal().getText().replace(" ", "").replace("i", "j"))
                    real = num.real
                    imag = num.imag
                    val = [real, imag]
                default_val = val
            elif ctx.literal().complex_number():
                num = complex(ctx.literal().getText().replace(" ", "").replace("i", "j"))
                real = num.real
                imag = num.imag
                default_val = [real, imag]
            else:
                raise RuntimeError(f"Could not determine type from grammar: {ctx.getText()}")
            self.current_component.add_default_value(default_val, self.flow_name)

        self.dims = []
    # Enter a parse tree produced by PMLangParser#flow_index.
    def enterFlow_index(self, ctx: PMLangParser.Flow_indexContext):
        if ctx.IDENTIFIER():
            val_name = ctx.IDENTIFIER().getText()
            val = val_name
            is_literal = False
        else:
            val = int(ctx.DECIMAL_INTEGER().getText())
            val_name = ctx.DECIMAL_INTEGER().getText()
            is_literal = True

        self.dims.append(val)
        self.current_component.add_var_dimension(val_name, self.flow_name, is_literal)

    # Exit a parse tree produced by PMLangParser#literal.
    def exitLiteral(self, ctx: PMLangParser.LiteralContext):
        if ctx.STRING_LITERAL():
            self.default = ctx.getText()
        elif ctx.number():
            lit = ctx.number()
            if lit.integer():
                val = int(ctx.getText())
            elif lit.FLOAT_NUMBER():
                val = float(ctx.getText())
            else:
                assert lit.IMAG_NUMBER(), "Error , unknown literal type"
                num = complex(ctx.getText().replace(" ", "").replace("i", "j"))
                real = num.real
                imag = num.imag
                val = [real, imag]
            self.default = val
        elif ctx.complex_number():
            num = complex(ctx.getText().replace(" ", "").replace("i", "j"))
            real = num.real
            imag = num.imag
            self.default = [real, imag]

    def exitIndex_declaration(self, ctx: PMLangParser.Index_declarationContext):
        id = get_text(ctx.IDENTIFIER())
        lower = get_text(ctx.expression()[0])
        upper = get_text(ctx.expression()[1])
        new_idx = Index(id, lower, upper)
        self.current_component.add_index_symbol(new_idx)

    def enterPrefix_expression(self, ctx: PMLangParser.Prefix_expressionContext):
        self.scope.append(get_text(ctx))

    def exitPrefix_expression(self, ctx: PMLangParser.Prefix_expressionContext):
        self.scope.pop()

    def enterAssignment_expression(self, ctx: PMLangParser.Assignment_expressionContext):
        if not ctx.predicate_expression():
            self.assignment = get_text(ctx.expression())


    def exitStatement(self, ctx: PMLangParser.StatementContext):
        self.inputs = None
        if not ctx.declaration_statement():
            self.current_component.add_expression(self.current_expression)

    # TODO: Add logic to specify def-use
    def exitAssignment_expression(self, ctx: PMLangParser.Assignment_expressionContext):
        var_name = ctx.prefix_expression().IDENTIFIER().getText()
        src_node_id = self.get_input(ctx)

        if ctx.prefix_expression().index_value_list():
            domains = self.idx_value_list
        else:
            domains = []
        edge_id = self.current_component.assignment_var(var_name, src_node_id, domains)


    def enterIndex_value_list(self, ctx: PMLangParser.Index_value_listContext):
        self.idx_value_list = []

    def exitIndex_value(self, ctx: PMLangParser.Index_valueContext):
        if ctx.IDENTIFIER():
            self.idx_value_list.append(ctx.IDENTIFIER().getText())
        else:
            self.idx_value_list.append(ctx.DECIMAL_INTEGER().getText())

    # Exit a parse tree produced by PMLangParser#group_expression.
    def exitGroup_expression(self, ctx: PMLangParser.Group_expressionContext):
        id = get_text(ctx.GROUP_FUNCTION())
        out_node = self.current_component.add_group_node(self.get_input(ctx), id, self.idx_value_list)
        self.inputs.append(out_node)

    # Enter a parse tree produced by PMLangParser#function_expression.
    def enterFunction_expression(self, ctx: PMLangParser.Function_expressionContext):
        self.expression_list = []

    # Exit a parse tree produced by PMLangParser#function_expression.
    def exitFunction_expression(self, ctx: PMLangParser.Function_expressionContext):
        fname = ctx.function_id().getText()
        input_names = list(reversed([self.get_input(ctx) for _ in self.expression_list]))
        if fname in FUNCTIONS or fname in DATATYPE_SPECIFIERS:
            out_node = self.current_component.add_intermediate_node(input_names, fname)
            self.add_edge_domains(ctx, input_names, out_node)
        else:
            out_node = self.current_component.add_component_node(input_names, fname)

    def enterIndex_expression_list(self, ctx: PMLangParser.Index_expression_listContext):
        self.domain_strs.append([])

    def exitIndex_expression(self, ctx: PMLangParser.Index_expressionContext):

        index_expression = get_text(ctx.expression())
        curr_domains = self.get_domain_str(ctx)
        curr_domains.append(index_expression)
        self.domain_strs.append(curr_domains)

    def exitIndex_expression_list(self, ctx: PMLangParser.Index_expression_listContext):

        index_domains = self.get_domain_str(ctx)
        self.inputs.reverse()
        for idx, dom in enumerate(self.inputs):
            if (idx+1) > len(index_domains):
                break
            if isinstance(dom, int):
                index_domains[-(idx + 1)] = dom
        self.inputs.reverse()
        self.domain_strs.append(index_domains)

    def enterExpression_list(self, ctx: PMLangParser.Expression_listContext):
        for exp in range(len(ctx.expression())):
            self.expression_list.append(get_text(ctx.expression()[exp]))

    def enterExpression(self, ctx: PMLangParser.ExpressionContext):
        self.scope.append(get_text(ctx))

    def exitExpression(self, ctx: PMLangParser.ExpressionContext):
        id = self.scope.pop()

        if ctx.unary_op():
            self.is_literal = False
            op = get_text(ctx.unary_op())
            unary_input = [self.get_input(ctx)]
            out_node = self.current_component.add_intermediate_node(unary_input, op)
            self.add_edge_domains(ctx, unary_input, out_node)
        elif len(ctx.children) > 2:
            op = get_text(ctx.getChild(1))
            binary_input = list(reversed([self.get_input(ctx) for _ in range(2)]))
            out_node = self.current_component.add_intermediate_node(binary_input, op)
            self.add_edge_domains(ctx, binary_input, out_node)
        elif ctx.STRING_LITERAL():
            const_var = Variable(id, "str")
            const_var.set_type_modifier("literal")
            const_var.add_default(id)
            self.current_component.add_var_symbol(const_var)
            self.inputs.append(id)
        elif ctx.IDENTIFIER():
            self.is_literal = False
            id = get_text(ctx.IDENTIFIER())
            if ctx.index_expression_list():

                domains = self.get_domain_str(ctx)
                self.domain_dict.append({id: domains})
                input_domains = list(self.inputs)[-len(domains):]
                for dom_idx, dom_id in enumerate(domains):
                    if dom_id != input_domains[dom_idx]:
                        var_name, domain_list = self.get_domain_dict(ctx, pop_front=True)
                        edge_id = self.current_component.add_var_edge(var_name, domain_list)
                        input_domains[dom_idx] = self.current_component.get_edge(edge_id)

                node_id = self.current_component.add_index_node(input_domains, id, domains)

                for _ in range(len(domains)):
                    _ = self.get_input(ctx)

                if node_id < 0:
                    self.inputs.append(id)
                else:
                    self.inputs.append(node_id)
            else:
                self.inputs.append(id)


    # Enter a parse tree produced by PMLangParser#statement.
    def enterStatement(self, ctx: PMLangParser.StatementContext):
        self.statement = []
        self.snodes = []
        self.line = ctx.getText()
        self.loop_stmt = ctx.getText()
        self.prefix = None
        self.assignment = None
        self.inputs = deque()
        self.domain_dict = deque()
        self.domain_strs = deque()


        self.is_literal = True

        if ctx.assignment_statement():
            self.statement_type = 'assignment'
            self.current_expression = Expression(ctx.getText())
        elif ctx.expression_statement():
            self.statement_type = 'expression'
            self.current_expression = Expression(ctx.getText())
        elif ctx.declaration_statement():
            self.statement_type = 'declaration'

    # Enter a parse tree produced by PMLangParser#declaration_statement.
    def enterDeclaration_statement(self, ctx: PMLangParser.Declaration_statementContext):
        if ctx.INDEX():
            self.type = 'int'
        else:
            self.type = get_text(ctx.dtype_specifier())


    # Exit a parse tree produced by PMLangParser#declaration_statement.
    def exitDeclaration_statement(self, ctx: PMLangParser.Declaration_statementContext):
        self.type = None

    # Enter a parse tree produced by PMLangParser#predicate_expression.
    def enterPredicate_expression(self, ctx: PMLangParser.Predicate_expressionContext):
        self.predicate_expression = get_text(ctx.bool_expression())
        self.true_assign = get_text(ctx.true_expression())
        self.false_assign = get_text(ctx.false_expression())

    # Exit a parse tree produced by PMLangParser#predicate_expression.
    def exitPredicate_expression(self, ctx: PMLangParser.Predicate_expressionContext):
        predicate_inputs = []
        for _ in range(3):
            predicate_inputs.append(self.get_input(ctx))
        node_id = self.current_component.add_intermediate_node(predicate_inputs, "pred_store")
        self.inputs.append(node_id)

    # Exit a parse tree produced by PMLangParser#number.
    def exitNumber(self, ctx: PMLangParser.NumberContext):
        id  = get_text(ctx)
        if ctx.integer():
            val_str = ctx.getText()
            val = int(val_str)
            type = 'int'

        elif ctx.FLOAT_NUMBER():
            val_str = ctx.getText()
            val = float(val_str)
            type = 'float'
        else:
            num = complex(ctx.getText().replace(" ", "").replace("i", "j"))
            real = num.real
            imag = num.imag
            val_str = [str(real), str(imag)]
            val = [real, imag]
            type = 'complex'

        self.current_component.add_literal(id, type, val)
        self.inputs.append(id)

    def get_input(self, ctx):
        if len(self.inputs) == 0:
            raise RuntimeError(f"Unable to get input for {ctx.getText()}")
        else:
            top_val = self.inputs.pop()
            return top_val

    def get_domain_str(self, ctx):
        if len(self.domain_strs) == 0:
            raise RuntimeError(f"Unable to get domain str for {ctx.getText()}")
        else:
            top_val = self.domain_strs.pop()
            return top_val

    def get_domain_dict(self, ctx, pop_front=False):
        if len(self.domain_dict) == 0:
            raise RuntimeError(f"Unable to get domain dict for {ctx.getText()}")
        elif pop_front:
            top_val = self.domain_dict.popleft()
            key = list(top_val.keys())[0]
        else:
            top_val = self.domain_dict.pop()
            key = list(top_val.keys())[0]
        return key, top_val[key]


    def add_edge_domains(self, ctx, inputs: List[Union[int, str]], node_id: int):
        edge_index = 0
        while len(self.domain_dict) > 0 and edge_index < len(inputs):
            var_name, domain = self.get_domain_dict(ctx)

            added_edge = self.current_component.add_edge_domain(node_id, var_name, domain)
            if not added_edge:
                self.domain_dict.append({var_name: domain})
            edge_index += 1
        self.inputs.append(node_id)
