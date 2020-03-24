# Generated from ./PMLang.g4 by ANTLR 4.7.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .PMLangParser import PMLangParser
else:
    from PMLangParser import PMLangParser

# This class defines a complete listener for a parse tree produced by PMLangParser.
class PMLangListener(ParseTreeListener):

    # Enter a parse tree produced by PMLangParser#pmlang.
    def enterPmlang(self, ctx:PMLangParser.PmlangContext):
        pass

    # Exit a parse tree produced by PMLangParser#pmlang.
    def exitPmlang(self, ctx:PMLangParser.PmlangContext):
        pass


    # Enter a parse tree produced by PMLangParser#component_list.
    def enterComponent_list(self, ctx:PMLangParser.Component_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#component_list.
    def exitComponent_list(self, ctx:PMLangParser.Component_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#component_definition.
    def enterComponent_definition(self, ctx:PMLangParser.Component_definitionContext):
        pass

    # Exit a parse tree produced by PMLangParser#component_definition.
    def exitComponent_definition(self, ctx:PMLangParser.Component_definitionContext):
        pass


    # Enter a parse tree produced by PMLangParser#flow_list.
    def enterFlow_list(self, ctx:PMLangParser.Flow_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#flow_list.
    def exitFlow_list(self, ctx:PMLangParser.Flow_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#flow.
    def enterFlow(self, ctx:PMLangParser.FlowContext):
        pass

    # Exit a parse tree produced by PMLangParser#flow.
    def exitFlow(self, ctx:PMLangParser.FlowContext):
        pass


    # Enter a parse tree produced by PMLangParser#flow_expression.
    def enterFlow_expression(self, ctx:PMLangParser.Flow_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#flow_expression.
    def exitFlow_expression(self, ctx:PMLangParser.Flow_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#literal.
    def enterLiteral(self, ctx:PMLangParser.LiteralContext):
        pass

    # Exit a parse tree produced by PMLangParser#literal.
    def exitLiteral(self, ctx:PMLangParser.LiteralContext):
        pass


    # Enter a parse tree produced by PMLangParser#flow_index_list.
    def enterFlow_index_list(self, ctx:PMLangParser.Flow_index_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#flow_index_list.
    def exitFlow_index_list(self, ctx:PMLangParser.Flow_index_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#flow_index.
    def enterFlow_index(self, ctx:PMLangParser.Flow_indexContext):
        pass

    # Exit a parse tree produced by PMLangParser#flow_index.
    def exitFlow_index(self, ctx:PMLangParser.Flow_indexContext):
        pass


    # Enter a parse tree produced by PMLangParser#flow_declaration_list.
    def enterFlow_declaration_list(self, ctx:PMLangParser.Flow_declaration_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#flow_declaration_list.
    def exitFlow_declaration_list(self, ctx:PMLangParser.Flow_declaration_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#flow_declaration.
    def enterFlow_declaration(self, ctx:PMLangParser.Flow_declarationContext):
        pass

    # Exit a parse tree produced by PMLangParser#flow_declaration.
    def exitFlow_declaration(self, ctx:PMLangParser.Flow_declarationContext):
        pass


    # Enter a parse tree produced by PMLangParser#index_declaration_list.
    def enterIndex_declaration_list(self, ctx:PMLangParser.Index_declaration_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#index_declaration_list.
    def exitIndex_declaration_list(self, ctx:PMLangParser.Index_declaration_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#index_declaration.
    def enterIndex_declaration(self, ctx:PMLangParser.Index_declarationContext):
        pass

    # Exit a parse tree produced by PMLangParser#index_declaration.
    def exitIndex_declaration(self, ctx:PMLangParser.Index_declarationContext):
        pass


    # Enter a parse tree produced by PMLangParser#prefix_expression.
    def enterPrefix_expression(self, ctx:PMLangParser.Prefix_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#prefix_expression.
    def exitPrefix_expression(self, ctx:PMLangParser.Prefix_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#array_expression.
    def enterArray_expression(self, ctx:PMLangParser.Array_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#array_expression.
    def exitArray_expression(self, ctx:PMLangParser.Array_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#group_expression.
    def enterGroup_expression(self, ctx:PMLangParser.Group_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#group_expression.
    def exitGroup_expression(self, ctx:PMLangParser.Group_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#function_expression.
    def enterFunction_expression(self, ctx:PMLangParser.Function_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#function_expression.
    def exitFunction_expression(self, ctx:PMLangParser.Function_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#function_id.
    def enterFunction_id(self, ctx:PMLangParser.Function_idContext):
        pass

    # Exit a parse tree produced by PMLangParser#function_id.
    def exitFunction_id(self, ctx:PMLangParser.Function_idContext):
        pass


    # Enter a parse tree produced by PMLangParser#nested_expression.
    def enterNested_expression(self, ctx:PMLangParser.Nested_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#nested_expression.
    def exitNested_expression(self, ctx:PMLangParser.Nested_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#index_expression.
    def enterIndex_expression(self, ctx:PMLangParser.Index_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#index_expression.
    def exitIndex_expression(self, ctx:PMLangParser.Index_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#index_expression_list.
    def enterIndex_expression_list(self, ctx:PMLangParser.Index_expression_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#index_expression_list.
    def exitIndex_expression_list(self, ctx:PMLangParser.Index_expression_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#index_value.
    def enterIndex_value(self, ctx:PMLangParser.Index_valueContext):
        pass

    # Exit a parse tree produced by PMLangParser#index_value.
    def exitIndex_value(self, ctx:PMLangParser.Index_valueContext):
        pass


    # Enter a parse tree produced by PMLangParser#index_value_list.
    def enterIndex_value_list(self, ctx:PMLangParser.Index_value_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#index_value_list.
    def exitIndex_value_list(self, ctx:PMLangParser.Index_value_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#expression_list.
    def enterExpression_list(self, ctx:PMLangParser.Expression_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#expression_list.
    def exitExpression_list(self, ctx:PMLangParser.Expression_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#expression.
    def enterExpression(self, ctx:PMLangParser.ExpressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#expression.
    def exitExpression(self, ctx:PMLangParser.ExpressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#unary_op.
    def enterUnary_op(self, ctx:PMLangParser.Unary_opContext):
        pass

    # Exit a parse tree produced by PMLangParser#unary_op.
    def exitUnary_op(self, ctx:PMLangParser.Unary_opContext):
        pass


    # Enter a parse tree produced by PMLangParser#multiplicative_op.
    def enterMultiplicative_op(self, ctx:PMLangParser.Multiplicative_opContext):
        pass

    # Exit a parse tree produced by PMLangParser#multiplicative_op.
    def exitMultiplicative_op(self, ctx:PMLangParser.Multiplicative_opContext):
        pass


    # Enter a parse tree produced by PMLangParser#additive_op.
    def enterAdditive_op(self, ctx:PMLangParser.Additive_opContext):
        pass

    # Exit a parse tree produced by PMLangParser#additive_op.
    def exitAdditive_op(self, ctx:PMLangParser.Additive_opContext):
        pass


    # Enter a parse tree produced by PMLangParser#relational_op.
    def enterRelational_op(self, ctx:PMLangParser.Relational_opContext):
        pass

    # Exit a parse tree produced by PMLangParser#relational_op.
    def exitRelational_op(self, ctx:PMLangParser.Relational_opContext):
        pass


    # Enter a parse tree produced by PMLangParser#assignment_expression.
    def enterAssignment_expression(self, ctx:PMLangParser.Assignment_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#assignment_expression.
    def exitAssignment_expression(self, ctx:PMLangParser.Assignment_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#statement.
    def enterStatement(self, ctx:PMLangParser.StatementContext):
        pass

    # Exit a parse tree produced by PMLangParser#statement.
    def exitStatement(self, ctx:PMLangParser.StatementContext):
        pass


    # Enter a parse tree produced by PMLangParser#statement_list.
    def enterStatement_list(self, ctx:PMLangParser.Statement_listContext):
        pass

    # Exit a parse tree produced by PMLangParser#statement_list.
    def exitStatement_list(self, ctx:PMLangParser.Statement_listContext):
        pass


    # Enter a parse tree produced by PMLangParser#expression_statement.
    def enterExpression_statement(self, ctx:PMLangParser.Expression_statementContext):
        pass

    # Exit a parse tree produced by PMLangParser#expression_statement.
    def exitExpression_statement(self, ctx:PMLangParser.Expression_statementContext):
        pass


    # Enter a parse tree produced by PMLangParser#declaration_statement.
    def enterDeclaration_statement(self, ctx:PMLangParser.Declaration_statementContext):
        pass

    # Exit a parse tree produced by PMLangParser#declaration_statement.
    def exitDeclaration_statement(self, ctx:PMLangParser.Declaration_statementContext):
        pass


    # Enter a parse tree produced by PMLangParser#assignment_statement.
    def enterAssignment_statement(self, ctx:PMLangParser.Assignment_statementContext):
        pass

    # Exit a parse tree produced by PMLangParser#assignment_statement.
    def exitAssignment_statement(self, ctx:PMLangParser.Assignment_statementContext):
        pass


    # Enter a parse tree produced by PMLangParser#predicate_expression.
    def enterPredicate_expression(self, ctx:PMLangParser.Predicate_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#predicate_expression.
    def exitPredicate_expression(self, ctx:PMLangParser.Predicate_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#bool_expression.
    def enterBool_expression(self, ctx:PMLangParser.Bool_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#bool_expression.
    def exitBool_expression(self, ctx:PMLangParser.Bool_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#true_expression.
    def enterTrue_expression(self, ctx:PMLangParser.True_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#true_expression.
    def exitTrue_expression(self, ctx:PMLangParser.True_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#false_expression.
    def enterFalse_expression(self, ctx:PMLangParser.False_expressionContext):
        pass

    # Exit a parse tree produced by PMLangParser#false_expression.
    def exitFalse_expression(self, ctx:PMLangParser.False_expressionContext):
        pass


    # Enter a parse tree produced by PMLangParser#iteration_statement.
    def enterIteration_statement(self, ctx:PMLangParser.Iteration_statementContext):
        pass

    # Exit a parse tree produced by PMLangParser#iteration_statement.
    def exitIteration_statement(self, ctx:PMLangParser.Iteration_statementContext):
        pass


    # Enter a parse tree produced by PMLangParser#component_type.
    def enterComponent_type(self, ctx:PMLangParser.Component_typeContext):
        pass

    # Exit a parse tree produced by PMLangParser#component_type.
    def exitComponent_type(self, ctx:PMLangParser.Component_typeContext):
        pass


    # Enter a parse tree produced by PMLangParser#flow_type.
    def enterFlow_type(self, ctx:PMLangParser.Flow_typeContext):
        pass

    # Exit a parse tree produced by PMLangParser#flow_type.
    def exitFlow_type(self, ctx:PMLangParser.Flow_typeContext):
        pass


    # Enter a parse tree produced by PMLangParser#dtype_specifier.
    def enterDtype_specifier(self, ctx:PMLangParser.Dtype_specifierContext):
        pass

    # Exit a parse tree produced by PMLangParser#dtype_specifier.
    def exitDtype_specifier(self, ctx:PMLangParser.Dtype_specifierContext):
        pass


    # Enter a parse tree produced by PMLangParser#integer.
    def enterInteger(self, ctx:PMLangParser.IntegerContext):
        pass

    # Exit a parse tree produced by PMLangParser#integer.
    def exitInteger(self, ctx:PMLangParser.IntegerContext):
        pass


    # Enter a parse tree produced by PMLangParser#number.
    def enterNumber(self, ctx:PMLangParser.NumberContext):
        pass

    # Exit a parse tree produced by PMLangParser#number.
    def exitNumber(self, ctx:PMLangParser.NumberContext):
        pass


    # Enter a parse tree produced by PMLangParser#complex_number.
    def enterComplex_number(self, ctx:PMLangParser.Complex_numberContext):
        pass

    # Exit a parse tree produced by PMLangParser#complex_number.
    def exitComplex_number(self, ctx:PMLangParser.Complex_numberContext):
        pass


