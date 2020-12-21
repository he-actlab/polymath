# Generated from /Users/seankinzer/ACTLab/pmlang/pmlang.code/cmstack/PMLang/PMLang.g4 by ANTLR 4.7.2
import inspect
from collections import deque

from antlr4 import *
from polymath.pmlang.antlr_generator.graphutils import *
from polymath.pmlang.antlr_generator.parser import PMLangParser
from polymath.srdfg.serialization import pmlang_mgdfg
import polymath.pmlang.antlr_generator.loopy_utils as lu
from polymath.srdfg.template import Template, Index, Variable

logger = logging.getLogger(__name__)


# This class defines a complete listener for a parse tree produced by PMLangParser.
class PMLangListener(ParseTreeListener):

    def __init__(self, filename,  draw=False):
        self.graph_name = filename
        self.scope = deque()
        self.dimensions = deque()
        self.components = {}
        self.draw = draw

    def add_node(self, key, copy_node=None,  **kwargs):
        self.snodes.append(key)

        if key in self.nodes.keys():
            current_frame = inspect.currentframe()
            callframe = inspect.getouterframes(current_frame, 2)
            logger.warning("{}:Node Key {} already exists".format(callframe[1][3], key))

        if copy_node:
            self.nodes[key] = copy_node
            self.update_node(key, 'line', self.line)
            self.update_node(key, 'statement_type', self.statement_type)
            self.statements.append(key)
        else:
            self.nodes[key] = kwargs
            self.update_node(key, 'line', self.line)
            self.update_node(key, 'statement_type', self.statement_type)

            if self.is_predicate:
                self.nodes[key]['predicate'] = self.predicate
                self.nodes[key]['predicate_bool'] = self.predicate_expression
            self.statements.append(key)
            if 'op' not in kwargs.keys() or not kwargs['op']:
                current_frame = inspect.currentframe()
                callframe = inspect.getouterframes(current_frame, 2)
                logger.warning("{}:Node op doesn't exist".format(callframe[1][3], key))
            for o in self.nodes[key]['outputs']:
                if o not in self.edges.keys():
                    logger.critical("Output {} not in edges"
                                    " for node {}".format(o, key))
                    continue
                if len(self.edges[o]['src']) > 1:
                    logging.warning("adding more node srcs to {} in node {}"
                          .format(o, key))
                self.edges[o]['src'].append(key)

            for i in self.nodes[key]['inputs']:
                if i not in self.edges.keys():
                    logger.warning("Input {} not in edges"
                                    " for node {}".format(i, key))
                    continue
                self.edges[i]['dst'].append(key)

    def add_symbol(self, key):

        if 'vtype' not in self.edges[key].keys():
            logger.error("Error, invalid vtype doesn't exist for key {}, in {}".format(key, self.line))
        if self.edges[key]['vtype'] == 'index':
            self.symbols['index'][key] = "$i" + str(len(self.symbols['index'].keys()))
        elif self.edges[key]['vtype'] == 'var':
            self.symbols['var'][key] = "$t" + str(len(self.symbols['var'].keys()))
        elif self.edges[key]['vtype'] == 'scalar':
            self.symbols['scalar'][key] = key
        else:
            logger.warning("Error, invalid vtype for key {}".format(key))


    def update_node(self, node, key, value):
        self.nodes[node][key] = value

    def update_edge(self, edge, key, value):
        self.edges[edge][key] = value

    def add_edge(self,key,copy_edge=None, **kwargs):
        current_frame = inspect.currentframe()
        callframe = inspect.getouterframes(current_frame, 2)


        if key in self.edges.keys():
            logger.info("{}:Edge Key {} already exists".format(callframe[1][3], key))
        elif copy_edge:
            self.edges[key] = self.edges[copy_edge].copy()
            self.edges[key]['src'] = []
            self.edges[key]['dst'] = []
            self.add_symbol(key)

        else:
            self.edges[key] = kwargs
            self.edges[key]['src'] = []
            self.edges[key]['dst'] = []
            self.add_symbol(key)
        if 'type' not in self.edges[key].keys():
            logger.critical("{}:Edge Key {} does not have type {}".format(callframe[1][3], key, self.edges[key].keys()))

    def output_inference(self, inputs, output):

        types = []
        dims = []
        vtypes = []
        for i in inputs:
            if i not in self.edges.keys():
                logger.warning("Error, no declaration of input {} for output {} to infer information".format(i, output))
            elif 'type' not in self.edges[i].keys():
                logger.warning("Error, no type for input {} for output {} to infer type".format(i, output))
            elif 'dimensions' not in self.edges[i].keys():
                logger.warning("Error, no dimensions for input {} for output {} to infer type".format(i, output))
            elif 'vtype' not in self.edges[i].keys():
                logger.warning("Error, no vtype for input {} for output {} to infer type".format(i, output))
            types.append(self.edges[i]['type'])
            dims.append(self.edges[i]['dimensions'])
            vtypes.append(self.edges[i]['vtype'])


        type = infer_type(types, inputs, output)
        dim = infer_dims(dims,inputs,  output)
        vtype = infer_vtype(vtypes,inputs,  output)

        return type, dim, vtype

    def function_output_inference(self, inputs, output, fname):
        type = STRING_FUNCTION_TO_STRING_TYPE[fname]
        if fname in ['fread', 'fwrite']:
            dims = [inputs[-1], inputs[-2]]
            vtype = 'var'
        elif len(inputs) == 0:
            dims = []
            vtype = 'scalar'
            self.loop_stmt = lu.process_constant(self.loop_stmt, fname)
        else:
            if len(inputs) > 1 and fname not in ['fread', 'fwrite']:
                logger.warning("Too many inputs to function {}".format(fname))
            dims = self.edges[inputs[0]]['dimensions']
            vtype = self.edges[inputs[0]]['vtype']

        return type, dims, vtype

    def exitPmlang(self, ctx:PMLangParser.PmlangContext):

        self.program = pmlang_mgdfg.create_mgdfg(self.components, self.graph_name)



    # Enter a parse tree produced by PMLangParser#component_definition.
    def enterComponent_definition(self, ctx:PMLangParser.Component_definitionContext):
        # NEW
        self.current_component = Template(ctx.IDENTIFIER().getText())
        # NEW

        self.scope.append(ctx.IDENTIFIER().getText())
        self.is_predicate = False
        self.component = ctx.IDENTIFIER().getText()
        self.statement_graphs = []
        self.nodes = {}
        self.edges = {}
        self.component_type = ctx.component_type().getText()
        self.statements = []
        self.statement_inputs = []
        self.s_input = {'inputs' : [],
                        'outputs': []}
        self.inputs = []
        self.states = []
        self.outputs = []
        self.params = []
        self.args = []
        self.executions = []
        self.fname = deque()
        self.statement_type = None
        self.symbols = {
            'index' : {},
            'var' : {},
            'scalar' : {}
        }
        self.snodes = []

        self.signature = {'args' : self.args,
                          'input' : self.inputs,
                          'state' : self.states,
                          'output' : self.outputs,
                          'param' : self.params}

        ##### Loopy vars ###
        self.domains = {}
        self.instructions = []
        self.vars = []
        ####################

        self.line = ctx.getText()

    # Exit a parse tree produced by PMLangParser#component_definition.
    def exitComponent_definition(self, ctx: PMLangParser.Component_definitionContext):
        scope = self.scope.pop()
        domains, assumptions = lu.create_domain_string(self.domains, self.edges)
        self.instructions = [i.replace(";","") for i in self.instructions]
        self.components[scope] = {'statements': self.statements,
                                    '_signature' : self.signature,
                                    'nodes' : self.nodes,
                                    'edges' : self.edges,
                                    'symbols' : self.symbols,
                                    'executions' : self.executions,
                                    'statement_graphs' : self.statement_graphs,
                                    'op_type' : self.component_type,
                                    'statement_inputs' : self.statement_inputs,
                                    'domains' : domains,
                                    'instructions': self.instructions,
                                    'assumptions': assumptions
                                  }
        # if len(self.instructions) > 0:
        #     knl = lu.create_loop_kernel(scope, self.domains, self.instructions, self.edges, self._signature)
        del self.inputs
        del self.states
        del self.outputs
        del self.params
        del self.args
        del self.nodes
        del self.edges
        del self.statements
        del self.statement_inputs
        del self.executions
        del self.signature
        del self.symbols
        ## LOOPY #######ic, h, w, n
        del self.domains
        del self.instructions
        del self.vars
        #################


    # Enter a parse tree produced by PMLangParser#flow_list.
    def enterFlow_list(self, ctx:PMLangParser.Flow_listContext):
        self.param_declared = False

    def enterFlow_declaration(self, ctx:PMLangParser.Flow_declarationContext):
        self.index_type.append('flow_declaration')
        # NEW
        if ctx.IDENTIFIER():
            new_var_name = ctx.IDENTIFIER().getText()
        else:
            new_var_name = ctx.array_expression().IDENTIFIER().getText()

        new_var = Variable(new_var_name, self.type)
        new_var.set_type_modifier("declaration")
        self.current_component.add_var_symbol(new_var)
        # NEW


    # Enter a parse tree produced by PMLangParser#flow_declaration.
    def exitFlow_declaration(self, ctx:PMLangParser.Flow_declarationContext):
        self.index_type.pop()
        if ctx.IDENTIFIER():
            id = ctx.IDENTIFIER().getText()
            if id in self.edges.keys():
                logger.warn("Redeclaration of flow: {}".format(ctx.getText()))
            edge_attributes = {
                'type': self.type,
                'dimensions': [],
                'vtype': 'var',
                'vcat': 'declaration',
                'vid': id,
                'iid': None,
                'default': None
            }
            self.add_edge(id, **edge_attributes)

            node_attributes = {
                'id': id,
                'inputs': [],
                'outputs': [id],
                'op_cat': 'declaration',
                'op': 'edge',
                'edge_type': self.type
            }
            self.add_node(id, **node_attributes)
            self.index = None

    # Enter a parse tree produced by PMLangParser#flow.
    def enterFlow(self, ctx:PMLangParser.FlowContext):
        self.default = None

        # NEW
        self.flow_name = ctx.flow_expression().IDENTIFIER().getText()
        dtype = ctx.dtype_specifier().getText()
        type_modifier = ctx.flow_type().getText()
        flow_var = Variable(self.flow_name, dtype)
        flow_var.set_type_modifier(type_modifier)
        self.current_component.add_argument(flow_var)
        # NEW

    # Enter a parse tree produced by PMLangParser#flow_expression.
    def enterFlow_expression(self, ctx:PMLangParser.Flow_expressionContext):
        self.scope.append(ctx.IDENTIFIER().getText())

        # NEW
        if ctx.literal():
            default_val = ctx.literal().getText()
            self.current_component.add_default_value(default_val, self.flow_name)
        # NEW

        self.dims = []

    # Enter a parse tree produced by PMLangParser#flow_index.
    def enterFlow_index(self, ctx:PMLangParser.Flow_indexContext):
        id = ctx.IDENTIFIER().getText()
        self.dims.append(id)

        # NEW
        self.current_component.add_var_dimension(id, self.flow_name)
        # NEW

        edge_attributes = {
            'type' : 'int',
            'dimensions' : [],
            'vtype' : 'var',
            'vcat': 'argument',
            'vid' : id,
            'iid' : None,
            'default' : None
        }
        self.add_edge(id, **edge_attributes)
        self.update_edge(id, 'src', [id])

    def exitFlow_expression(self, ctx:PMLangParser.Flow_expressionContext):
        self.dimensions.append(self.dims)

    # Exit a parse tree produced by PMLangParser#flow.
    def exitFlow(self, ctx:PMLangParser.FlowContext):
        flow_type = ctx.flow_type().getText()
        self.flow_name = None

        if flow_type == 'param':
            self.param_declared = True

        scope = self.scope.pop()
        type = ctx.dtype_specifier().getText()
        self.signature[flow_type].append(scope)
        self.signature['args'].append(scope)
        edge_attributes = {
            'type' : type,
            'dimensions' : self.dimensions.pop(),
            'vtype' : 'var',
            'vcat' : 'argument',
            'vid' : scope,
            'iid' : None,
            'default' : self.default
        }

        self.add_edge(scope, **edge_attributes)
        node_attributes = {
            'id' : get_text(ctx.flow_expression()),
            'inputs' : self.edges[scope]['dimensions'],
            'outputs' : [scope],
            'op_cat' : 'argument',
            'op' : 'connect',
            'edge_type' : type,
        }
        self.add_node(get_text(ctx.flow_expression()), **node_attributes)


    # Exit a parse tree produced by PMLangParser#literal.
    def exitLiteral(self, ctx:PMLangParser.LiteralContext):
        if ctx.STRING_LITERAL():
            self.default = ctx.getText()
        elif ctx.number():
            lit = ctx.number()
            if lit.integer():
                val = int(ctx.getText())
            elif lit.FLOAT_NUMBER():
                val = float(ctx.getText())
            else:
                assert lit.IMAG_NUMBER(), "Error, unknown literal type"
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

    # Exit a parse tree produced by PMLangParser#index_declaration.
    def exitIndex_declaration(self, ctx:PMLangParser.Index_declarationContext):
        id = get_text(ctx.IDENTIFIER())
        self.s_input['outputs'].append(id)
        lower = get_text(ctx.expression()[0])
        upper = get_text(ctx.expression()[1])

        domain_str = f"{lower} <= {id} <= {upper}"
        # NEW
        new_idx = Index(id, lower, upper)
        self.current_component.add_index_symbol(new_idx)
        # NEW
        self.domains[id] = domain_str

        edge_attributes = {
            'type' : self.type,
            'dimensions' : [],
            'vtype' : 'index',
            'vcat'  : 'declaration',
            'vid' : None,
            'iid' : id,
            'default' : self.default,
            'lower' : lower,
            'upper' : upper
        }
        self.add_edge(id, **edge_attributes)

        node_attributes = {
            'id' : id,
            'inputs' : [lower, upper],
            'outputs' : [id],
            'op' : 'index',
            'op_cat' : 'index',
            'edge_type': 'int'
        }
        self.add_node(id, **node_attributes)


    # Enter a parse tree produced by PMLangParser#prefix_expression.
    def enterPrefix_expression(self, ctx:PMLangParser.Prefix_expressionContext):
        self.scope.append(get_text(ctx))
        if ctx.IDENTIFIER():
            self.prefix_id = get_text(ctx.IDENTIFIER())
            self.prefix = self.prefix_id
        else:
            self.prefix = get_text(ctx)
        self.index_type.append('prefix')
        self.index = None


    # Exit a parse tree produced by PMLangParser#prefix_expression.
    def exitPrefix_expression(self, ctx:PMLangParser.Prefix_expressionContext):
        self.scope.pop()
        self.index_type.pop()

        # NEW
        if ctx.IDENTIFIER():
            var_name = ctx.IDENTIFIER().getText()
            src_node_id = self.get_input(ctx)
            edge_id = self.current_component.assignment_var(var_name, src_node_id, [])
        # NEW

        # Enter a parse tree produced by PMLangParser#assignment_expression.

    def enterAssignment_expression(self, ctx: PMLangParser.Assignment_expressionContext):
        self.prefix = None
        self.assignment = None
        self.is_literal = True
        self.inputs = deque()
        self.domain_strs = deque()

        if not ctx.predicate_expression():
            self.assignment = get_text(ctx.expression())

        # Exit a parse tree produced by PMLangParser#assignment_expression.

    # TODO: Add logic to specify def-use
    def exitAssignment_expression(self, ctx: PMLangParser.Assignment_expressionContext):
        if ctx.predicate_expression():
            op_cat = 'assign'
            op = 'predicate_assign'
            self.assignment = self.true_assign
            inputs = [self.predicate_expression, self.true_assign, self.false_assign]
            self.loop_stmt = lu.process_if_stmt(self.prefix, self.predicate_expression, self.true_assign, self.false_assign)
        elif self.is_literal:
            op_cat = 'assign'
            op = 'mov'
            inputs = [self.assignment]
        else:
            op_cat = 'assign'
            op = 'assign'
            inputs = [self.assignment]

        self.loop_stmt = lu.process_cdlang_stmt(self.loop_stmt)

        if self.prefix_id not in self.edges.keys():
            self.add_edge(self.prefix_id, copy_edge=self.assignment)
            self.update_edge(self.prefix_id, 'vcat', 'assign')
            self.update_edge(self.prefix_id, 'vid', self.prefix_id)
            if ctx.predicate_expression():
                self.update_edge(self.prefix_id, 'vtype', 'var')

            self.loop_stmt = lu.add_temporary(self.loop_stmt)

        self.instructions.append(self.loop_stmt)
        if self.prefix not in self.edges.keys():

            self.add_edge(self.prefix, copy_edge=self.prefix_id)
            self.update_edge(self.prefix, 'iid', self.prefix_index)
        self.s_input['outputs'].append(self.prefix)


        key = get_text(ctx)
        edge_type = self.edges[self.prefix]['type']
        node_attributes = {
            'id': key,
            'inputs': inputs,
            'outputs': [self.prefix],
            'op_cat': op_cat,
            'op': op,
            'edge_type' : edge_type
        }
        self.add_node(key, **node_attributes)

        if self.prefix_id in self.signature['output'] or \
                self.prefix_id in self.signature['state']:
            comp_node_attributes = {
                'id': self.prefix_id,
                'inputs': [self.prefix],
                'outputs': [self.prefix_id],
                'op_cat': 'read_write',
                'op': 'write',
                'edge_type': edge_type
            }
            self.add_node(self.prefix_id, **comp_node_attributes)
        self.inputs = None
        self.domain_strs = None
        self.is_predicate = False
        self.predicate_expression = None
        self.prefix_id = None
        self.prefix_index = None
        self.prefix = None


    # Enter a parse tree produced by PMLangParser#array_expression.
    def enterArray_expression(self, ctx:PMLangParser.Array_expressionContext):
        # NEW
        self.domain_strs.append([])
        # NEW
        self.scope.append(ctx.IDENTIFIER().getText())

    # Exit a parse tree produced by PMLangParser#array_expression.
    def exitArray_expression(self, ctx:PMLangParser.Array_expressionContext):
        id = self.scope.pop()
        key = get_text(ctx)
        index_type = self.index_type.pop()
        self.index_type.append(index_type)

        # NEW
        domains = self.domain_strs.pop()
        if index_type != 'group_expression':
            if index_type == 'prefix':
                src_node_id = self.get_input(ctx)
                edge_id = self.current_component.assignment_var(id, src_node_id, domains)
            else:
                edge_id = self.current_component.add_var_edge(id, domains)
                self.inputs.append(id)
        # NEW

        if index_type == 'prefix':
            self.prefix_id = id
            self.prefix_index = self.index
            self.index = None
        elif index_type == 'flow_declaration':
            if id in self.edges.keys():
                logger.warn("Redeclaration of flow: {}".format(ctx.getText()))
            self.s_input['outputs'].append(id)
            edge_attributes = {
                'type': self.type,
                'dimensions': self.indices,
                'vtype': 'var',
                'vcat': 'declaration',
                'vid': id,
                'iid': None,
                'default': None
            }
            self.add_edge(id, **edge_attributes)

            node_attributes = {
                'id': key,
                'inputs': self.indices,
                'outputs': [id],
                'op_cat': 'declaration',
                'op': 'edge',
                'edge_type' : self.type
            }
            self.add_node(key, **node_attributes)
            self.index = None
        elif index_type == 'expression':
            self.s_input['inputs'].append(id)
            self.s_input['inputs'].append(self.index)
            if id not in self.edges.keys():
                logger.critical("Error, index declared without variable declaration for"
                                 " {} and {}".format(id, key))
            if key not in self.edges.keys():
                self.add_edge(key, copy_edge=id)
                self.update_edge(key, 'iid', self.index)
                self.update_edge(key, 'src', [key])

                edge_type = self.edges[key]['type']
                node_attributes = {
                    'id': key,
                    'inputs': [id] + self.indices,
                    'outputs': [key],
                    'op_cat': 'offset',
                    'op': 'offset_array',
                    'edge_type': edge_type
                }
                self.add_node(key, **node_attributes)
        elif index_type == 'group_expression':
            self.fname.append(get_text(ctx.IDENTIFIER()))
            self.s_input['inputs'].append(self.index)
            self.group_index = self.index
            self.index = None
        else:
            logger.warning("Invalid index type or "
                            "not set {} for {}".format(index_type, key))



    # Enter a parse tree produced by PMLangParser#group_expression.
    def enterGroup_expression(self, ctx:PMLangParser.Group_expressionContext):
        self.index_type.append('group_expression')

    # Exit a parse tree produced by PMLangParser#group_expression.
    def exitGroup_expression(self, ctx:PMLangParser.Group_expressionContext):
        group_arg = get_text(ctx.expression())
        arr_expr = get_text(ctx.array_expression())
        id = get_text(ctx)
        # NEW
        out_node = self.current_component.add_group_node(self.get_input(ctx), id, self.domain_strs.pop())
        self.inputs.append(out_node)
        # NEW
        edge_attributes = {
                'type' : self.edges[group_arg]['type'],
                'dimensions' : [],
                'vtype' : 'var',
                'vcat' : 'expression',
                'vid' : id,
                'iid' : None,
                'default' : None
            }
        self.add_edge(id, **edge_attributes)
        fname = self.fname.pop()
        node_attributes = {
            'id' : id,
            'inputs': [group_arg, self.group_index],
            'outputs': [id],
            'op_cat': 'group',
            'op' : fname,
            'edge_type': self.edges[group_arg]['type'],

        }
        self.loop_stmt = lu.make_group_func(self.prefix, arr_expr, group_arg)
        self.add_node(id, **node_attributes)
        self.index_type.pop()
        self.index = None
        self.group_index = None


    # Enter a parse tree produced by PMLangParser#function_expression.
    def enterFunction_expression(self, ctx:PMLangParser.Function_expressionContext):
        self.fname.append(get_text(ctx.function_id()))
        self.expression_list = []

    # Exit a parse tree produced by PMLangParser#function_expression.
    def exitFunction_expression(self, ctx:PMLangParser.Function_expressionContext):
        id = get_text(ctx)
        fname = self.fname.pop()
        self.s_input['inputs'].append(fname)

        if fname in FUNCTIONS or fname in DATATYPE_SPECIFIERS:
            if not self.prefix:
                self.instructions.append(self.loop_stmt)
            # NEW

            if self.prefix:
                func_input = [self.get_input(ctx) for _ in self.expression_list]
                out_node = self.current_component.add_intermediate_node(func_input, fname)
                self.inputs.append(out_node)
            # NEW

            type, dims, vtype = self.function_output_inference(self.expression_list,id,fname)
            edge_attributes = {
                'type': type,
                'dimensions': dims,
                'vtype': vtype,
                'vid': id,
                'vcat': 'function',
                'iid': None,
                'default': None
            }
            self.add_edge(id, **edge_attributes)
            node_attributes = {
                'id': id,
                'inputs': self.expression_list,
                'outputs': [id],
                'op_cat': 'function',
                'op': fname,
                'edge_type': type,

            }
            self.add_node(id, **node_attributes)
        else:
            if self.prefix:
                logger.error(f"Use of unsupported function or assignment to component: {fname}.")
                exit(1)
            self.instructions.append(self.loop_stmt)
            node_attributes = {
                'id': id,
                'inputs': self.expression_list,
                'outputs': [],
                'op_cat': 'component',
                'op': fname,
                'component' : fname,
                'edge_type': 'component',
            }
            self.add_node(id, **node_attributes)

    # Enter a parse tree produced by PMLangParser#index_expression_list.
    def enterIndex_expression_list(self, ctx:PMLangParser.Index_expression_listContext):
        self.index_expression = True
        self.indices = []

    # Exit a parse tree produced by PMLangParser#index_expression.
    def exitIndex_expression(self, ctx:PMLangParser.Index_expressionContext):

        index_expression = get_text(ctx.expression())

        # NEW
        curr_domains = self.domain_strs.pop()
        curr_domains.append(index_expression)
        self.domain_strs.append(curr_domains)
        # NEW

        self.indices.append(index_expression)
        inputs = [index_expression]
        index_type = self.index_type.pop()
        self.index_type.append(index_type)

        if not self.index:
            self.prev_index = None
            self.index = index_expression
            prev_type = None
        elif not self.prev_index:
            self.prev_index = self.index
            self.index = '[' + self.index + '][' + index_expression + ']'
            prev_type = self.edges[self.prev_index]['vtype']
            inputs.insert(0,self.prev_index)
        else:
            self.prev_index = self.index
            self.index += '[' + index_expression + ']'
            if index_type != 'flow_declaration':
                prev_type = self.edges[self.prev_index]['vtype']
            else:
                prev_type = None
            inputs.insert(0,self.prev_index)


        new_type = self.edges[index_expression]['vtype']

        if self.index not in self.nodes.keys() and index_type != 'flow_declaration':
            if prev_type == 'index' or new_type == 'index':
                edge_attributes = {
                    'type': 'int',
                    'dimensions': self.indices,
                    'vtype': 'index',
                    'vcat' : 'expression',
                    'vid': None,
                    'iid': self.index,
                    'default': None
                }
                self.add_edge(self.index, **edge_attributes)
                node_attributes = {
                    'id': self.index,
                    'inputs': inputs,
                    'outputs': [self.index],
                    'op_cat': 'index',
                    'op': 'index',
                    'edge_type': 'int',

                }
                self.add_node(self.index, **node_attributes)
            else:
                edge_attributes = {
                    'type': 'int',
                    'dimensions': self.indices,
                    'vtype': 'var',
                    'vcat': 'expression',
                    'vid': self.index,
                    'iid': None,
                    'default': None
                }
                self.add_edge(self.index, **edge_attributes)
                node_attributes = {
                    'id': self.index,
                    'inputs': inputs,
                    'outputs': [self.index],
                    'op_cat': 'offset',
                    'op': 'offset_index',
                    'edge_type': 'int',

                }
                self.add_node(self.index, **node_attributes)

    # Exit a parse tree produced by PMLangParser#index_expression_list.
    def exitIndex_expression_list(self, ctx:PMLangParser.Index_expression_listContext):
        self.index_expression = False


    # Enter a parse tree produced by PMLangParser#expression_list.
    def enterExpression_list(self, ctx:PMLangParser.Expression_listContext):
        for exp in range(len(ctx.expression())):
            self.expression_list.append(get_text(ctx.expression()[exp]))

    # Enter a parse tree produced by PMLangParser#expression.
    def enterExpression(self, ctx:PMLangParser.ExpressionContext):
        self.scope.append(get_text(ctx))

        if ctx.unary_op():
            pass
        elif len(ctx.children) > 2:
            pass
        elif ctx.STRING_LITERAL():
            pass
        elif ctx.IDENTIFIER():
            pass
        elif ctx.array_expression():
            self.index_type.append('expression')

    # Exit a parse tree produced by PMLangParser#expression.
    def exitExpression(self, ctx:PMLangParser.ExpressionContext):
        id = self.scope.pop()
        if ctx.unary_op():
            self.is_literal = False
            op = STRING_TEXT_TO_UNEXP[get_text(ctx.unary_op())]
            operand = get_text(ctx.expression()[0])
            edge_attributes = {
                'type': self.edges[operand]['type'],
                'dimensions': self.edges[operand]['dimensions'],
                'vtype': self.edges[operand]['vtype'],
                'vcat': 'expression',
                'vid': id,
                'iid': self.edges[operand]['iid'],
                'default': None
            }
            self.add_edge(id, **edge_attributes)
            node_attributes = {
                'id': id,
                'inputs': [operand],
                'outputs': [id],
                'op_cat': 'unary',
                'op': op,
                'edge_type': self.edges[operand]['type']
            }
            self.add_node(id, **node_attributes)

            # NEW
            unary_input = [self.get_input(ctx)]
            out_node = self.current_component.add_intermediate_node(unary_input, op)
            self.inputs.append(out_node)
            # NEW
        elif len(ctx.children) > 2:
            self.is_literal = False
            op = STRING_TEXT_TO_BINEXP[get_text(ctx.getChild(1))]
            if op == 'exp':
                self.loop_stmt = lu.process_pow(self.loop_stmt)
            operand1 = get_text(ctx.expression()[0])
            operand2 = get_text(ctx.expression()[1])

            if operand1 not in self.edges.keys():
                logger.warning("line: {} - Use of undefined variable: {}".format(ctx.start.line, operand1))
            if operand2 not in self.edges.keys():
                logger.warning("line: {} - Use of undefined variable: {}".format(ctx.start.line, operand2))

            inferred_type, inferred_dims, inferred_vtype = self.output_inference([operand1, operand2], id)
            if inferred_vtype == 'index':
                edge_attributes = {
                    'type': inferred_type,
                    'dimensions': inferred_dims,
                    'vtype': inferred_vtype,
                    'vcat': 'expression',
                    'vid': None,
                    'iid': id,
                    'default': None
                }
                op_type = 'index'
            else:
                edge_attributes = {
                    'type': inferred_type,
                    'dimensions': inferred_dims,
                    'vtype': inferred_vtype,
                    'vcat': 'expression',
                    'vid': id,
                    'iid': None,
                    'default': None
                }
                op_type = 'temp'
            self.add_edge(id, **edge_attributes)

            node_attributes = {
                'id': id,
                'inputs': [operand1, operand2],
                'outputs': [id],
                'op_cat': 'binary',
                'op': op,
                'op_type' : op_type,
                'edge_type': inferred_type
            }
            self.add_node(id, **node_attributes)

            # NEW
            # print(f"Inputs: {self.inputs}"
            #       f"\nOp1: {operand1}\tOp2: {operand2}")
            binary_input = [self.get_input(ctx) for _ in range(2)]
            out_node = self.current_component.add_intermediate_node(binary_input, op)
            self.inputs.append(out_node)
            # NEW

        elif ctx.STRING_LITERAL():
            self.s_input['inputs'].append(id)
            edge_attributes = {
            'type' : 'str',
            'dimensions' : [],
            'vtype' : 'scalar',
            'vcat': 'literal',
            'vid' : id,
            'iid' : None,
            'default' : None,
            'value': id
        }
            self.add_edge(id, **edge_attributes)
            self.update_edge(id, 'src', [id])

            # NEW
            const_var = Variable(id, "str")
            const_var.set_type_modifier("param")
            const_var.add_default(id)
            self.current_component.add_var_symbol(const_var)
            _ = self.current_component.add_var_edge(id, [])
            self.inputs.append(id)
            # NEW

        elif ctx.IDENTIFIER():

            self.is_literal = False
            id = get_text(ctx.IDENTIFIER())
            self.s_input['inputs'].append(id)

            if id not in self.edges.keys():
                logger.warning("line: {} - Use of undefined variable: {}".format(ctx.start.line, id))

            # NEW
            _ = self.current_component.add_var_edge(id, [])
            self.inputs.append(id)
            # NEW
        elif ctx.array_expression():
            var_id = ctx.array_expression().IDENTIFIER().getText()
            self.index_type.pop()
            self.index = None



    # Enter a parse tree produced by PMLangParser#statement.
    def enterStatement(self, ctx:PMLangParser.StatementContext):
        self.statement = []
        self.s_input = {"name" : "statement" + str(len(self.statement_inputs)),
                        "inputs": [],
                       "outputs": []}
        self.snodes = []
        self.line = ctx.getText()
        self.loop_stmt = ctx.getText()
        self.index_type = deque()

        if ctx.assignment_statement():
            self.statement_type = 'assignment'
        elif ctx.expression_statement():
            self.statement_type = 'expression'
        elif ctx.declaration_statement():
            self.statement_type = 'declaration'

        self.s_input['statement_type'] = self.statement_type

    # Exit a parse tree produced by PMLangParser#statement.
    def exitStatement(self, ctx:PMLangParser.StatementContext):
        self.statements += self.statement
        self.s_input['nodes'] = self.snodes
        self.s_input['inputs'] = set(self.s_input['inputs'])
        self.s_input['outputs'] = set(self.s_input['outputs'])
        self.statement_inputs.append(self.s_input)
        self.statement_graphs.append(self.snodes)
        if ctx.assignment_statement():
            self.executions.append(get_text(ctx.assignment_statement().assignment_expression()))
        elif ctx.expression_statement():
            self.executions.append(get_text(ctx.expression_statement().expression()))

    # Enter a parse tree produced by PMLangParser#declaration_statement.
    def enterDeclaration_statement(self, ctx:PMLangParser.Declaration_statementContext):
        if ctx.FLOW():
            self.type = get_text(ctx.dtype_specifier())
        else:
            self.type = 'int'

    # Exit a parse tree produced by PMLangParser#declaration_statement.
    def exitDeclaration_statement(self, ctx:PMLangParser.Declaration_statementContext):
        self.type = None

    # Enter a parse tree produced by PMLangParser#predicate_expression.
    def enterPredicate_expression(self, ctx:PMLangParser.Predicate_expressionContext):
        self.predicate_expression = get_text(ctx.bool_expression())
        self.true_assign = get_text(ctx.true_expression())
        self.false_assign = get_text(ctx.false_expression())

    # Exit a parse tree produced by PMLangParser#predicate_expression.
    def exitPredicate_expression(self, ctx:PMLangParser.Predicate_expressionContext):
        pass

    # Enter a parse tree produced by PMLangParser#bool_expression.
    def enterBool_expression(self, ctx:PMLangParser.Bool_expressionContext):
        self.is_predicate = False

    # Exit a parse tree produced by PMLangParser#bool_expression.
    def exitBool_expression(self, ctx:PMLangParser.Bool_expressionContext):
        self.is_predicate = True


    # Enter a parse tree produced by PMLangParser#true_expression.
    def enterTrue_expression(self, ctx:PMLangParser.True_expressionContext):
        self.predicate = 't'
        self.is_predicate = True

    # Exit a parse tree produced by PMLangParser#true_expression.
    def exitTrue_expression(self, ctx:PMLangParser.True_expressionContext):
        if self.is_literal:
            op_cat = 'scalar_predicate_assign'
            op = 'mov'
            key = get_text(ctx)
            inputs = [self.prefix, key]
            node_attributes = {
                'id': key,
                'inputs': inputs,
                'outputs': [self.prefix],
                'op_cat': op_cat,
                'op': op,
                'edge_type' : self.edges[self.prefix]['type']
            }
            self.add_node(key, **node_attributes)
        self.predicate = None
        self.is_predicate = False


    # Enter a parse tree produced by PMLangParser#false_expression.
    def enterFalse_expression(self, ctx:PMLangParser.False_expressionContext):
        self.predicate = 'f'
        self.is_predicate = True

    # Exit a parse tree produced by PMLangParser#false_expression.
    def exitFalse_expression(self, ctx:PMLangParser.False_expressionContext):
        if self.is_literal:
            op_cat = 'scalar_predicate_assign'
            op = 'mov'
            key = get_text(ctx)
            inputs = [key]
            node_attributes = {
                'id': key,
                'inputs': inputs,
                'outputs': [self.prefix],
                'op_cat': op_cat,
                'op': op,
                'edge_type': self.edges[self.prefix]['type']
            }
            self.add_node(key, **node_attributes)
        self.predicate = None
        self.is_predicate = False

    def get_input(self, ctx):
        if len(self.inputs) == 0:
            raise RuntimeError(f"Unable to get input for {ctx.getText()}")
        else:
            return self.inputs.pop()

    # Exit a parse tree produced by PMLangParser#number.
    def exitNumber(self, ctx:PMLangParser.NumberContext):
        id  = get_text(ctx)
        self.s_input['inputs'].append(id)
        if ctx.integer():
            val = int(ctx.getText())
            val_str = ctx.getText()
            type = 'int'

        elif ctx.FLOAT_NUMBER():
            val = float(ctx.getText())
            val_str = ctx.getText()
            type = 'float'
        else:
            num = complex(ctx.getText().replace(" ", "").replace("i", "j"))
            real = num.real
            imag = num.imag
            val_str = [str(real), str(imag)]
            val = [real, imag]
            type = 'complex'

        edge_attributes = {
            'type' : type,
            'dimensions' : [],
            'vtype' : 'scalar',
            'vcat': 'literal',
            'vid' : get_text(ctx),
            'iid' : None,
            'default' : None,
            'value' : val
        }
        self.add_edge(id, **edge_attributes)
        self.update_edge(id, 'src', [id])

        # NEW
        const_var = Variable(id, type)
        const_var.set_type_modifier("param")
        const_var.add_default(val_str)
        self.current_component.add_var_symbol(const_var)
        _ = self.current_component.add_var_edge(id, [])
        self.inputs.append(id)
        # NEW

