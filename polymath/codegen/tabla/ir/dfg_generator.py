from .dataflow_graph import *
from . import func_table, group_ops
import pprint
import itertools
from collections import OrderedDict

class DFGGenerator:

    def __init__(self):
        self.expression = ''
        self.sym_table = {}
        self.iter_table = {}
        self.const_table = {}
        self.link_table = {}
        self.gradient_table = {}
        self.dfg = DataFlowGraph()

    def add_source_sink(self):

        self.source = DFGNode()
        self.source.operation = 'source'
        self.dfg.add(self.source)

        self.sink = DFGNode()
        self.sink.operation = 'sink'
        self.sink.dist2sink = 0
        self.dfg.add(self.sink)

    def gen_hash_tables(self):
        self.const_table = self.create_const_table()
        self.iter_table = self.create_iter_table()
        self.create_symbol_table()

    def gen_dfg(self):
        # Get stat_list for all Stats
        stat_list = self.parse_tree.stat_list() # type: TablaParser.Stat_listContext

        # Creation of DFG.
        for stat in stat_list.children:
            self.stat_traversal(stat)

    def add_sgd(self):
        # Append SGD
        for g in self.link_table:
            left_bound = g.find("[") + 1
            right_bound = g.rfind("]")
            iter_list = g[left_bound:right_bound].split('][')
            for i in range(len(iter_list)):
                if not iter_list[i].isdigit():
                    iter_list[i] = self.const_table[iter_list[i]]
            if len(iter_list) is 0:
                mult = DFGNode()
                mult.operation = "*"
                # mult.dataType = 'gradient'
                mult.dataType = None
                left_bound(mult)
                self.connect_node(self.sym_table[g], mult)
                self.connect_node(self.sym_table["mu"], mult)
                sub = DFGNode()
                sub.operation = "-"
                sub.dataType = 'model'
                self.dfg.add(sub)
                self.connect_node(mult, sub)
                self.connect_node(self.sym_table[self.link_table[g]], sub)
                self.sym_table[self.sym_table[self.link_table[g]]] = sub
            else:
                for i in range(iter_list[0]):
                    if len(iter_list) is 1:
                        gradient_sym = g[0:g.find('[')] + '[' + str(i) + ']'
                        if gradient_sym in self.sym_table:
                            mult = DFGNode()
                            mult.operation = "*"
                            mult.dataType = None
                            self.dfg.add(mult)
                            self.connect_node(self.sym_table[gradient_sym], mult)
                            self.connect_node(self.sym_table["mu"], mult)
                            sub = DFGNode()
                            sub.operation = "-"
                            sub.dataType = 'model'
                            self.dfg.add(sub)
                            w_sym = self.link_table[g] + '[' + str(i) + ']'
                            self.connect_node(mult, sub)
                            self.connect_node(self.sym_table[w_sym], sub)
                            self.sym_table[w_sym] = sub
                    else:
                        for j in range(iter_list[1]):
                            gradient_sym = g[0:g.find('[')] + '[' + str(i) + '][' + str(j) + ']'
                            if gradient_sym in self.sym_table:
                                mult = DFGNode()
                                mult.operation = "*"
                                # mult.dataType = 'gradient'
                                mult.dataType = None
                                self.dfg.add(mult)
                                self.connect_node(self.sym_table[gradient_sym], mult)
                                self.connect_node(self.sym_table["mu"], mult)
                                sub = DFGNode()
                                sub.operation = "-"
                                sub.dataType = 'model'
                                self.dfg.add(sub)
                                w_sym = self.link_table[g] + '[' + str(i) + '][' + str(j) + ']'
                                self.connect_node(mult, sub)
                                self.connect_node(self.sym_table[w_sym], sub)
                                self.sym_table[w_sym] = sub

    def finalize_graph(self):
        # Needs to connect correct outputs to the sink node
        for node in self.sym_table.values():  # Connect outputs
            if len(node.children) is 0 and len(node.parents) is not 1:
                self.connect_node(node, self.dfg.get(1))

        # Calculates all the distances to sink
        self.set_dist2sink(self.sink)

    def remove_nodes(self):
        # Remove useless nodes
        # self.dfg.update_id()
        removed_nodes = []
        for node in self.dfg.nodes:
            if node.dist2sink is None:
                for child in node.children:
                    child.parents.remove(node)
                for parent in node.parents:
                    parent.children.remove(node)
                removed_nodes.append(node)
        for node in removed_nodes:
            self.dfg.remove(node)

    def create(self, tree: TablaParser.ProgramContext):

        self.parse_tree = tree

        self.dfg = DataFlowGraph()

        # Create source and sink nodes first.
        self.add_source_sink()
        self.gen_hash_tables()
        self.gen_dfg()
        self.add_sgd()
        self.finalize_graph()
        self.remove_nodes()
        self.dfg.update_id()

        return self.dfg


    def write_to(self, dfg, path):
        dfg.save(path)

    def get_var_dims(self, var: TablaParser.Id_tailContext):
        dims = []

        while var.LEFT_BRACK() is not None:

            if var.ID() is not None:
                idx = var.ID().getText()
            else:
                idx = int(var.INTLIT().getText())
            dims.append(idx)

            var = var.id_tail()

        return dims

    def gather_iter_combinations(self, dims):

        dim_vals = []

        for dim in dims:
            if dim not in self.iter_table.keys():
                if isinstance(dim, int) or dim.isdigit():
                    val = int(dim)
                    dim_vals.append([{str(val): dim}])
                else:
                    print(f"Error, index {dim} not found in iter keys. Exiting.")
                    exit(1)
            else:

                low = self.iter_table[dim][0]
                high = self.iter_table[dim][1]
                dim_vals.append([{dim: i} for i in range(low, high)])

        combinations = list(itertools.product(*dim_vals))
        out_combos = []

        for c in combinations:
            iter_combos = OrderedDict()
            for t in c:
                iter_combos.update(t)
            out_combos.append(iter_combos)

        return out_combos


    def stat_traversal(self, stat: TablaParser.StatContext):
        self.expression = stat.getText()

        var = stat.var().var_id() # type: TablaParser.Var_idContext
        var_str = var.getText()

        id = var.ID().getText()

        dims = self.get_var_dims(var.id_tail())

        if var.id_tail().LEFT_BRACK() is None:
            result_node = self.expr_traversal(stat.expr(), {})
            # No check to see if this already exists in symbol table. Is that needed?
            self.sym_table[var_str] = result_node

        else:

            combos = self.gather_iter_combinations(dims)
            for c in combos:

                result_node = self.expr_traversal(stat.expr(), c.copy())
                if id in self.gradient_table and len(dims) > 1:
                    result_node.dataType = 'gradient'
                cvals = [str(i) for i in c.values()]
                key = id + '[' + ']['.join(cvals) + ']'

                self.sym_table[key] = result_node



    def expr_traversal(self, expr_node: TablaParser.ExprContext, iter_dict):

        if expr_node.term2_tail().children is not None:
            node = self.term2_tail_traversal(expr_node.term2_tail(), iter_dict)   # Go into term2Tail
            left_parent = self.term2_traversal(expr_node.term2(), iter_dict)
            self.connect_node(left_parent, node)
        else:
            node = self.term2_traversal(expr_node.term2(), iter_dict)

        return node


    def term2_tail_traversal(self, curr_term2_tail: TablaParser.Term2_tailContext, iter_dict):


        if curr_term2_tail.term2_tail().children is not None:
            right_parent = self.term2_tail_traversal(curr_term2_tail.term2_tail(), iter_dict)
            right_left_parent = self.term2_traversal(curr_term2_tail.term2(), iter_dict)
            self.connect_node(right_left_parent, right_parent)
        else:
            right_parent = self.term2_traversal(curr_term2_tail.term2(), iter_dict)

        node = self.create_node(curr_term2_tail.compare_op().getText(),right_parent)

        return node


    def term2_traversal(self, term2_node: TablaParser.Term2Context, iter_dict):

        if term2_node.term1_tail().children is not None:
            node = self.term1_tail_traversal(term2_node.term1_tail(), iter_dict)
            leftParent = self.term1_traversal(term2_node.term1(), iter_dict)
            self.connect_node(leftParent, node)
        else:
            node = self.term1_traversal(term2_node.getChild(0), iter_dict)

        return node


    def term1_tail_traversal(self, curr_term1_tail: TablaParser.Term1_tailContext, iter_dict):




        if curr_term1_tail.term1_tail().children is not None:
            right_parent = self.term1_tail_traversal(curr_term1_tail.term1_tail(), iter_dict)
            right_left_parent = self.term1_traversal(curr_term1_tail.term1(), iter_dict)
            self.connect_node(right_left_parent, right_parent)
        else:
            right_parent = self.term1_traversal(curr_term1_tail.getChild(1), iter_dict)

        node = self.create_node(curr_term1_tail.add_op().getText(), right_parent)

        return node


    def term1_traversal(self, term1_node: TablaParser.Term1Context, iter_dict):

        if term1_node.term0_tail().children is not None:
            node = self.term0_tail_traversal(term1_node.term0_tail(), iter_dict)
            left_parent = self.term0_traversal(term1_node.term0(), iter_dict)
            self.connect_node(left_parent, node)
        else:
            node = self.term0_traversal(term1_node.term0(), iter_dict)

        return node


    def term0_tail_traversal(self, curr_term0_tail:TablaParser.Term0_tailContext, iter_dict):
        # print('term0_tail_traversal ' + curr_term0_tail.getText())


        if curr_term0_tail.getChild(2).children is not None:
            right_parent = self.term0_tail_traversal(curr_term0_tail.term0_tail(), iter_dict)
            right_left_parent = self.term0_traversal(curr_term0_tail.term0(), iter_dict)
            self.connect_node(right_left_parent, right_parent)
        else:
            right_parent = self.term0_traversal(curr_term0_tail.getChild(1), iter_dict)

        node = self.create_node(curr_term0_tail.mul_op().getText(), right_parent)

        return node


    def term0_traversal(self, curr_term0:TablaParser.Term0Context , iter_dict):
        if curr_term0.expr() is not None:                                # expr child
            return self.expr_traversal(curr_term0.getChild(1), iter_dict)
        elif curr_term0.function() is not None:                              # func child
            return self.func_traversal(curr_term0, iter_dict)
        elif curr_term0.INTLIT() is not None:                 # INTLIT
            val = curr_term0.INTLIT().getText()
            # Create seperate node for numerical values that are connected to the src.
            if val in self.sym_table.keys() or int(val) in self.sym_table.keys():
                return self.sym_table[str(val)]
            else:
                node = self.create_node(val, self.dfg.get(0), dtype='constant', key=val)
                return node
        else:                                                           # ID
            var = curr_term0.var().var_id()

            key = var.ID().getText()

            dims = self.get_var_dims(var.id_tail())

            if len(dims) == 0:
                return self.sym_table[key]
            else:
                for it in dims:         # replace [i] with iteration value
                    key = key + '[' + str(iter_dict[it]) + ']'
                return self.sym_table[key]


    def func_traversal(self, curr_term0:TablaParser.Term0Context, iter_dict):

        func_parents = func_table[curr_term0.function().getText()]

        if func_parents == 1 :
            return self.func_single_parent_traversal(curr_term0, iter_dict)
        else:
            return self.func_multi_parent_traversal(curr_term0, iter_dict)


    def func_single_parent_traversal(self, curr_term0:TablaParser.Term0Context, iter_dict):

        parent = self.expr_traversal(curr_term0.function_args().expr(), iter_dict)
        node = self.create_node(curr_term0.function().getText(), parent)

        return node


    def func_multi_parent_traversal(self, curr_term0:TablaParser.Term0Context, iter_dict):
        func_args = curr_term0.function_args()
        func_range_iterator = func_args.ID().getText()

        func_iters = self.iter_table[func_range_iterator]
        funcvals = []
        func_dict = iter_dict
        # Calculates all expressions
        for i in range(func_iters[0], func_iters[1]):
            func_dict[func_range_iterator] = i
            funcvals.append(self.expr_traversal(func_args.expr(), func_dict))

        func_type = curr_term0.function().getText()
        operator = group_ops[func_type]

        if func_type == 'norm':      # We square the output expression and sum
            x = []
            for val in funcvals:
                node = self.create_node('*', val)
                self.connect_node(val, node)
                x.append(node)
            funcvals = x

        while len(funcvals) > 1:
            tmp_funcvals = []
            while len(funcvals) >= 2:
                left = funcvals.pop(0)
                right = funcvals.pop(0)
                node = self.create_node(operator, right)
                self.connect_node(left, node)
                tmp_funcvals.append(node)

            if len(funcvals) == 1:
                tmp_funcvals.append(funcvals.pop(0))
            funcvals = tmp_funcvals

        return funcvals.pop(0)


    def create_const_table(self):

        const_table = {}
        data_dec_list = self.parse_tree.data_decl_list() # type: TablaParser.Data_decl_listContext


        for dec in data_dec_list.getChildren(): # type: TablaParser.Data_declContext

            dtype = dec.data_type()
            
            if dtype.ASSIGN() is not None:
                x = int(dtype.INTLIT().getText())
                const_table[dtype.ID().getText()] = x

        return const_table


    def create_iter_table(self):
        iter_table = {}

        data_dec_list = self.parse_tree.data_decl_list() # type: TablaParser.Data_decl_listContext

        for dec in data_dec_list.getChildren():

            dtype = dec.data_type()

            if dtype.ASSIGN() is None and dtype.getChild(0).getText() == 'iterator':
                vlist_iter = dtype.getChild(1) # type: TablaParser.Var_list_iteratorContext
                key = vlist_iter.ID()[0].getText()

                lower = vlist_iter.getChild(2).getText()
                upper = vlist_iter.getChild(4).getText()

                if lower.isdigit():
                    lower = int(lower)
                else:
                    lower = self.const_table[lower]

                if upper.isdigit():
                    upper = int(upper)
                else:
                    upper = self.const_table[upper]

                iter_table[key] = (lower, upper)

        return iter_table

    def add_gradient(self, vlist):
        grad_var = vlist.var()[0].getText()
        model_var = vlist.var()[1].getText()
        self.link_table[grad_var] = model_var
        grad_key = vlist.var()[0].var_id().ID().getText()
        self.gradient_table[grad_key] = True

    def create_var_dims(self, dtype, id, dims: TablaParser.Id_tailContext):

        idx_vars = [id]

        while dims.LEFT_BRACK() is not None:

            if dims.ID() is not None:
                idx = self.const_table[dims.ID().getText()]
            else:
                idx = int(dims.INTLIT().getText())

            new_vars = []

            for i in idx_vars:
                for j in range(idx):
                    str_key = i + '[' + str(j) + ']'
                    new_vars.append(str_key)
                    _ = self.create_node(str_key, self.dfg.get(0), dtype, str_key)

            idx_vars = new_vars


            dims = dims.id_tail()


    def create_single_var(self, dtype, var: TablaParser.Var_idContext):

        id = var.ID().getText()
        if var.id_tail().LEFT_BRACK() is None:
            self.sym_table[id] = 0
            _ = self.create_node(id, self.dfg.get(0), dtype, id)
        else:
            self.create_var_dims(dtype, id, var.id_tail())

    def create_vars(self, dtype: TablaParser.Data_typeContext):

        vlist = dtype.var_list()  # type: TablaParser.Var_listContext
        vlist_tail = vlist.var_list_tail()  # type: TablaParser.Var_list_tailContext
        var = vlist.var().var_id()  # type: TablaParser.Var_idContext
        self.create_single_var(dtype, var)


        while vlist_tail.var_list() is not None:
            vlist = vlist_tail.var_list()  # type: TablaParser.Var_listContext
            vlist_tail = vlist.var_list_tail()
            var = vlist.var().var_id()  # type: TablaParser.Var_idContext
            self.create_single_var(dtype, var)



    def create_symbol_table(self):


        for const in self.const_table.keys():
            node = DFGNode()
            node.operation = const
            node.dataType = 'constant'
            self.dfg.add(node)
            self.connect_node(self.dfg.get(0), node)
            self.sym_table[const] = node

        data_dec_list = self.parse_tree.data_decl_list() # type: TablaParser.Data_decl_listContext

        for dec in data_dec_list.getChildren():
            dtype = dec.data_type()

            if dtype.GRADIENT() is not None:

                vlist = dtype.var_with_link_list()  # type: TablaParser.Var_with_link_listContext
                vlist_tail = vlist.var_with_link_list_tail()
                self.add_gradient(vlist)


                while vlist_tail.var_with_link_list() is not None:
                    vlist = vlist_tail.var_with_link_list()
                    vlist_tail = vlist.var_with_link_list_tail()
                    self.add_gradient(vlist)

            elif isinstance(dtype.getChild(0), TablaParser.Non_iteratorContext):
                self.create_vars(dtype)


    def create_node(self, op, src, dtype=None, key=None):
        node = DFGNode()
        node.operation = op

        if dtype:
            if isinstance(dtype, TablaParser.Data_typeContext):
                node.dataType = dtype.getChild(0).getText()
            elif isinstance(dtype, str):
                node.dataType = dtype
            else:
                print(f"Could not find valid datatype for {op}")

        self.dfg.add(node)
        self.connect_node(src, node)

        if key:
            self.sym_table[key] = node

        return node

    def connect_node(self, parent, child):
        child.parents.insert(0, parent)
        parent.children.append(child)

    def set_dist2sink(self, curr_node):
        for parent in curr_node.parents:
            if parent.dist2sink is None or parent.dist2sink < curr_node.dist2sink + 1:
                parent.dist2sink = curr_node.dist2sink + 1
            self.set_dist2sink(parent)
