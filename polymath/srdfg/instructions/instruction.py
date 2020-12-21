import logging

logger = logging.getLogger(__name__)

class Instruction:
    type_map = {"int" : "i32",
                "float" : "float",
                "complex" : "cmp",
                "str": "str"}

    def __init__(self, operation,
                 instr_type=None, instance_name=None,
                 pred_reg=None,pred=None,pred_index=None,
                 dims=None):
        self.operation = operation
        self.instance_name = instance_name
        self.args = []
        self.pred_reg = pred_reg
        self.pred_index = pred_index
        self.pred = pred

        self.dims = dims
        self.type = instr_type

    def add_op(self, op, op_iter=None, dst=False):
        if op_iter and op != op_iter:
            op = op_iter + "(" + op + ")"

        if dst:
            self.args.insert(0, op)
        else:
            self.args.append(op)

    def get_reg_type(self, optype):
        if optype == 'index':
            return 'i'
        elif optype == 'scalar':
            return ''
        elif optype == 'queue':
            return 'c'
        else:
            return 't'

    def create_instruction(self):

        assert self.operation, "No operation found"
        # assert self.component_type, "No component type found"

        self.instr = "\t" + self.operation


        #TODO: Construct types
        if self.pred_reg:
            self.add_pred()

        if self.operation == 'wait':
            self.instr += ' ' + self.instance_name
        else:
            self.instr += ' '
            if self.type:
                self.construct_type()
            for a in self.args:
                self.instr += a + ','
            self.instr = self.instr[:-1]

        return self.instr

    def construct_type(self):

        if self.dims:
            self.instr += " [" + self.dims[0] + " x " + self.type_map[self.type] + "]"

            for i in range(1,len(self.dims)):
                self.instr = "[" + self.dims[i] + " x " + self.instr + "]"
        else:
            self.instr += " " + self.type_map[self.type]
        self.instr += " "


    def add_pred(self):

        self.instr += "_" + self.pred + "<"
        if self.pred_index:
            self.instr += self.pred_index + "(" + self.pred_reg + ")>"
        else:
            self.instr += self.pred_reg + ">"