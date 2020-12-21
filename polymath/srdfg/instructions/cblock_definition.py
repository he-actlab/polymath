import polymath.srdfg.serialization.srdfg_utils as interface
from polymath.srdfg.instructions import Instruction
import inspect
bin_exp = ["mul","div","add","sub","tlt","tgt","tlte","tgte","exp"]
un_exp = ["mov", "neg"]
function = ["pi","log","log2","float","int","bin","random","ceiling","floor","e","fread","fwrite"]
import logging
logger = logging.getLogger(__name__)

class CBLockDefinition:

    def __init__(self, template, commented=False, print_instr=False):
        self.temp_name = template.name
        self.nodes = template.sub_graph
        self.edges = template.edge_info
        self.start = ".cbegin " + template.name
        self.end = ".cend"
        self.ordered_args = interface.get_attribute_value(template.attributes['ordered_args'])
        self.inputs = list(template.input)
        self.outputs = list(template.output)
        self.states = list(template.state)
        self.params = list(template.parameters)

        self.temps = 0
        self.local_queues = {}
        self.indices = 0
        self.symbols = {}
        self.instructions = []
        self.header = []
        self.body = []
        self.footer = []

        self.instance_instructions = []

        self.commented = commented

        self.generate_header()
        self.generate_body()
        # if print_instr:
        #     self.print_code()

    def get_reg(self, id):
        vtype = interface.get_attribute_value(self.edges[id].attributes['vtype'])
        if vtype == 'scalar' and id not in self.symbols.keys():
            self.create_reg(id)
        return self.symbols[id]

    def get_iid_vid(self, id):
        vid = self.edges[id].vid
        iid = self.edges[id].iid

        return vid, iid

    def create_reg(self, id, queue=False, node_op=''):
        vtype = interface.get_attribute_value(self.edges[id].attributes['vtype'])

        if queue:
            assert id not in self.local_queues.keys(), "Error! Variable {} already created".format(id)
            self.local_queues[id] = '$c' + str(len(self.local_queues.keys()))
        else:
            current_frame = inspect.currentframe()
            callframe = inspect.getouterframes(current_frame, 2)
            # TODO: Test if result is index
            assert id not in self.symbols.keys(), "{}: Error! Variable {} already created".format(callframe[1][3], id)

            vid, iid = self.get_iid_vid(id)

            if vtype == 'index':
                assert iid, "Error! Index output with no index id for {}".format(id)
                self.symbols[id] = ('$i' + str(self.indices), '$i' + str(self.indices))
                self.indices += 1
            elif vtype == 'scalar':
                self.symbols[id] = (str(id), None)
            else:
                idx_reg = None
                t_reg = None
                if vid and vid not in self.symbols.keys():
                    t_reg = '$t' + str(self.temps)
                    self.temps += 1
                elif vid and vid in self.symbols.keys():
                    t_reg = self.symbols[vid][0]

                if iid and iid not in self.symbols.keys():
                    idx_reg = '$i' + str(self.indices)
                    self.indices += 1
                elif iid and iid in self.symbols.keys():
                    assert self.symbols[iid][1], "Error! Index symbol for {} with no index set for {}".format(id, iid)
                    idx_reg = self.symbols[iid][1]
                self.symbols[id] = (t_reg, idx_reg)



    def handle_bin_exp(self, node, instruction):
        inputs = node.input
        outputs = node.output
        # TODO: Check if arg has iterator and add
        self.create_reg(outputs[0], node_op=node.op_type)
        for arg in (list(outputs) + list(inputs)):
            op, op_iter = self.get_reg(arg)
            instruction.add_op(op, op_iter=op_iter)
        return instruction

    def handle_un_exp(self, node, instruction):
        inputs = node.input
        outputs = node.output
        # TODO: Check if arg has iterator and add
        self.create_reg(outputs[0], node_op=node.op_type)
        for arg in (list(outputs) + list(inputs)):
            op, op_iter = self.get_reg(arg)
            instruction.add_op(op, op_iter=op_iter)
        return instruction

    def handle_function(self, node, instruction):
        inputs = node.input
        outputs = node.output
        # TODO: Check if arg has iterator and add
        self.create_reg(outputs[0], node_op=node.op_type)
        print(node.input)
        print(node.output)
        for arg in (list(outputs) + list(inputs)):
            op, op_iter = self.get_reg(arg)
            instruction.add_op(op, op_iter=op_iter)
        return instruction
         ## TODO: Handle read results from assignment

    def handle_component(self, node, instruction):
        return instruction

    def handle_edge(self, node, instruction):
        pass

    def handle_index(self, node, instruction):
        inputs = node.input
        outputs = node.output
        # TODO: Check if arg has iterator and add
        self.create_reg(outputs[0], node_op=node.op_type)
        for arg in (list(outputs) + list(inputs)):
            op, op_iter = self.get_reg(arg)
            instruction.add_op(op, op_iter=op_iter)
        return instruction

    def handle_read_write(self, node, instruction):
        pass

    def handle_offset(self, node):

        inputs = node.input
        outputs = node.output
        if node.op_type == 'offset_array':
            self.create_reg(outputs[0], node_op=node.op_type)
        elif node.op_type == 'offset_index':
            print("inputs: {}, outputs:{}".format(inputs, outputs))
        # TODO: Check if arg has iterator and add


    def handle_assign(self, node, pred=None, pred_index=None,pred_reg=None):
        inputs = node.input
        outputs = node.output
        vid, iid = self.get_iid_vid(outputs[0])
        if node.op_type == 'mov':
            instruction = Instruction(node.op_type,
                                      pred=pred, pred_index=pred_index, pred_reg=pred_reg)
            self.create_reg(outputs[0], node_op=node.op_type)
            op, op_iter = self.get_reg(outputs[0])
            instruction.add_op(op, op_iter=op_iter)
        elif vid and vid not in self.symbols.keys():
            assert inputs[0] in self.symbols.keys(), "Assignment val {} not in symbols".format(inputs[0])
            self.symbols[vid] = (self.symbols[inputs[0]][0], None)
            if iid:
                self.symbols[outputs[0]] = (self.symbols[inputs[0]][0], self.symbols[iid][1])
            else:
                self.symbols[outputs[0]] = self.symbols[inputs[0]]

        if vid in self.outputs:
            write_instruction = Instruction('write')
            op, op_iter = self.get_reg(vid)
            write_instruction.add_op(op, op_iter=op_iter)
            dest = self.local_queues[vid]
            write_instruction.add_op(dest, dst=True)
            self.add_instruction('body', write_instruction, comment=vid)

    def handle_group(self, node, instruction):
        inputs = node.input
        outputs = node.output
        # TODO: Check if arg has iterator and add
        # print("Group args: input {}  output: {}".format(inputs,outputs))
        self.create_reg(outputs[0], node_op=node.op_type)
        for arg in (list(outputs) + list(inputs)):
            op, op_iter = self.get_reg(arg)
            instruction.add_op(op, op_iter=op_iter)
        return instruction

    def generate_body(self):
        ## TODO: Handle iteradd, itermul, etc for index nodes
        for n in self.nodes:
            if 'predicate' in n.attributes:
                pred_id = interface.get_attribute_value(n.attributes['predicate_bool'])
                pred_reg, pred_index = self.get_reg(pred_id)
                pred = interface.get_attribute_value(n.attributes['predicate'])
            else:
                pred_reg = None
                pred_index = None
                pred = None

            op_cat = n.op_cat
            if op_cat == 'binary':
                instruction = Instruction(n.op_type,
                                          pred=pred, pred_index=pred_index, pred_reg=pred_reg)
                instruction = self.handle_bin_exp(n, instruction)
                self.add_instruction('body', instruction, comment=n.name)
            elif op_cat == 'unary':
                instruction = Instruction(n.op_type,
                                          pred=pred, pred_index=pred_index, pred_reg=pred_reg)
                instruction = self.handle_un_exp(n, instruction)
                self.add_instruction('body', instruction, comment=n.name)
            elif op_cat == 'function':
                instruction = Instruction(n.op_type,
                                          pred=pred, pred_index=pred_index, pred_reg=pred_reg)
                instruction = self.handle_function(n, instruction)
                self.add_instruction('body', instruction, comment=n.name)
            elif op_cat == 'fdeclaration':
                instruction = Instruction('edge',
                                          pred=pred, pred_index=pred_index, pred_reg=pred_reg)
                instruction = self.handle_edge(n, instruction)
            elif op_cat == 'index':
                instruction = Instruction('index',
                                          pred=pred, pred_index=pred_index, pred_reg=pred_reg)
                instruction = self.handle_index(n, instruction)
                self.add_instruction('body', instruction, comment=n.name)
            elif op_cat == 'component':
                instruction = Instruction('wait',
                                          pred=pred, pred_index=pred_index, pred_reg=pred_reg)
                instruction = self.handle_component(n, instruction)
                # self.add_instruction('body', instruction, comment=n.name)

            elif op_cat =='read_write':
                instruction = Instruction(n.op_type,
                                          pred=pred, pred_index=pred_index, pred_reg=pred_reg)
                instruction = self.handle_read_write(n, instruction)
                print("READ WRITE: {}".format(n.op_type))
            elif op_cat == 'offset':
                self.handle_offset(n)
            elif op_cat == 'assign':
                self.handle_assign(n, pred=pred, pred_index=pred_index, pred_reg=pred_reg)
            elif op_cat == 'group':
                instruction = Instruction(n.op_type,
                                          pred=pred, pred_index=pred_index, pred_reg=pred_reg)
                instruction = self.handle_group(n, instruction)
                self.add_instruction('body', instruction, comment=n.name)
            elif op_cat == 'argument':
                # print("Argument: {}".format(n.op_type))
                pass
            elif op_cat == 'declaration':
                pass
            else:
                print("Unmatched op category: {}".format(op_cat))



    def generate_header(self):
        op = "read"
        read_queues = self.inputs + self.states + self.params

        for a in self.ordered_args:
            if a in read_queues:
                self.create_reg(a, queue=True, node_op='read queue')
                self.create_reg(a, queue=False, node_op='read queue')

                instruction = Instruction(op)
                instruction.add_op(self.symbols[a][0], dst=True)
                instruction.add_op(self.local_queues[a])
                self.add_instruction('header', instruction, comment=a)
            dims = interface.get_attribute_value(self.edges[a].attributes['dimensions'])
            for d in dims:
                if d not in self.symbols.keys():
                    self.create_reg(d, queue=True, node_op='read dimension')
                    self.create_reg(d, queue=False,node_op='read dimension' )

                    instruction = Instruction(op)
                    instruction.add_op(self.symbols[d][0], dst=True)
                    instruction.add_op(self.local_queues[d])
                    self.add_instruction('header', instruction, comment=d)

        for a in self.ordered_args:
            if a in read_queues:
                continue
            else:
                self.create_reg(a, queue=True, node_op='read queue')

    def add_instruction(self,part, instruction, comment=''):
        # TODO: Add typed instructions
        self.instructions.append(instruction)
        if part == 'body':
            if self.commented:
                self.body.append("{:<40s} ; {:<45s}".format(instruction.create_instruction(), comment))
            else:
                self.body.append(instruction.create_instruction())
        elif part == 'header':
            if self.commented:
                self.header.append("{:<40s} ; {:<45s}".format(instruction.create_instruction(), comment))

            else:
                self.header.append(instruction.create_instruction())
        elif part == 'footer':
            if self.commented:
                self.footer.append("{:<40s} ; {:<45s}".format(instruction.create_instruction(), comment))

            else:
                self.footer.append(instruction.create_instruction())
        elif part == 'instance_instr':
            if self.commented:
                self.instance_instructions.append("{:<40s} ; {:<45s}".format(instruction.create_instruction(), comment))
            else:
                self.footer.append(instruction.create_instruction())

    def print_code(self):

        print(self.start)

        for head in self.header:
            print(head)

        for body in self.body:
            print(body)

        for foot in self.footer:
            print(foot)

        print(self.end)
        print("\n")

