from polymath.mgdfg.serialization import load_store
from polymath.mgdfg.instructions import CBLockDefinition

class AxelVM:

    def __init__(self, proto_file):
        self.program = load_store.load_program(proto_file)
        self.graph = self.program.graph
        self.cblocks = {}
        self.cblock_instances = {}
        self.generate_axelvm()

    def generate_axelvm(self):
        self.generate_cblock_definitions()

    def generate_cblock_definitions(self):
        for t in self.program.templates:
            block = CBLockDefinition(self.program.templates[t],
                                     commented=True, print_instr=True)
            instr = block.header + block.body + block.footer
            # instance_temp = block.
            self.cblocks[t] = instr

    def generate_cblock_instances(self):
        pass