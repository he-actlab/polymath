import json

class DFGNode:
    def __init__(self):
        self.id = None
        self.operation = None
        self.children = []    # order doesn't matter
        self.parents = []      # order matters
        self.dist2sink = None
        self.dataType = None    # e.g. model_input, gradient, etc.

    def to_dict(self):
        childrenIDs = []
        parentsIDs = []
        for child in self.children:
            childrenIDs.append(child.id)
        for parent in self.parents:
            parentsIDs.append(parent.id)
        return {'id': self.id, 'operation': self.operation, 'children': childrenIDs, 'parents': parentsIDs, 'dist2sink': self.dist2sink, 'dataType': self.dataType}
    
    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def insert_parent(self, int, val):
        self.parents.insert(int, val)

    def from_dict(self, d):
        self.id = d['id']
        self.operation = d['operation']
        self.children = d['children']
        self.parents = d['parents']
        self.dist2sink = d['dist2sink']
        self.dataType = d['dataType']
    
    def from_str(self, s):
        self.from_dict(json.loads(s))
    
    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.__str__())

    def load(self, path):
        with open(path, 'r') as f:
            self.from_str(f.read())

        
if __name__ == '__main__':
    src = DFGnode(0, None, [(1, '#1'), (1, 10)], None)
    n1 = DFGnode(1, '+', [(2, '#0'), (2, '#0')], [(1, '#1'), (1, 10)])
    n2 = DFGnode(2, '*', [3, '#2'], [(2, '#0'), (2, '#0')])
    sink = DFGnode(3, None, None, [3, '#2'])

    #src.save('./srcNode.json')
    node = DFGnode()
    node.load('./srcNode.json')
    node.save('./srcNodeCpy.json')
    

    '''
    print(src)
    print('-'*64)
    print(n1)
    print('-'*64)
    print(n2)
    print('-'*64)
    print(sink)
    '''

    '''    
    def __init__(self, package:Package, predecessor=None, successor=None, dist2sink=None, readyCycle=None):
        self.id = package.id
        self.operation = package.operation
        self.arguments = package.arguments
        self.output = package.output
        self.predecessor = predecessor
        self.successor = successor
        self.dist2sink = dist2sink
        self.readyCycle = readyCycle
    pass

    
    def to_dict(self):
        return {'id': self.id, 'operation': self.operation, 'arguments' :self.arguments, 'output': self.output, 'predecessor': self.predecessor, 'successor': self.successor, 'dist2sink': self.dist2sink, 'readyCycle': self.readyCycle }
    pass

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)
    pass

    def from_dict(self, d):
        package = Package(d['id'], d['operation'], d['arguments'], d['output'], d['predecessor'], d['successor'])
        self.__init__(package, d['predecessor'], d['successor'], d['dist2sink'], d['readyCycle'])
    pass

    def from_str(self, s):
        self.from_dict(json.loads(s))
    pass

if __name__ == '__main__':
    p1 = Package(0, '+', ['#1', 10], '#0', None, 1)
    print('-'*64)
    print(p1)
    print('-'*64)
    n1 = DFGnode(p1)
    print(n1)
    print('-'*64)
    '''
