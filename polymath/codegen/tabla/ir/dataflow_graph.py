import json
from .dfg_node import DFGNode

class DataFlowGraph:

    def __init__(self):
        self.nodes = []

    def __copy__(self):
        copy = DataFlowGraph()
        copy.nodes = self.nodes.copy()
        return copy

    def add(self, node:DFGNode):
        self.nodes.append(node)

    def get(self, index):
        return self.nodes[index]

    def remove(self, node:DFGNode):
        self.nodes.remove(node)

    def isEmpty(self):
        return len(self.nodes) == 0

    def update_id(self):
        for i in range(len(self.nodes)):
            self.nodes[i].id = i

    '''
    def to_dict(self):
        d = {}
        for node in self.nodes:
            id = node.id
            d[id] = node.to_dict()

        for key in self.nodes:
            dfgNode = self.nodes[key]
            d[key] = dfgNode.to_dict()

        return d
    '''

    def to_list(self):
        list = []
        for node in self.nodes:
            list.append(node.to_dict())
        return list

    def __str__(self):
        return json.dumps(self.to_list(), sort_keys=True, indent=2)

    '''
    def from_dict(self, d):

        dfg = self.__init__()
        for nodeKey in d.keys():
            node = d[nodeKey]
            newNode = DFGnode()
            newNode.from_dict(node)
            dfg.add(newNode)

        pass
    '''

    def from_list(self, l):
        for nodeDict in l:
            node = DFGNode()
            node.from_dict(nodeDict)
            self.add(node)

    def fromStr(self, s):
        self.from_list(json.loads(s))

    def getSize(self):
        return len(self.nodes)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.__str__())

    def load(self, path):
        with open(path, 'r') as f:
            self.fromStr(f.read())


if __name__ == '__main__':
    '''
    src = DFGnode(0, None, [(1, '#1'), (1, 10)], None)
    n1 = DFGnode(1, '+', [(2, '#0'), (2, '#0')], [(1, '#1'), (1, 10)])
    n2 = DFGnode(2, '*', [3, '#2'], [(2, '#0'), (2, '#0')])
    sink = DFGnode(3, None, None, [3, '#2'])
    '''
    #nodes = [src, n1, n2, sink]
    #g = DataFlowGraph()
    '''
    g.add(src)
    g.add(n1)
    g.add(n2)
    g.add(sink)
    '''
    #g.save('./dfg.json')

    g = DataFlowGraph()
    g.load('./dfg.json')
    #srcNode = g.nodes[0]
    #print(srcNode.id)

    '''
    n3 = DFGnode(2, '*', [3, '#2'], [(2, '#0'), (2, '#0')])
    g.add(n3)
    '''
    print(g.getSize())
    for node in g.nodes:
        print(type(node))
        print(node)
    print(type(g))
    print(type(g.nodes[0]))


    '''
    print('\n#0 = #1 + 10')
    p1 = Package(0, '+', ['#1', 10], '#0', None, 1)
    print('-'*32 + 'package0' + '-'*32)
    print(p1)
    print('-'*32 + 'node0' + '-'*32)
    n1 = DFGnode(p1)
    print(n1)
    print('-'*64)

    print('\n#2 = #0 * #0')
    p2 = Package(1, '*', ['#0', '#0'], '#2', 0, None)
    print('-'*32 + 'package1' + '-'*32)
    print(p2)
    print('-'*32 + 'node1' + '-'*32)
    n2 = DFGnode(p2)
    print(n2)
    print('-'*64)

    g = DataFlowGraph()
    g.add(n1)
    g.add(n2)
    print('\n' + '-'*32 + 'DFG' + '-'*32)
    print(g)
    print('-'*64)
    '''
