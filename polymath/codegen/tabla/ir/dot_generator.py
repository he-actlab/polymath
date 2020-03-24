from .dataflow_graph import *
from collections import deque, OrderedDict
import copy
import json

class DotGenerator:

    def generate_dot(self, importedDFG, cycle2id):
        dfg = copy.copy(importedDFG)
        # for val in dfg.nodes:
            # print(val)
        fileName = "tabla" + ".dot"
        strList = []
        header = 'digraph G {' + '\n'
        footer = '}' + '\n'

        strList.append(header)

        # Setup for BFS
        queue = deque([dfg.get(0)])
        visitedList = set([])

        idDict = {}
        idDict[dfg.get(0)] = '"source"'
        idDict[dfg.get(1)] = '"sink"'

        # Perform BFS to connect nodes
        while len(queue) > 0:
            currNode = queue.popleft()

            # Connecting currNode with children
            left = idDict[currNode]
            for child in currNode.children:
                if child not in visitedList:
                    queue.append(child)
                    visitedList.add(child)

                # Child node doesn't have operation label
                if child not in idDict:
                    newLabel = self.getLabel(child)
                    idDict[child] = newLabel
                right = idDict[child]

                # flow is a line
                flow = str.format('{} -> {};\n', left, right)
                strList.append(flow)

            visitedList.add(currNode)

        # append rank
        srcNode = dfg.get(0)
        srcImmChildren = []
        for child in srcNode.children:
            srcImmChildren.append(child.id)
        rankCode = self.generateRankCode(cycle2id, srcImmChildren)
        strList.append(rankCode)

        strList.append(footer)

        dotCode = ''.join(strList)
        # print(dotCode)
        return dotCode


    nodecolor = {
        'model_input': 'skyblue',
        'model_output': 'hotpink',
        'model': 'yellow',
        'constant': 'gray',
        'gradient': 'green'
    }

    def getLabel(self, node):
        if node.dataType in DotGenerator.nodecolor:
            color = DotGenerator.nodecolor[node.dataType]
        else:
            color = None
        # data type color
        # if node.dataType == 'model_input':
        #     color = 'skyblue'
        # elif node.dataType == 'model_output':
        #     color = 'hotpink'
        # elif node.dataType == 'model':
        #     color = 'yellow'
        # elif node.dataType == 'gradient':
        #     color = 'green'
        # elif node.dataType == 'constant':
        #     color = 'gray'
        if color is None:
            return '{"' + str(node.id) + '" [label="' + node.operation + '"]' + '}'
        else:
            return '{"' + str(node.id) + '" [label="' + node.operation + '" style=filled fillcolor="' + color + '"]' + '}'


    def write_to(self, dotCode, path):
        # Write to file
        dotFile = open(path, 'w')
        dotFile.write(dotCode)
        dotFile.close()

    def generateRankCode(self, cycle2id, srcImmChildren):
        rankCode = ''
        rankSource = '{rank = source; "source";};\n'
        rankSink = '{rank = sink; "sink";};\n'

        rankCode += rankSource

        # srcImmChildren is a list of id's of source's immediate children
        '''
        sameRankIds = ''
        for id in srcImmChildren:
            sameRankIds += '"' + str(id) + '"' + '; '
        rankTempl = '{rank = same; ' + sameRankIds + '};\n'
        rankCode += rankTempl
        '''

        # cycle2id is a dictionary of cycle to node id list
        for cycle in cycle2id:
            rankTempl = '{rank = same; '
            idList = cycle2id[cycle]
            sameRankIds = ''
            for id in idList:
                sameRankIds += '"' + str(id) + '"' + '; '
            rankTempl += sameRankIds + '};\n'
            rankCode += rankTempl
        rankCode += rankSink

        #print(rankCode)
        return rankCode

    def read_sched(self, path):
        with open(path, 'r') as f:
            contentString = f.read()
        cycle2nodes = json.loads(contentString, object_pairs_hook=OrderedDict) # json.loads() gets the order of the keys messed up, so we're using OrderedDict
        cycle2id = OrderedDict()
        for cycle in cycle2nodes:
            nodeList = cycle2nodes[cycle]
            idList = []
            for node in nodeList:
                id = node['id']
                idList.append(id)
            cycle2id[cycle] = idList
        return cycle2id

# digraph{a [label="node"]; b [label="node"]; a->b}
# digraph{{a [label="node"]}->{b [label="node"]}};

# digraph G {
#   subgraph cluster1 {
#       main -> parse -> execute;
#       main -> init;
#   }
#     main -> cleanup;
#     execute -> make_string;
#     execute -> printf
#     init -> make_string;
#   subgraph sg1 {
#       main -> printf;
#       execute -> compare;
#   }

#   "src" -> "+1" [label="1"];
#   "src" -> "+1" [label="2"];

#   "src" -> "*1" [label="3"];
#       "+1" -> "*1";
#       "*1" -> "+2";


#   "src" -> "+2" [label="4"];

#   "+2" -> "sink";
# }

    '''
       These functions read a json file and returns the corresponding DFG object
    '''
    def readFrom(self, path):
        return self.load(path)

    def load(self, path):
        with open(path, 'r') as f:
            return self.fromStr(f.read())

    def fromStr(self, s):
        return self.fromList(json.loads(s))

    def fromList(self, l):
        dfg = DataFlowGraph()
        for nodeDict in l:
            node = DFGNode()
            node.from_dict(nodeDict)
            dfg.nodes.insert(node.id, node)

        for node in dfg.nodes:
            children = node.children
            for index,childId in enumerate(children):
                children[index] = dfg.get(childId)

            parents = node.parents
            for index,parentId in enumerate(parents):
                parents[index] = dfg.get(parentId)

        return dfg
