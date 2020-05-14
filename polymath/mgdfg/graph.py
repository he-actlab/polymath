from collections import OrderedDict
from indexed import IndexedOrderedDict
class Graph(dict):

    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(*args, **kwargs)

    def __hash__(self):
        return hash(tuple([hash(node) for _, node in self.items()]))

    def as_list(self):
        return [v for _, v in self.items()]

    def last(self):
        return self[next(reversed(self))]

    def item_by_index(self, key):
        return list(self.values())[key]

    def item_index(self, key):
        return list(self.keys()).index(key)

    def func_hash(self):
        return hash(tuple([(node.func_hash()) for _, node in self.items()]))

