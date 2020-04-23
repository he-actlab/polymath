from collections import OrderedDict
from indexed import IndexedOrderedDict
class Graph(IndexedOrderedDict):

    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(*args, **kwargs)

    def __hash__(self):
        # return hash(tuple([(node.func_hash()) for _, node in self.items()]))
        return hash(tuple([hash(node) for _, node in self.items()]))

    def as_list(self):
        return [v for _, v in self.items()]

    def last(self):
        return self[next(reversed(self))]

    def item_by_index(self, key):
        return self.values()[key]

    def item_index(self, key):
        return list(self.keys()).index(key)

    def replace_item(self, name, key, value):
        name_idx = self.item_by_index(name) - 1
        self.pop(name)
        self[key] = value
        for ii, k in enumerate(list(self.keys())):
            if ii > name_idx and k != key:
                self.move_to_end(k)

    def insert_after(self, index, key, value):
        assert index in self.keys()
        if key in self.keys():
            self.pop(key)
        idx = list(self.keys()).index(index)
        self[key] = value
        for ii, k in enumerate(list(self.keys())):
            if ii > idx and k != key:
                self.move_to_end(k)