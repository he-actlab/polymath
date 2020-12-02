import polymath as pm
from polymath.mgdfg.base import _noop_callback

class Template(pm.Node):

    def __init__(self, *args, name=None, **kwargs):
        op_name = self.__class__.__name__
        super(Template, self).__init__(*args, op_name=op_name, name=name, **kwargs)
        self.flow_map = {}
        self.graph_map = {}
        self.initialize_args()

        if "graph" in kwargs:
            kwargs.pop("graph")

        if "dependencies" in kwargs:
            kwargs.pop("dependencies")

        if "dependencies" in kwargs:
            kwargs.pop("dependencies")

        with self:
           self.define_graph(*self.args, **kwargs)

        if self.graph:
            self.reset_graphs()

        for k, v in kwargs.items():
            if k not in self.kwargs:
                self.kwargs[k] = v

    def initialize_args(self):
        for a in self.args:
            if isinstance(a, pm.Node) and a.name not in self.nodes:
                # if not isinstance(a, (pm.placeholder, pm.parameter, pm.slice_op)):
                #     raise TypeError(f"Argument {a} for node {self.name} is invalid.")
                self.graph_map[a] = a.graph
                if isinstance(a, (pm.state, pm.output)):
                    self.graph_map[a.current_value()] = a.current_value().graph
                    a.current_value().graph = self
                    self.nodes[a.current_value().name] = a.current_value()
                a.graph = self
                self.nodes[a.name] = a
                for s in a.shape:
                    if isinstance(s, pm.Node) and s.name not in self.nodes:
                        self.graph_map[s] = s.graph
                        s.graph = self
                        self.nodes[s.name] = s

    def reset_graphs(self):
        for a in self.args:
            if isinstance(a, pm.Node) and a.name in self.nodes:
                if isinstance(a, (pm.state, pm.output)):
                    self.graph_map[a].nodes[a.current_value().name] = a.current_value()
                    a.current_value().graph = self.graph_map[a]
                a.graph = self.graph_map[a]
                for s in a.shape:
                    if isinstance(s, pm.Node) and s.name in self.nodes and s in self.graph_map:
                        s.graph = self.graph_map[s]

    def define_graph(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, context, callback=None):
        callback = callback or _noop_callback
        values = []

        with callback(self, context):
            for a in self.args:
                if isinstance(a, (pm.state, pm.output, pm.temp)):
                    write_name = self.get_write_name(a)
                    context[a] = self.nodes[write_name].evaluate(context)
                    values.append(context[a])
            assert len(values) > 0
        return values if len(values) > 1 else values[0]

    def get_write_name(self, n):
        if n.write_count > 0:
            name = []
            for i in n.name.split("/"):
                name.append(f"{i}{n.write_count - 1}")
            return "/".join(name)
        else:
            return n.name

