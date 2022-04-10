import polymath as pm
from polymath.srdfg.base import _noop_callback
import inspect
# CONTEXT_TEMPLATE_TYPES = (pm.state, pm.output, pm.temp, pm.write)
CONTEXT_TEMPLATE_TYPES = (pm.state, pm.output, pm.temp)



class Template(pm.Node):

    def __init__(self, *args, **kwargs):
        #TODO: Add check for define graph signature to make sure it doesnt have certain keys
        # Also, need to make sure that popping keys off doesn't break things
        op_name = self.__class__.__name__
        if 'skip_definition' in kwargs and kwargs['skip_definition']:
            skip_definition = kwargs.pop('skip_definition')
        else:
            skip_definition = False
        super(Template, self).__init__(*args, op_name=op_name, **kwargs)
        self.flow_map = {}
        self.graph_map = {}
        temp_kwargs = {}

        if "graph" in kwargs:
            temp_kwargs["graph"] = kwargs.pop("graph")

        if "dependencies" in kwargs:
            temp_kwargs["dependencies"] = kwargs.pop("dependencies")

        if "name" in kwargs:
            temp_kwargs["name"] = kwargs.pop("name")

        self.args, self.kwargs = self.get_template_kwargs(*args, **kwargs)
        self.initialize_signature()
        # if not skip_definition:
        with self:
           self.define_graph(*self.args, **self.kwargs)

        if self.graph:
            self.reset_graphs()

    def initialize_arg(self, arg):
        if isinstance(arg, pm.Node) and arg.name not in self.nodes:
            # if not isinstance(a, (pm.placeholder, pm.parameter, pm.slice_op)):
            #     raise TypeError(f"Argument {a} for node {self.name} is invalid.")
            self.graph_map[arg] = arg.graph
            if isinstance(arg, CONTEXT_TEMPLATE_TYPES):
                self.graph_map[arg.current_value()] = arg.current_value().graph
                arg.current_value().graph = self
                self.nodes[arg.current_value().name] = arg.current_value()
            arg.graph = self
            self.nodes[arg.name] = arg
            for s in arg.shape:
                if isinstance(s, pm.Node) and s.name not in self.nodes:
                    self.graph_map[s] = s.graph
                    s.graph = self
                    self.nodes[s.name] = s


    def initialize_signature(self):
        for a in self.args:
            self.initialize_arg(a)

        for _, v in self.kwargs.items():
            self.initialize_arg(v)

    def reset_arg(self, arg):
        if isinstance(arg, pm.Node) and arg.name in self.nodes:
            if isinstance(arg, CONTEXT_TEMPLATE_TYPES):
                self.graph_map[arg].nodes[arg.current_value().name] = arg.current_value()
                arg.current_value().graph = self.graph_map[arg]
            arg.graph = self.graph_map[arg]
            for s in arg.shape:
                if isinstance(s, pm.Node) and s.name in self.nodes and s in self.graph_map:
                    s.graph = self.graph_map[s]

    def reset_graphs(self):
        for a in self.args:
            self.reset_arg(a)

        for _, v in self.kwargs.items():
            self.reset_arg(v)

    def define_graph(self, *args, **kwargs):
        raise NotImplementedError

    def get_template_kwargs(self, *args, **kwargs):
        sig = inspect.signature(self.define_graph)
        call_args = inspect.getcallargs(self.define_graph, *args, **kwargs)
        call_args.pop("self")
        # tcall_args = sig.bind(*args, **kwargs)

        new_args = []
        new_kwargs = {}
        kw_start_idx = len(sig.parameters.items())
        for vname, param in sig.parameters.items():
            if not param.default is inspect.Parameter.empty:
                kw_start_idx = list(sig.parameters).index(vname)
                break

        for idx, arg in enumerate(list(call_args.items())):

            if idx < kw_start_idx:
                new_args.append(arg[1])
            else:
                new_kwargs[arg[0]] = arg[1]
        return tuple(new_args), new_kwargs


    def evaluate_arg(self, values, context, arg):
        # if isinstance(arg, (pm.state, pm.output, pm.temp)):
        if isinstance(arg, CONTEXT_TEMPLATE_TYPES):
            write_name = self.get_write_name(arg)
            context[arg] = self.nodes[write_name].evaluate(context)
            values.append(context[arg])

    def evaluate(self, context, callback=None):
        callback = callback or _noop_callback
        values = []

        with callback(self, context):

            for a in self.args:
                self.evaluate_arg(values, context, a)

            for _, v in self.kwargs.items():
                self.evaluate_arg(values, context, v)

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

    @property
    def inputs(self):
        raise NotImplementedError(f"Inputs are not defined for {self.op_name}")

    @property
    def outputs(self):
        raise NotImplementedError(f"Outputs are not defined for {self.op_name}")

    @property
    def signature(self):
        sig = [self.op_name]
        ipt_shapes = []
        for i in self.inputs:
            # if not i.is_shape_finalized():
            #     raise RuntimeError(f"Cannot create signature for unitialized input variable {i.name} in {self.op_name}")
            ipt_shapes.append(len(i.shape))
        sig.append(tuple(ipt_shapes))
        opt_shapes = []
        for o in self.outputs:
            # if not o.is_shape_finalized():
            #     raise RuntimeError(f"Cannot create signature for unitialized output variable {o.name} in {self.op_name}")
            opt_shapes.append(len(o.shape))
        sig.append(tuple(opt_shapes))
        return tuple(sig)