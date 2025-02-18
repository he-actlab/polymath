import polymath as pm
from collections import OrderedDict
import numpy as np
from .tabla_utils import sigmoid_lut, gaussian_lut
import tqdm
import sys
from polymath.srdfg.util import is_iterable
LUT_NODES = {"sigmoid": sigmoid_lut,
             "gaussian": gaussian_lut}
TABLA_OP_MAP = {"add": "+",
                "mul": "*",
                "sub": "-",
                "gt": ">",
                "lt": "<",
                "sigmoid": "sigmoid",
                "sqrt": "sqrt"
                }

@pm.register_pass
class TablaPass(pm.Pass):

    def __init__(self, test_values=None, add_kwargs=False, debug=True, add_node_name=True):
        self.evaluations = 0
        self.dfg = OrderedDict()
        self.used = {}
        self.hsh_map = {}
        self.add_kwargs = add_kwargs
        self.add_node_name = add_node_name
        self.test_values = test_values or {}
        self.temp_map = {}
        self.norm_context = {}

        self.dfg["source"] = self.create_node("source", dfg_node_name="source")
        self.dfg["sink"] = self.create_node("sink", dfg_node_name="source")
        super(TablaPass, self).__init__(self.dfg, debug=debug)

    def apply_pass(self, node, ctx):

        if node.graph is None:
            return node

        n_key = self.node_key(node)
        if isinstance(node, pm.parameter):
            self.add_constants(node)
            assert n_key not in self.dfg
            self.set_dfg_node(node, self.create_node(_normalize_name(node.name), dtype="param", parents=[0], dfg_node_name=node))
            self.get_dfg_node("source")["children"].append(self.get_dfg_node(node)["id"])
        elif isinstance(node, pm.write):
            a0_key = self.node_key(node.args[0])

            if isinstance(node.args[2], (pm.temp, pm.output)):
                self.temp_map[node.args[2].name] = node.args[0]

            if a0_key not in self.dfg:
                assert not isinstance(node.args[0], pm.Node), f"Error, argument not found for write:{node.name} - {a0_key}\n" \
                                                              f"Keys: {list(self.dfg.keys())}"
                self.set_dfg_node(node.args[0], self.create_node(str(node.args[0]), dtype="constant", parents=[0], dfg_node_name=node.args[0]))
                self.get_dfg_node("source")["children"].append(self.get_dfg_node(node.args[0])["id"])
            self.get_dfg_node(node.args[0])["dataType"] = node.args[2].type_modifier
            self.get_dfg_node(node.args[0])["children"].append(1)
            self.get_dfg_node("sink")["parents"].append(self.get_dfg_node(node.args[0])["id"])
            self.set_used_node(node.args[0], "sink")
            return node
        elif isinstance(node, (pm.output, pm.temp)):
            return node
        elif isinstance(node, pm.placeholder):
            self.add_constants(node)
            assert n_key not in self.dfg
            self.set_dfg_node(node, self.create_node(_normalize_name(node.name), dtype=node.type_modifier, parents=[0], dfg_node_name=node))
            self.get_dfg_node("source")["children"].append(self.get_dfg_node(node)["id"])

        elif isinstance(node, pm.func_op):
            self.add_constants(node)
            a0_key = self.node_key(node.args[0])
            a1_key = self.node_key(node.args[1])

            if a0_key not in self.dfg:
                raise KeyError(f"Arg0 with key {a0_key} not found in dfg for func op node {node.name}\n"
                               f"Arg: {node.args[0].name} - {node.args[0].op_name}\n"
                               f"Args: {node.args}\n"
                               f"Graph ndoes: {node.graph.nodes.keys()}\n"
                               f"Keys: {self.dfg.keys()}")

            if a1_key not in self.dfg:
                raise KeyError(f"Arg1 with key {a1_key} not found in dfg for func op node {node.name}\n"
                               f"Args: {node.args}\n"
                                f"Graph ndoes: {node.graph.nodes.keys()}\n"
                               f"Keys: {self.dfg.keys()}")

            self.set_dfg_node(node, self.create_node(node.op_name, parents=[self.get_dfg_node(node.args[0])["id"], self.get_dfg_node(node.args[1])["id"]], dfg_node_name=node))
            self.get_dfg_node(node.args[0])["children"].append(self.get_dfg_node(node)["id"])
            self.get_dfg_node(node.args[1])["children"].append(self.get_dfg_node(node)["id"])
        elif isinstance(node, pm.NonLinear):
            self.add_constants(node)
            a0_key = self.node_key(node.args[0])
            if a0_key not in self.dfg:
                raise KeyError(f"Input value {a0_key} for NonLinear ({node.op_name}) node {node.name} not found in dfg\n"
                               f"Args: {node.args}\n"
                               f"Keys: {self.dfg.keys()}")
            self.set_dfg_node(node, self.create_node(node.op_name, parents=[self.get_dfg_node(node.args[0])["id"]], dfg_node_name=node))
            self.get_dfg_node(node.args[0])["children"].append(self.get_dfg_node(node)["id"])

        self.add_uses(node)

        return node

    def get_dfg_ids(self):
        return [n['id'] for key, n in self.dfg.items()]

    def finalize_pass(self, node, ctx):
        if node.graph is None:
            self.norm_context = {node.nodes[n]: v for n, v in self.test_values.items()}
            return node
        key = self.node_key(node)
        if key not in self.used and not isinstance(node, (pm.output, pm.state, pm.temp, pm.write)):
            if node.graph and node.name in node.graph.nodes:
                for a in node.args:
                    a_key = self.node_key(a)
                    if isinstance(a, pm.Node) and a.name in node.graph.nodes:
                        node.graph.nodes.pop(a.name)
                    self.used[a_key].remove(self.get_dfg_node(node)["id"])
                    if len(self.get_used_list(a)) == 0:
                        self.remove_node(a)
                self.remove_node(node)
                node.graph.nodes.pop(node.name)
        elif isinstance(node, (pm.output, pm.state, pm.temp)) and self.add_kwargs:
            self.add_dfg_params(node)
        elif isinstance(node, (pm.write)):
            node.graph.nodes[node.name] = node.args[0]
            if self.add_kwargs and isinstance(node.args[0], pm.Node):
                self.add_dfg_params(node.args[0])
        elif key not in self.used and node.graph:

            node.graph.nodes.pop(node.name)
            self.remove_node(node)
        elif self.add_kwargs:
            self.add_dfg_params(node)

        return node

    def add_dfg_params(self, node):
        node_info = self.get_dfg_node(node)
        node.add_attribute("children", node_info["children"])
        node.add_attribute("parents", node_info["parents"])

        node.add_attribute("tabla_dtype", node_info["dataType"])
        node.add_attribute("tabla_op", node_info["operation"])
        node.add_attribute("tabla_id", node_info["id"])

        if len(self.test_values.keys()) > 0:
            if node.name in self.test_values:
                node.add_attribute("computed", self.test_values[node.name])
                if is_iterable(self.test_values[node.name]):
                    if np.prod(node.shape) > 1:
                        raise RuntimeError(f"Cannot replace value with computed value for non-scalar value "
                                           f"{node} with value {self.test_values[node.name]}\n")

                    node_info["computed"] = int(self.test_values[node.name][0])
                else:
                    node_info["computed"] = int(self.test_values[node.name])
            elif node.value is not None:
                self.test_values[node.name] = node.value
                node_info["computed"] = int(node.value)
            else:
                assert all([not isinstance(v,str) for v in self.test_values.values()])
                ebefore = node.graph.evaluated_nodes
                if node.op_name in LUT_NODES:
                    input_val = self.test_values[node.args[0].name]
                    comp_res = LUT_NODES[node.op_name](input_val)
                else:
                    comp_res = node.evaluate(self.norm_context)
                self.norm_context[node] = comp_res
                ediff = node.graph.evaluated_nodes - ebefore
                self.evaluations += ediff
                self.test_values[node.name] = comp_res
                node_info["computed"] = int(comp_res)

    def add_constants(self, node):
        for a in node.args:
            if not isinstance(a, pm.Node):
                self.set_dfg_node(a, self.create_node(str(a), dtype="constant", parents=[0], dfg_node_name=a))
                self.get_dfg_node("source")["children"].append(self.get_dfg_node(a)["id"])

    def add_uses(self, node):
        for a in node.args:
            self.set_used_node(a, node)

    def create_node(self, operation, dtype=None, parents=None, dfg_node_name=None):

        parents = parents if isinstance(parents, list) else []

        node = {"id": len(self.dfg), "parents": parents, "dataType": dtype, "children": []}

        if dfg_node_name is not None:
            if isinstance(dfg_node_name, pm.Node):
                node['dfg_node_name'] = dfg_node_name.name
            else:
                node['dfg_node_name'] = dfg_node_name

        if dtype == "constant":
            node["computed"] = int(operation)
        if operation in TABLA_OP_MAP:
            node["operation"] = TABLA_OP_MAP[operation]
        else:
            node["operation"] = operation
        return node

    def add_node_child(self, parent_arg, child_node):
        p_key = self.node_key(parent_arg)
        if p_key not in self.dfg:
            assert np.isreal(parent_arg)
            self.set_dfg_node(parent_arg, self.create_node(str(parent_arg), dtype="constant", parents=[0], dfg_node_name=parent_arg))

        self.get_dfg_node(parent_arg)["children"].append(child_node["id"])

    def remove_node(self, node):
        nid = self.get_dfg_node(node)["id"]

        if nid == 1 or node in ["sink", "source"]:
            return
        else:
            for _, n in self.dfg.items():
                if nid in n["children"]:
                    n["children"].remove(nid)
            self.pop_node(node)

    def set_dfg_node(self, node, value):
        key = self.node_key(node)
        if key in self.dfg:
            prev = self.dfg[key]
            assert prev['operation'] == value['operation'] and prev['parents'] == value['parents']
        else:
            self.dfg[key] = value
            self.hsh_map[node] = node

    def set_used_node(self, a, node):
        id_v = self.get_dfg_node(node)["id"]
        a_key = self.node_key(a)

        if a_key not in self.used:
            self.used[a_key] = [id_v]
        else:
            self.get_used_list(a).append(id_v)

    def get_dfg_node(self, node):
        key = self.node_key(node)

        if key not in self.dfg:

            for k, v in self.hsh_map.items():
                if isinstance(v, pm.Node) and v.name == node.name:
                    self.hsh_map[node] = node
                    return self.dfg[k]
            raise ValueError(f"Could not find {key} in dfg for node {node.name} - {node.op_name}\n"
                             f"Node args: {node.args}.")
        else:

            return self.dfg[key]

    def get_used_list(self, node):
        key = self.node_key(node)
        return self.used[key]

    def pop_used(self, node):
        key = self.node_key(node)
        self.used.pop(key)

    def pop_node(self, node):
        key = self.node_key(node)
        if key not in self.dfg:
            for k,v in self.hsh_map.items():
                if isinstance(v, pm.Node) and v.name == node.name:
                    self.hsh_map[node] = node
                    self.dfg.pop(k)
                    return
            raise ValueError
        else:
            self.dfg.pop(key)

    def node_key(self, node):
        if isinstance(node, pm.Node) and node.name in self.temp_map:
            return self.node_key(self.temp_map[node.name])
        elif isinstance(node, pm.write):
            return self.node_key(node.args[0])
        else:
            return node.name if isinstance(node, pm.Node) else node

def _normalize_name(name):
    return name.rsplit("/", 1)[-1]