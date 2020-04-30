import polymath as pm
from collections import OrderedDict
import numpy as np
from .tabla_utils import sigmoid_lut, gaussian_lut
import tqdm
import sys
LUT_NODES = {"sigmoid": sigmoid_lut,
             "gaussian": gaussian_lut}
TABLA_OP_MAP = {"add": "+",
                "mul": "*",
                "sub": "-",
                "gt": ">",
                "sigmoid": "sigmoid",
                }

@pm.register_pass
class TablaPass(pm.Pass):

    def __init__(self, test_values=None, add_kwargs=False, debug=True):
        self.evaluations = 0
        self.debug = debug
        self.dfg = OrderedDict()
        self.used = {}
        self.hsh_map = {}
        self.add_kwargs = add_kwargs
        self.test_values = test_values or {}
        self.dfg["source"] = self.create_node("source")
        self.dfg["sink"] = self.create_node("sink")
        self.pbar = tqdm.tqdm(desc="Applying first pass to nodes", file=sys.stdout, dynamic_ncols=True,)
        super(TablaPass, self).__init__(self.dfg)

    def apply_pass(self, node, ctx):

        if node.graph is None:
            return node
        if self.debug:
            if not self.pbar.total:
                self.pbar.reset(total=len(node.graph.nodes))
            self.pbar.update(1)
        n_key = self.node_key(node)
        if isinstance(node, pm.parameter):
            self.add_constants(node)
            assert n_key not in self.dfg
            self.set_dfg_node(node, self.create_node(_normalize_name(node.name), dtype="param", parents=[0]))
            self.get_dfg_node("source")["children"].append(self.get_dfg_node(node)["id"])
        elif isinstance(node, pm.write):
            a0_key = self.node_key(node.args[0])
            assert a0_key in self.dfg
            self.get_dfg_node(node.args[0])["dataType"] = node.args[2].type_modifier
            self.get_dfg_node(node.args[0])["children"].append(1)
            self.get_dfg_node("sink")["parents"].append(self.get_dfg_node(node.args[0])["id"])
            self.set_used_node(node.args[0], "sink")
            return node
        elif isinstance(node, pm.placeholder):
            self.add_constants(node)
            assert n_key not in self.dfg
            self.set_dfg_node(node, self.create_node(_normalize_name(node.name), dtype=node.type_modifier, parents=[0]))
            self.get_dfg_node("source")["children"].append(self.get_dfg_node(node)["id"])

        elif isinstance(node, pm.func_op):
            self.add_constants(node)
            a0_key = self.node_key(node.args[0])
            a1_key = self.node_key(node.args[1])
            assert a0_key in self.dfg
            assert a1_key in self.dfg
            self.set_dfg_node(node, self.create_node(node.op_name, parents=[self.get_dfg_node(node.args[0])["id"], self.get_dfg_node(node.args[1])["id"]]))
            self.get_dfg_node(node.args[0])["children"].append(self.get_dfg_node(node)["id"])
            self.get_dfg_node(node.args[1])["children"].append(self.get_dfg_node(node)["id"])
        elif isinstance(node, pm.NonLinear):
            self.add_constants(node)
            a0_key = self.node_key(node.args[0])
            assert a0_key in self.dfg
            self.set_dfg_node(node, self.create_node(node.op_name, parents=[self.get_dfg_node(node.args[0])["id"]]))
            self.get_dfg_node(node.args[0])["children"].append(self.get_dfg_node(node)["id"])

        self.add_uses(node)

        return node

    def finalize_pass(self, node, ctx):
        if node.graph is None:
            return node
        if self.debug:
            if self.pbar.n == self.pbar.total:
                self.pbar.reset(total=len(node.graph.nodes))
            self.pbar.set_description(f"Applying finalize pass to node {node.name} - {node.op_name}")
            self.pbar.update(1)

        key = self.node_key(node)

        if key not in self.used and not isinstance(node, (pm.output, pm.state, pm.write)):

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
        elif isinstance(node, (pm.output, pm.state)) and self.add_kwargs:
            self.add_dfg_params(node)
        elif isinstance(node, (pm.write)):
            node.graph.nodes[node.name] = node.args[0]
            if self.add_kwargs:
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
                node_info["computed"] = int(self.test_values[node.name])
            elif node.value is not None:
                self.test_values[node.name] = node.value
                node_info["computed"] = int(node.value)
            else:
                ctx_cpy = self.test_values.copy()
                assert all([not isinstance(v,str) for v in self.test_values.values()])
                ebefore = node.graph.evaluated_nodes
                if node.op_name in LUT_NODES:
                    input_val = self.test_values[node.args[0].name]
                    comp_res = LUT_NODES[node.op_name](input_val)
                else:
                    comp_res = node.graph(node, ctx_cpy)
                ediff = node.graph.evaluated_nodes - ebefore
                self.evaluations += ediff
                print(f"Total evaluations for {node.name}/{node.op_name}\t{ediff}\n"
                      f"Cumulative evaluations: {self.evaluations}\n\n\n")
                self.test_values[node.name] = comp_res
                node_info["computed"] = int(comp_res)

    def add_constants(self, node):
        for a in node.args:
            if not isinstance(a, pm.Node):
                self.set_dfg_node(a, self.create_node(str(a), dtype="constant", parents=[0]))
                self.get_dfg_node("source")["children"].append(self.get_dfg_node(a)["id"])

    def add_uses(self, node):

        for a in node.args:
            self.set_used_node(a, node)

    def create_node(self, operation, dtype=None, parents=None):
        parents = parents if isinstance(parents, list) else []
        node = {"id": len(self.dfg), "parents": parents, "dataType": dtype, "children": []}
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
            self.set_dfg_node(parent_arg, self.create_node(str(parent_arg), dtype="constant", parents=[0]))

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
            raise ValueError(f"Could not find {key} in dfg for node {node.name} - {node.op_name}")
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
        if isinstance(node, pm.write):
            return self.node_key(node.args[0])
        else:
            return node.name if isinstance(node, pm.Node) else node

def _normalize_name(name):
    return name.rsplit("/", 1)[-1]