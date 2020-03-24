import json
from collections import OrderedDict
from compiler.backend import data_type_to_ns

import pprint
from . import data_priorities

class Node:
    def __init__(self, id=None, op=None, cycle=None, pe=None, pu=None):
        self.id = id
        self.op = op
        self.cycle = cycle
        self.pe = pe
        self.parents = []
        self.children = []
        self.parent_pes = []
        self.children_pes = []
        self.node_outputs = {}
        self.dest_pe_map = {}
        self.in_dtype = []
        self.out_dtype = []
        self.inst = []
        self.cycle_offset = 0

    def write_to(self, path):
        with open(path, 'w') as f:
            f.write(self.__str__())

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=False, indent=2)
    
    def to_dict(self):
        d = {}
        d["id"] = self.id
        d["op"] = self.op
        d["cycle"] = self.cycle
        d["in_dtype"] = self.in_dtype
        d["out_dtype"] = self.out_dtype
        d["parents"] = self.to_list(self.parents)
        d["children"] = self.to_list(self.children)
        if self.inst is not None:
            d["inst"] = [i.to_dict() for i in self.inst]
        else:
            d["inst"] = None
        if self.pe is not None:
            d["pe"] = self.pe.to_dict()
        else:
            d["pe"] = None
        return d

    def to_list(self, neighbor_nodes):
        neighbor_list = []
        for node in neighbor_nodes:
            if type(node) is int or type(node) is str:
                neighbor_list.append(node)
            else:
                d = {
                    "id": node.id,
                    "op": node.op,
                    "out_dtype": node.out_dtype
                }
                if node.pe is not None:
                    d["pe"] = node.pe.to_dict()
                else:
                    d["pe"] = None
                neighbor_list.append(d)
        return neighbor_list
    
    def inherit_parent_pe(self):

        if len(self.parents) == 1:
            self.pe = self.parents[0].pe
            return
        parent0 = self.parents[0]
        parent1 = self.parents[1]
        parent0_priority = data_priorities[parent0.out_dtype]
        parent1_priority = data_priorities[parent1.out_dtype]

        if parent0_priority == parent1_priority:
            if parent0.pe.id <= parent1.pe.id:
                self.pe = parent0.pe
            else:
                self.pe = parent1.pe
        else:
            if parent0_priority > parent1_priority:
                self.pe = parent0.pe                
            else:
                self.pe = parent1.pe
    @property
    def head_pe(self):
        return self.pe.head_pe

    @property
    def pu(self):
        return self.pe.pu

    def set_children_pes(self, nodes):    # can't find out until second pass
        for node in nodes:
            children = node.child
            for child_id in children:
                for child_node in nodes:
                    if child_id == child_node.id:
                        child_pe = child_node.pe
                        node.children_pes.append(child_pe)
    

class NodeGraph:
    def __init__(self, schedule_file,config, debug=False):
        self.nodes = {}
        self.size = 0
        self.pe_used = []
        self.num_pes = config['num_pes']
        self.pes_per_pu = config["pes_per_pu"]
        self.ns_size = config["namespace_size"]
        self.ns_int_size = config["namespace_interim_size"]
        self.read_from(schedule_file)
        self.generate_node_graph(debug=debug)

    def read_from(self, schedule_file):
        with open(schedule_file, 'r') as f:
            contents = f.read()
        f.close()
        self.schedule = json.loads(contents, object_pairs_hook=OrderedDict)

    def generate_node_graph(self, debug=False):
        for i, cycle in enumerate(self.schedule):
            nodes_in_cycle = self.schedule[cycle]
            for node in nodes_in_cycle:
                new_node = Node()
                new_node.id = node["id"]
                new_node.op = node["operation"]
                new_node.cycle = i
                new_node.out_dtype = node["dataType"]
                new_node.parents = node["parents"]  ## at this point, this is a list of id's
                new_node.children = node["children"]  ## also a list of id's

                if debug:
                    check_node_vals(new_node)
                self.add(new_node)

    def add(self, node):
        self.nodes[node.id] = node
        self.size += 1

    def get(self, node_id):
        if node_id in self.nodes:
            return self.nodes[node_id]
        else:
            return None

    def get_max_cycle(self):
        max_cycle = 0
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if node.cycle > max_cycle:
                max_cycle = node.cycle
        return max_cycle

    def assign_pe_initial(self, pu_pe):
        '''
        For each input (x's and y's) and weight data, this function assigns a PE.
        We call this "anchoring".
        This function also returns a list of PE id's that are used ("active")
        '''
        pe_list = pu_pe[1]
        if len(self.w_nodes) and len(self.xy_nodes):
            fill_ynodes = True
            i = 0
            for wnode in self.w_nodes:
                pe = self.fill_node(i, pe_list)
                wnode.pe = pe
                if i < len(self.xy_nodes):
                    self.xy_nodes[i].pe = pe
                else:
                    fill_ynodes = False
                i += 1
            if fill_ynodes:
                for ynode in self.xy_nodes[i:]:
                    ynode.pe = self.fill_node(i, pe_list)



    def fill_node(self, index, pe_list):
        pe = pe_list[index % self.num_pes]
        if pe.id not in self.pe_used:
            self.pe_used.append(pe.id)
        return pe


    def classify_initial_nodes(self):
        self.xy_nodes = []
        self.w_nodes = []
        self.c_nodes = []
        for node_id in self.nodes:
            node = self.get(node_id)
            if node.parents == ["Source"]:
                if node.out_dtype == "model_input" or node.out_dtype == "model_output":
                    self.xy_nodes.append(node)
                elif node.out_dtype == "model":
                    self.w_nodes.append(node)
                elif node.out_dtype == "constant":
                    self.c_nodes.append(node)

    def assign_constant_nodes(self):
        for constant_node in self.c_nodes:
            namespace_id = data_type_to_ns[constant_node.out_dtype]
            for child_node in constant_node.children:
                index = child_node.pe.namespace_map[namespace_id].tail
                child_node.pe.add_namespace(namespace_id, 0)

                constant_node.node_outputs[child_node.id] = { 'cycle' : 0,
                                                              'index' : index,
                                                              'namespace' : namespace_id,
                                                              'pe' : child_node.pe.id,
                                                              'comm_bit' : ''
                                                    }
    def assign_data_nodes(self):

        for xy_node in self.xy_nodes:
            namespace_id = data_type_to_ns[xy_node.out_dtype]
            index = xy_node.pe.namespace_map[namespace_id].tail
            xy_node.pe.add_namespace(namespace_id, 0)
            for child_node in xy_node.children:
                xy_node.node_outputs[child_node.id] = {'cycle': 0,
                                                       'index': index,
                                                       'namespace': namespace_id,
                                                       'pe': xy_node.pe.id,
                                                        'comm_bit': ''
                                                                 }
        for w_node in self.w_nodes:
            namespace_id = data_type_to_ns[w_node.out_dtype]
            index = w_node.pe.namespace_map[namespace_id].tail
            w_node.pe.add_namespace(namespace_id, 0)
            for child_node in w_node.children:
                w_node.node_outputs[child_node.id] = {'cycle': 0,
                                                        'index': index,
                                                        'namespace': namespace_id,
                                                        'pe': w_node.pe.id,
                                                        'comm_bit': ''
                                                    }

                if child_node.out_dtype == 'model':
                    child_node.node_outputs['Sink'] = {
                        'cycle': -1,
                        'index': index,
                        'namespace': namespace_id,
                        'pe': w_node.pe.id,
                        'comm_bit': ''
                    }

    def assign_source_data(self):
        self.assign_constant_nodes()
        self.assign_data_nodes()

    def assign_pes(self, pu_pe):

        self.classify_initial_nodes()
        self.assign_pe_initial(pu_pe)
        for cycle in self.schedule:
            if cycle == "cycle0":
                continue
            per_cycle = [node["id"] for node in self.schedule[cycle]]
            for node_id in per_cycle:

                node = self.get(node_id)

                node.inherit_parent_pe()


    def get_adjacent_nodes(self, node_ids, sink_or_source):
        nodes = []
        for node_id in node_ids:
            if self.get(node_id) is not None:
                node = self.get(node_id)
            else:
                node = sink_or_source
            nodes.append(node)
        return nodes

    def set_parents_and_children(self):
        """
        For each node, sets the parents to reference to the actual node object,
        instead of id number. Same for children.
        """
        for node_id in self.nodes:
            node = self.nodes[node_id]
            node.parents = self.get_adjacent_nodes(node.parents, "Source")
            node.children = self.get_adjacent_nodes(node.children, "Sink")

    def get_inst_list(self, node):
        inst_list = []
        if type(node.inst) == list:
            for cycle_offset, single_inst in enumerate(node.inst):
                node_c_temp = node.cycle + cycle_offset
                inst_list.append({"cycle": node_c_temp,
                                  "inst": single_inst})
        else:
            inst_list.append({"cycle": node.cycle, "inst": node.inst})

        return inst_list

    def separate_inst_by_pe(self):
        pe_to_inst = {} # pe-id to instruction list map
        max_cycle = self.get_max_cycle()
        for c in range(max_cycle):
            nodes = self.get_nodes_in_cycle(c + 1)
            for node in nodes:
                pe_id = node.pe.id
                if pe_id in pe_to_inst:
                    pe_to_inst[pe_id] += self.get_inst_list(node)
                else:
                    pe_to_inst[pe_id] = self.get_inst_list(node)
        return pe_to_inst

    def get_nodes_in_cycle(self, cycle):
        nodes = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if node.cycle == cycle:
                nodes.append(node)
        return nodes
        
    def write_to(self, path):
        with open(path, 'w') as f:
            f.write(self.__str__())
        
    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=False, indent=2)

    def to_dict(self):
        d = {}
        for node_id in self.nodes:
            node = self.nodes[node_id]
            d[node_id] = node.to_dict()
        return d


def check_node_vals(node:Node):
    print(node)



