from polymath.pmlang.antlr_generator.parser import FileStream, CommonTokenStream, PMLangParser, ParseTreeWalker
from polymath.pmlang.symbols import PMLangListener
from polymath.pmlang.antlr_generator.lexer import PMLangLexer
import polymath.srdfg.serialization.mgdfgv2_pb2 as mgdfg
from typing import Dict, Tuple
import polymath.srdfg.template as temp
import os
import copy

def parse_file(file_path: str) -> PMLangListener:
    input_file = FileStream(file_path)
    lexer = PMLangLexer(input_file)
    stream = CommonTokenStream(lexer)
    parser = PMLangParser(stream)
    tree = parser.pmlang()
    pmlang_graph = PMLangListener(file_path)
    walker = ParseTreeWalker()
    walker.walk(pmlang_graph, tree)

    return pmlang_graph

def compile_to_pb(file_path: str, orig_listener=None):
    if orig_listener:
        listener = orig_listener
    else:
        listener = parse_file(file_path)

    output_dir, output_file = os.path.split(file_path)
    graph_name = output_file[:-3]

    program = mgdfg.Program(name=graph_name)
    for comp_name, comp in listener.components.items():
        program.templates[comp_name].CopyFrom(comp.serialize())
    new_graph = mgdfg_gen(listener.components)
    program.graph.CopyFrom(new_graph.serialize())
    return program


def store_pb(dir_path: str, program: mgdfg.Program):
    file_path = f"{dir_path}/{program.name}.pb"
    with open(file_path, "wb") as program_file:
        program_file.write(program.SerializeToString())

def load_pb(filepath: str) -> mgdfg.Program:
    new_program = mgdfg.Program()
    with open(filepath, "rb") as program_file:
        new_program.ParseFromString(program_file.read())
    return new_program

def mgdfg_gen(templates: Dict[str, temp.Template]) -> temp.Node:
    main_node = temp.Node(0, "main")
    main_node.instantiate([], templates["main"], templates)
    return main_node

def mgdfg_from_pb(file_path: str) -> Tuple[Dict[str,temp.Template], temp.Node]:
    program = load_pb(file_path)
    comp_dict = {}
    main_node = temp.Node(0, "main")
    main_node.deserialize(program.graph)
    for comp in program.templates:
        comp_dict[comp] = temp.Template(comp)
        comp_dict[comp].deserialize(program.templates[comp])

    return (comp_dict, main_node)

def create_component_with_name(comp_name: str) -> temp.Template:
    return temp.Template(comp_name)