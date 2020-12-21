from __future__ import absolute_import
import argparse
import os
from typing import Text
import sys

project_root = os.getcwd().rsplit("/", 1)[0]
sys.path.insert(0, project_root)

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
from polymath.srdfg.instructions.ir import AxelVM
from polymath.pmlang.antlr_generator.lexer import PMLangLexer
from polymath.pmlang.antlr_generator.parser import PMLangParser
from polymath.pmlang.symbols_old import PMLangListener
from polymath.srdfg.serialization import load_store
from polymath.srdfg import visualize
from polymath.srdfg.onnx_mgdfg.onnx_polymath import ONNXCMStack
from polymath.codegen.loopygen.generator import LoopyGen
from polymath.codegen.tabla.tabla_translate import TablaTranslation
from polymath.codegen.translator import Translator


def serialize_pmlang(pmlang_file, output_cmstack, viz=False):
    input_file = FileStream(pmlang_file)
    lexer = PMLangLexer(input_file)
    stream = CommonTokenStream(lexer)
    parser = PMLangParser(stream)
    tree = parser.pmlang()

    pmlang_graph = PMLangListener(pmlang_file, output_cmstack)
    walker = ParseTreeWalker()
    walker.walk(pmlang_graph, tree)
    output_dir, output_file = os.path.split(pmlang_file)

    outputfile = output_dir + '/' + output_file[:-3] + '.pb'

    load_store.save_program(pmlang_graph.program, outputfile)


def generate_axelvm(input_cmstack, output_axelvm, viz=False):
    avm = AxelVM(input_cmstack)
    avm.generate_axelvm()

def serialize_onnx(input_proto, output_cmstack, viz=False):
    converter = ONNXCMStack(input_proto)
    converter.run()

def visualize_graph(fname):
    visualize.visualize_program(fname, rankdir='TB')

def genccode(input_proto):
    code = LoopyGen(input_proto)

def gentabla(input_proto):
    code = TablaTranslation(input_proto)

def translate(input_proto, config):
    code = Translator(input_proto, config, ['map_nodes', 'flatten'])

def main():
    parser = argparse.ArgumentParser(description="PolyMath compilation framework")
    parser.add_argument("action",
                        type=Text,
                        help="One of the following: 'pmlang', 'onnx', or 'instructions' which generates a"
                             " serialized CMstack graph_name from either "
                             "a CMLang file or an ONNX protobuf file, or generates instructions "
                             "code from a CMStack file.",
                        choices=["pmlang", "onnx", "instructions", "visualize", "c", "tabla", "translate"])
    parser.add_argument("--input",
                        type=Text, required=True,
                        help="The input pmlang, onnx protobuf, or cmstack protobuf file")
    parser.add_argument("--tconfig",
                        type=Text, required=False,
                        help="The configuration file for translation")

    parser.add_argument("--output",
                            type=Text, required=False,
                            help="The output cmstack protobuf filename or "
                                 "instructions filename")

    args = parser.parse_args()

    if args.action == 'pmlang':
        serialize_pmlang(args.input, args.output)
    elif args.action == 'onnx':
        serialize_onnx(args.input, args.output)
    elif args.action == 'instructions':
        generate_axelvm(args.input, args.output)
    elif args.action == 'visualize':
        visualize_graph(args.input)
    elif args.action == 'c':
        genccode(args.input)
    elif args.action == 'tabla':
        gentabla(args.input)
    elif args.action == 'translate':
        translate(args.input, args.tconfig)


if __name__ == '__main__':
    main()
