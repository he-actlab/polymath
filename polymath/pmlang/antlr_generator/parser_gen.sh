#!/bin/bash

ANTLR="java -Xmx500M -cp '/usr/local/lib/antlr-4.7.1-complete.jar:$CLASSPATH' org.antlr.v4.Tool"
# Generate PMLang.tokens, PMLangLexer.interp, PMLangLexer.py, PMLangLexer.tokens, PMLangListener.py, and PMLangParser.py
$ANTLR -o ./ -listener -Dlanguage=Python3 -no-visitor -lib ./ ./PMLang.g4

# Change names of python files:
mv PMLangLexer.py lexer.py
mv PMLangParser.py parser.py
mv PMLangListener.py listener.py


