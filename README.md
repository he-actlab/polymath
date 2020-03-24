# Welcome to PolyMath!

PolyMath is both a high-level language and and embedded Python language for compilation on heterogenous hardware.

### Installation instructions

To install PolyMath, change into the main directory of the repository and run the following command:

```bash
pip install --user 
```
### Dependencies  
If you would like to generate the lexer and parser, please refer to the "To generate lexer and parser directly" section below. Otherwise, you only need Python 3.4.3 or higher, in order to successfully run the compiler. If you would like to view the graphical representations of the compiler-generated dataflow graphs, Graphviz - graph visualization software - is needed. Please refer to the respective online resources in order to install them on your environment.  


### How to invoke the compiler   
To run the compiler, run the following command:

```
$ python3 main.py <*.t file>
```


This generates a JSON representation of data flow graph and schedule each in a separate file. It also creates a Dot file for a visual representation of data flow graph. Note that this reflects the graph after scheduling is done; every node in the same horizontal level is the operations scheduled to execute in the same cycle. Run the following command to generate a jpeg file:

```
$ dot -Tjpeg <*.dot file> -o <filename>.jpeg
```  

### To generate lexer and parser directly
*Dependencies* The parser is implemented with ANTLR v4.5 parser generator, with Python as the target language. You also need Java Runtime Environment 1.6 or higher in order to run the compiler, since ANTLR is primarily written in Java. 
  
Run the following command:

```
$ java -cp "/usr/local/lib/antlr-4.5-complete.jar:$CLASSPATH" org.antlr.v4.Tool -Dlanguage=Python3 Tabla.g
```

If you would like to see the lexer tokens, run:

```
$ python3 pygrun.py Tabla program --tokens TEST_FILE.t
```  

### Developers
This compiler was developed by Joon Kyung Kim and Chenkai Shao, both undergraduate students at Georgia Institute of Technology. For any inquiries, please contact *jkkim@gatech.edu* or *cshao31@gatech.edu*.




### DESIGN BUILDER

The design builder converts all the configuration provided by the compiler to customize the hardware template.

run: 
```
$ cd design-builder
$ python builder.py
```


### HARDWARE

The TABLA template design is a clustered hierarchical architecture constituting 
1. Processing Units (PUs)
2. Processing Engines (PEs).

This clustered template architecture is scalable, general, and highly customizable.

The code for the entire template, memory interface, FPGA wrapper is in fpga/hw-imp. 

### Directory Hierarchy:

fpga/hw-imp/source -> source Verilog files
fpga/hw-imp/source/mem_interface -> source files which have the memory interface Verilog files
fpga/hw-imp/source/ALU -> compute Verilog modules that perform the arithmetic functions
fpga/hw-imp/source/basic -> basic multiplexer and other files

To simulate this code. Change the top module in fpga/hw-imp/tb.list and run from fpga directory:
```
$ make test
```

