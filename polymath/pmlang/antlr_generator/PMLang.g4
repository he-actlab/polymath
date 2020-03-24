grammar PMLang;

pmlang
   : component_list
   ;
component_list
   : component_definition+
   ;


component_definition
   : IDENTIFIER '(' flow_list? ')' '{' statement_list '}'
   ;

flow_list
    : flow (',' flow)*
    ;

flow
    : flow_type dtype_specifier flow_expression
    ;

flow_expression
    : IDENTIFIER flow_index_list
    | IDENTIFIER
    | IDENTIFIER EQ literal
    ;
literal
    : STRING_LITERAL
    | number
    | complex_number
    ;
flow_index_list
    : flow_index+
    ;
flow_index
    : '[' (IDENTIFIER | DECIMAL_INTEGER) ']'
    ;
flow_declaration_list
    : flow_declaration (',' flow_declaration)*
    ;
flow_declaration
    : IDENTIFIER index_expression_list
    | IDENTIFIER
    ;

index_declaration_list
    : index_declaration (',' index_declaration)*
    ;

index_declaration
    : IDENTIFIER '[' expression ':' expression ']'
    ;
prefix_expression
    : IDENTIFIER index_value_list
    | IDENTIFIER
    ;

array_expression
    : IDENTIFIER index_expression_list
    ;

group_expression
    : GROUP_FUNCTION index_value_list '(' expression ')'
    ;

function_expression
    : function_id '(' expression_list? ')'
    ;

function_id
    : dtype_specifier
    | IDENTIFIER
    ;

nested_expression
    : '(' expression ')'
    ;

index_expression
    : '[' expression ']'
    ;
index_expression_list
    : index_expression+
    ;

index_value
    : '[' IDENTIFIER ']'
    | '[' DECIMAL_INTEGER ']'
    ;
index_value_list
    : index_value+
    ;

expression_list
    : expression (',' expression)*
    ;

expression
    : <assoc=right> IDENTIFIER index_expression_list
    | nested_expression
    | group_expression
    | function_expression
    | unary_op expression
    | <assoc=right> expression POW expression
    | expression multiplicative_op expression
    | expression additive_op expression
    | expression relational_op expression
    | number
    | STRING_LITERAL
    | IDENTIFIER
    ;

unary_op
    : '+'
    | '-'
    ;

multiplicative_op
   : '*'
   | '/'
   | '%'
   ;

additive_op
   : '+'
   | '-'
   ;

relational_op
   : '<'
   | '>'
   | LE_OP
   | GE_OP
   | EQ_OP
   | NE_OP
   ;

assignment_expression
   : <assoc=right> prefix_expression EQ  expression
   | <assoc=right> prefix_expression EQ predicate_expression
   ;

statement
   : assignment_statement
   | declaration_statement
   | expression_statement
   ;

statement_list
   : statement+
   ;

expression_statement
    : expression SEMI
    ;

declaration_statement
    : INDEX index_declaration_list SEMI
    | dtype_specifier flow_declaration_list SEMI
    ;
assignment_statement
    : assignment_expression SEMI
    ;

predicate_expression
   : bool_expression '?' true_expression ':'  false_expression
   ;
bool_expression
    : expression
    ;
true_expression
    : expression
    ;
false_expression
    : expression
    ;

iteration_statement
   : WHILE expression statement_list END SEMI
   | FOR IDENTIFIER EQ expression statement_list END SEMI
   | FOR '(' IDENTIFIER EQ expression ')' statement_list END SEMI
   ;

component_type
   : COMPONENT
   | SPRING
   | RESERVOIR
   ;
flow_type
    : INPUT
    | OUTPUT
    | STATE
    | PARAMETER
    ;
dtype_specifier
    : 'int'
    | 'float'
    | 'str'
    | 'bool'
    | 'complex'
    | 'fxp' DECIMAL_INTEGER '_' DECIMAL_INTEGER
    ;

INPUT
    : 'input'
    ;
OUTPUT
    : 'output'
    ;
STATE
    : 'state'
    ;
PARAMETER
    : 'param'
    ;

SPRING
    : 'spring'
    ;

RESERVOIR
    : 'reservoir'
    ;

COMPONENT
    : 'component'
    ;
INDEX
    : 'index'
    ;

FLOW
    : 'flow'
    ;

ARRAYMUL
    : '.*'
    ;


ARRAYDIV
    : '.\\'
    ;


ARRAYRDIV
    : './'
    ;


POW
    : '^'
    ;


BREAK
    : 'break'
    ;


RETURN
    : 'return'
    ;


FUNCTION
    : 'function'
    ;

GROUP_FUNCTION
    : 'sum'
    | 'prod'
    | 'argmax'
    | 'argmin'
    | 'min'
    | 'max'
    ;
FOR
    : 'for'
    ;


WHILE
    : 'while'
    ;


END
    : 'end'
    ;


GLOBAL
    : 'global'
    ;


IF
    : 'if'
    ;


CLEAR
    : 'clear'
    ;


ELSE
    : 'else'
    ;


ELSEIF
    : 'elseif'
    ;


LE_OP
    : '<='
    ;


GE_OP
    : '>='
    ;


EQ_OP
    : '=='
    ;

NE_OP
    : '!='
    ;


TRANSPOSE
    : 'transpose'
    ;


NCTRANSPOSE
    : '.\''
    ;

SEMI
    : ';'
    ;

STRING_LITERAL
    : '"' .*? '"'
    ;


IDENTIFIER
    : NONDIGIT (NONDIGIT | DIGIT)*
    ;


integer
    : DECIMAL_INTEGER
    | OCT_INTEGER
    | HEX_INTEGER
    | BIN_INTEGER
    ;

number
    : integer
    | FLOAT_NUMBER
    | IMAG_NUMBER
    ;

complex_number
    : (integer | FLOAT_NUMBER) additive_op IMAG_NUMBER
    ;

DECIMAL_INTEGER
    : NON_ZERO_DIGIT DIGIT*
    | '0'+
    ;

OCT_INTEGER
    : '0' [oO] OCT_DIGIT+
    ;

HEX_INTEGER
    : '0' [xX] HEX_DIGIT+
    ;

BIN_INTEGER
    : '0' [bB] BIN_DIGIT+
    ;

IMAG_NUMBER
    : ( FLOAT_NUMBER | INT_PART ) 'i'
    ;

FLOAT_NUMBER
    : POINT_FLOAT
    | EXPONENT_FLOAT
    ;

EQ
    : '='
    ;
WHITESPACE
    : [ \t]+ -> skip
    ;
NEWLINE
    : ('\r' '\n'? | '\n') -> skip
    ;
BLOCKCOMMENT
    : '/*' .*? '*/' -> skip
    ;
LINECOMMENT
    : '//' ~ [\r\n]* -> skip
    ;

fragment NONDIGIT
    : [a-zA-Z_]
    ;
fragment NON_ZERO_DIGIT
    : [1-9]
    ;

/// digit          ::=  "0"..."9"
fragment DIGIT
    : [0-9]
    ;

/// octdigit       ::=  "0"..."7"
fragment OCT_DIGIT
    : [0-7]
    ;

/// hexdigit       ::=  digit | "a"..."f" | "A"..."F"
fragment HEX_DIGIT
    : [0-9a-fA-F]
    ;

/// bindigit       ::=  "0" | "1"
fragment BIN_DIGIT
    : [01]
    ;

/// pointfloat    ::=  [intpart] fraction | intpart "."
fragment POINT_FLOAT
    : INT_PART? FRACTION
    | INT_PART '.'
    ;

/// exponentfloat ::=  (intpart | pointfloat) exponent
fragment EXPONENT_FLOAT
    : ( INT_PART | POINT_FLOAT ) EXPONENT
    ;

/// intpart       ::=  digit+
fragment INT_PART
    : DIGIT+
    ;

/// fraction      ::=  "." digit+
fragment FRACTION
    : '.' DIGIT+
    ;

/// exponent      ::=  ("e" | "E") ["+" | "-"] digit+
fragment EXPONENT
    : [eE] [+-]? DIGIT+
    ;

fragment SIGN
    : ('+' | '-')
    ;
