grammar Tabla;

/* scanner tokens */
MODEL_INPUT : 'model_input';
MODEL_OUTPUT : 'model_output';
MODEL : 'model';
GRADIENT : 'gradient';
ITERATOR : 'iterator';
ADD : '+';
SUB : '-';
LT : '<';
GT : '>';
MUL : '*';
PI : 'pi';
SUM : 'sum';
NORM : 'norm';
GAUSSIAN : 'gaussian';
SIGMOID : 'sigmoid';
SIG_SYM : 'sigmoid_symmetric';
LOG : 'log';
SEMI : ';';
COLON: ':';
LEFT_BRACK : '[';
RIGHT_BRACK : ']';
LEFT_PAREN : '(';
RIGHT_PAREN : ')';
COMMA : ',';
ASSIGN : '=';


/* var name */
ID
    : (LOWER | UPPER) (LOWER | UPPER | DIGIT | '_')* ('\'')?
    ;

fragment LOWER: 'a'..'z';
fragment UPPER: 'A'..'Z';
fragment DIGIT: '0'..'9';

WHITESPACE
    : (' ' | '\t' | '\n' | '\r')+ -> skip
    ;

COMMENT
//    : '/*' .*? '*/'
    : '//' .+? ('\n' | EOF) -> skip
    ;

INTLIT
    : '0'
    | '1'..'9' (DIGIT)*
    ;

/* LL(1) parser rules */
program 
    : data_decl_list stat_list EOF
    ;

data_decl_list 
    : data_decl*
    ;

data_decl 
    : data_type SEMI
    ;

data_type
    : non_iterator var_list
    | GRADIENT var_with_link_list
    | iterator var_list_iterator
    | ID ASSIGN INTLIT
    ;

non_iterator
    : MODEL_INPUT
    | MODEL_OUTPUT 
    | MODEL 
//    | GRADIENT 
    ;

iterator
    : ITERATOR
    ;

var_with_link_list
    : var '->' var var_with_link_list_tail
    ;

var_with_link_list_tail
    : ',' var_with_link_list
    | // elipse
    ;

var_list
    : var var_list_tail
    ;

var 
    : var_id
    ;

var_id 
    : ID id_tail
    ;

id_tail 
    : LEFT_BRACK (ID | INTLIT) RIGHT_BRACK id_tail
	| // epsilon
    ;

var_list_tail
    : COMMA var_list
    | // epsilon
    ;

var_list_iterator
    : ID LEFT_BRACK (ID | INTLIT) COLON (ID | INTLIT) RIGHT_BRACK
    ;

stat_list
    : stat*
	| // epsilon
    ;

stat
    : var ASSIGN expr SEMI
    ;

// expr_list : expr expr_list;
//	| ;

expr
    : term2 term2_tail
    ;

function
    : PI
	| SUM
    | NORM
	| GAUSSIAN
	| SIGMOID
	| SIG_SYM
	| LOG
    ;

function_args
    : LEFT_BRACK ID RIGHT_BRACK LEFT_PAREN expr RIGHT_PAREN
    | LEFT_PAREN expr RIGHT_PAREN
    ;

term2_tail
    : compare_op term2 term2_tail
	| // epsilon
    ;

term2
    : term1 term1_tail
    ;

term1_tail
    : add_op term1 term1_tail
	| // epsilon
    ;

term1
    : term0 term0_tail
    ;

term0_tail
    : mul_op term0 term0_tail
	| // epsilon
    ;

term0
    : var
    | LEFT_PAREN expr RIGHT_PAREN
    | INTLIT
    | function function_args
    ;

mul_op
    : MUL
    ;

add_op
    : ADD
    | SUB
    ;

compare_op
    : LT
    | GT
    ;
