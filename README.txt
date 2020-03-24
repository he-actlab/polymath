TABLA Language Spec
===

Grammar 
    The grammar has been written for an LL(1) parser to understand. Grammar rules can be found in Tabla.g file.
    ### NOTE: In variable declaration, only int literal assignment is allowed (for now) ###
    Every statement has to end with a semicolon(;). 
    Variables have to be declared before being used.

    There are five data types supported in TABLA: 
        model_input
	model_output
	model
	gradient
	iterator
    Variables of these data types must be declared. Multiple variables of the same data type can be decalred in the same line, even if they have different dimensions. For example, the following is legal:
        model_input i[x], j[y][z];
    On the other hand, integer data types are not declared. In other words, if a declared variable does not have a data type declared with it, it is assumbed to be integer data type. For example, 
        m = 10;
    This is a valid statement, even though the variable m does not have a data type associated with it explicitly. 
    The following code snippet is legal:
        m = 15;
	model_input x[m];
    However, the following is not legal, since n is not declared:
        model_output y[n];    

    Iterator data type has a special syntax associated with its variables. A variable name is immediately followed by a left bracket, a starting point and an end point, delimited by a colon, and a right bracket. In other words,
        (data type) (variable name)(left bracket)(digit or an integer variable)(colon)(digit or an integer variable)(right bracket)
    Using a token notation,
        ITERATOR ID LEFT_BRACK (ID | INTLIT) COLON (ID | INTLIT) RIGHT_BRACK SEMI
    This is because iterator data type serves the same functionality as a for loop. The range of values to be looped is expressed inside the brackets. These are integer values incrementing by 1. Either raw values or variables containing an integer value (or both) can be used for this. 
    Here are examples of valid iterator declartion:
	iterator i[0:10]; // all iterator arguments as integer literals
	iterator j[m:n]; // all iterator arguments as integer variables (that should havebeen decalred before this statment)
	iterator k[m:10]; // one iterator argument as an integer variable, the other as an integer literal
	iterator l[0:n]; // same as before, but the other way around
    The following is illegal, since it does not give the range of values to be looped:
        iterator x;

Lexical Rules
    Variable names follow the similar rules as the ones in C. The properties are:
    a. Variable names start with either an upper case letter or an upper case letter, followed by an arbitrary length of alphanumeric characters including underscore (_), and it can end with a single quote ('). 
    b. Variables can be of any dimension. For example, the following are all legal:
        a b[m] c[x][y] d[i][j][k]

    Comments begin with // and the rest of the line is ignored.

Operator Precedence
    The basic operators follow the following precedence from highest to lowest:
    1. (), []
    2. *
    3. +, -
    4. <, >
    5. =

Functions
    Aside from the basic operators, there are two types of operations: group and non linear. In group operations, pi and sum operates on two arguments, whereas norm operates on one argument. However, even though pi and sum operates on two arguments, this is only in a semantic manner. Syntatically, it appears they take in one. In other words, in between the parentheses, pi and sum operators do not require an argument followed by a comma and then another argument, as one would normally expect from other languages. For example, if one would write a sum function in C, it would look like: 
        sum(2, 3);
    However, in TABLA, it would look something like this:
        sum[i](x[i] * w[j][i]);
    where i and j are iterators. Notice there is no comma (,) inside the parentheses. 
    Also, since pi and sum are operated group-wise, they require an iterator. This is wrapped inside square brackets, as shown above in the sum operator. Syntatically, sum and pi operators come in the following format:
        (SUM | PI) LEFT_BRACK ID RIGHT_BRACK LEFT_PAREN expr RIGHT_PAREN SEMI
    whereas the rest of the operators are expressed in the following format:
        (NORM | GAUSSIAN | SIGMOID | SIG_SYM | LOG) LEFT_PAREN expr RIGHT_PAREN SEMI

    For now, functions can't be mixed with other basic operators in the same line, but this can be fixed if needed. 
    The above functions are recognized as language tokens by the parser. 


Parser Implementation
===

Tools used
    ANTLR parser generator was used to automatically generate the language parser. ANTLR version 4.5 was used, with Python as the target parser language. Python version used is 2.7.6.

To generate lexer and parser:
    java -cp "/usr/local/lib/antlr-4.5-complete.jar:$CLASSPATH" org.antlr.v4.Tool -Dlanguage=Python3 Tabla.g

To see tokens:
    python3 pygrun.py Tabla program --tokens TEST_FILE
