#define _GNU_SOURCE
#include "pipe.h"
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

// Define values
#define THREADS 4
#define STRINGSIZE 1024
#define MAXCOLSIZE 2048

// Type definitions for pipes
typedef pipe_t* flow;
typedef pipe_producer_t* output;
typedef pipe_consumer_t* input;
typedef char string[STRINGSIZE];
typedef char* str;
// typedef char * string;

// Macros for queues
#define FLOW(handle) ((pipe_t*)(handle))
#define component void
#define OUTPUT_QUEUE(x) pipe_producer_new(x)
#define INPUT_QUEUE(x) pipe_consumer_new(x)
#define QUEUE(x) pipe_new(x, 0)

#define WRITE(x,data) pipe_push(x, data, 1)
#define READ(x,data) pipe_pop(x, data, 1)
#define READB(x,data, bsize) pipe_pop(x, data, bsize)

#define FREE_OUTPUT_QUEUE(x) pipe_producer_free(x)
#define FREE_INPUT_QUEUE(x) pipe_consumer_free(x)
#define FREE_QUEUE(x) pipe_free(x)

// Macros for casting strings to datatypes
#define INT_CAST(x) atoi(x)
#define FLOAT_CAST(x) atof(x)
#define BOOL_CAST(x) ((bool) atoi(x) != 0)
#define COMPLEX_CAST(x) (cast_complex(x))
#define BINARY_CAST(x) (cast_binary(x))

//Macros for reading and writing
#define FREAD1D(cols, data, infile) parse_csv1d(cols, data, infile)
#define FREAD2D(cols,rows, data, infile) parse_csv2d(cols,rows, data, infile)

//#define FWRITE(path, sep, lineq, cols, rows) parse_csv(path, sep, lineq, cols, rows)

// Utility function signatures
int parse_csv1d(int num_cols, char *(*), FILE *);
int parse_csv2d(int,int, char *(*)[], FILE *);
//void write_csv(char *, char *, output, int, int);
complex float cast_complex(char *);

char* cast_binary(char* s);
int getcols( const char * const line, const char * const delim, char ***out_storage, int );
char * read_text(char *path);

