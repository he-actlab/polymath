#include "utils.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "csv.h"





/* For more that 100 columns or lines (when delimiter = \n), minor modifications are needed. */
int getcols( const char * const line, const char * const delim, char ***out_storage, int cols )

{
    const char *start_ptr, *end_ptr, *iter;
    char **out;
    int i;                                          //For "for" loops in the old c style.
    int tokens_found = 1, delim_size, line_size;    //Calculate "line_size" indirectly, without strlen() call.
    int start_idx[MAXCOLSIZE];
    int end_idx[MAXCOLSIZE];   //Store the indexes of tokens. Example "Power;": loc('P')=1, loc(';')=6



    //Change 100 with MAX_TOKENS or use malloc() for more than 100 tokens. Example: "b1;b2;b3;...;b200"

    if ( *out_storage != NULL )                 return -4;  //This SHOULD be NULL: Not Already Allocated
    if (!delim ){
        printf("delim here\n");
        return -1;
    }
    if (!line){
            printf("line here\n");
        return -1;
    }
               //NULL pointers Rejected Here
    if ( (delim_size = strlen( delim )) == 0 )  return -2;  //Delimiter not provided

    start_ptr = line;   //Start visiting input. We will distinguish tokens in a single pass, for good performance.
                        //Then we are allocating one unified memory region & doing one memory copy.
    while ( ( end_ptr = strstr( start_ptr, delim ) ) ) {

        start_idx[ tokens_found -1 ] = start_ptr - line;    //Store the Index of current token
        end_idx[ tokens_found - 1 ] = end_ptr - line;       //Store Index of first character that will be replaced with
                                                            //'\0'. Example: "arg1||arg2||end" -> "arg1\0|arg2\0|end"
        tokens_found++;                                     //Accumulate the count of tokens.
        start_ptr = end_ptr + delim_size;                   //Set pointer to the next c-string within the line
    }

    for ( iter = start_ptr; (*iter!='\0') ; iter++ );

    start_idx[ tokens_found -1 ] = start_ptr - line;    //Store the Index of current token: of last token here.
    end_idx[ tokens_found -1 ] = iter - line;           //and the last element that will be replaced with \0

    line_size = iter - line;    //Saving CPU cycles: Indirectly Count the size of *line without using strlen();

    int size_ptr_region = (1 + tokens_found)*sizeof( char* );   //The size to store pointers to c-strings + 1 (*NULL).
    out = (char**) malloc( size_ptr_region + ( line_size + 1 ) + 5 );   //Fit everything there...it is all memory.
    //It reserves a contiguous space for both (char**) pointers AND string region. 5 Bytes for "Out of Range" tests.
    *out_storage = out;     //Update the char** pointer of the caller function.

    //"Out of Range" TEST. Verify that the extra reserved characters will not be changed. Assign Some Values.
    //char *extra_chars = (char*) out + size_ptr_region + ( line_size + 1 );
    //extra_chars[0] = 1; extra_chars[1] = 2; extra_chars[2] = 3; extra_chars[3] = 4; extra_chars[4] = 5;

    for ( i = 0; i < tokens_found; i++ )    //Assign adresses first part of the allocated memory pointers that point to
        out[ i ] = (char*) out + size_ptr_region + start_idx[ i ];  //the second part of the memory, reserved for Data.
    out[ tokens_found ] = (char*) NULL; //[ ptr1, ptr2, ... , ptrN, (char*) NULL, ... ]: We just added the (char*) NULL.
                                                        //Now assign the Data: c-strings. (\0 terminated strings):
    char *str_region = (char*) out + size_ptr_region;   //Region inside allocated memory which contains the String Data.
    memcpy( str_region, line, line_size );   //Copy input with delimiter characters: They will be replaced with \0.

    //Now we should replace: "arg1||arg2||arg3" with "arg1\0|arg2\0|arg3". Don't worry for characters after '\0'
    //They are not used in standard c lbraries.
    for( i = 0; i < tokens_found; i++) str_region[ end_idx[ i ] ] = '\0';

    //"Out of Range" TEST. Wait until Assigned Values are Printed back.
    //for ( int i=0; i < 5; i++ ) printf("c=%x ", extra_chars[i] ); printf("\n");

    // *out memory should now contain (example data):
    //[ ptr1, ptr2,...,ptrN, (char*) NULL, "token1\0", "token2\0",...,"tokenN\0", 5 bytes for tests ]
    //   |__________________________________^           ^              ^             ^
    //          |_______________________________________|              |             |
    //                   |_____________________________________________|      These 5 Bytes should be intact.

    return tokens_found;
}

char *read_text(char *path) {
    char *buffer;
    /* declare a file pointer */
    FILE    *infile;
    long    numbytes;

    /* open an existing file for reading */
    infile = fopen(path, "r");

    /* quit if the file does not exist */
    if(infile == NULL)
        exit(1);

    /* Get the number of bytes */
    fseek(infile, 0L, SEEK_END);
    numbytes = ftell(infile);

    /* reset the file position indicator to
    the beginning of the file */
    fseek(infile, 0L, SEEK_SET);

    /* grab sufficient memory for the
    buffer to hold the text */
    buffer = (char*)calloc(numbytes + 1, sizeof(char));

    /* memory error */
    if(buffer == NULL)
        exit(1);

    /* copy all the text into the buffer */
    fread(buffer, sizeof(char), numbytes, infile);
    buffer[numbytes] = '\0';
    fclose(infile);
    return buffer;
    /* confirm we have read the file by
    outputing it to the console */

    /* free the memory we used for the buffer */
}

int parse_csv2d(int num_cols, int num_rows, char* (*data)[num_cols], FILE *infile){
    char *line;
    int j;
    int i = 0;

    while (((line = csvgetline(infile)) != NULL) && (i < num_rows)) {
		for (j = 0; j < num_cols ; j++) {
            data[i][j] = strdup(csvfield(j));
		}
        i++;
    }

    if (line != NULL || num_rows == i){
        return 1;
    } else {
        return 0;
    }
}
int parse_csv1d(int num_cols, char *(*data), FILE *infile){
    char *line;
    int j;
    line = csvgetline(infile);
    if (line != NULL){
		for (j = 0; j < num_cols ; j++)
            data[j] = csvfield(j);
        return 1;
    } else {
        return 0;
    }
}
complex float cast_complex(char *value) {
  char *tok;
  char *plus=malloc(sizeof(char) * strlen(value));
  strcpy(plus,value);

  float real;
  float imag;
  tok = strsep(&value, "+");
  if (strlen(tok) != strlen(plus)){
    real = atof(tok);
    imag = atof(strsep(&value, "+"));
    return (real + imag*I);
  }

  char *minus=malloc(sizeof(char) * strlen(plus));
  strcpy(minus,plus);
  tok = strsep(&plus, "-");

  if (strlen(tok) != strlen(minus)){
    real = atof(tok);
    imag = atof(strsep(&plus, "-"));
    return (real - imag*I);
  }
  return (atof(tok) + 0*I);

}

char* stringToBinary(char* s) {
    if(s == NULL) return 0; /* no input string */
    size_t len = strlen(s);
    char *binary = malloc(len*8 + 1); // each char is one byte (8 bits) and + 1 at the end for null terminator
    binary[0] = '\0';
    for(size_t i = 0; i < len; ++i) {
        char ch = s[i];
        for(int j = 7; j >= 0; --j){
            if(ch & (1 << j)) {
                strcat(binary,"1");
            } else {
                strcat(binary,"0");
            }
        }
    }
    return binary;
}
