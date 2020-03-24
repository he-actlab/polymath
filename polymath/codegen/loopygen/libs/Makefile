CC=clang

OBJS=pipe.c utils.c csv.c
CFLAGS=-Wall -Wextra -Wpointer-arith -fstrict-aliasing -std=c11 -lm -DFORTIFY_SOURCE=2 -pipe -pedantic #-Werror
D_CFLAGS=-DDEBUG -g -O0
R_CFLAGS=-DNDEBUG -O3 -funroll-loops #-pg #-flto

all: $(NAME)_exec

$(NAME)_exec: $(OBJS) $(NAME).c
	$(CC) $(CFLAGS)  $(D_CFLAGS) -o $(NAME)_exec $(OBJS) $(NAME).c  -Wcomplex-component-init

target = $(shell sh -c '$(CC) -v 2>&1 | grep "Target:"')

utils.h:
utils.c: utils.h pipe.h
pipe.h:
pipe.c: pipe.h
csv.h:
csv.c: csv.h

$(NAME).c: pipe.h utils.h csv.h

clean:
	rm -f *.plist $(NAME)_exec  a.out