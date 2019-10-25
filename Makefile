SHELL       = /bin/sh
ARCH        = LINUX
CC          = /usr/bin/mpicc
CLINKER     = $(CC)
AR          = ar crl
RANLIB      = ranlib
LOG_LIB     = -mpilog -lm
PROF_LIB    = -lmpe -lm
OPTFLAGS    = 
MPE_DIR     = 
MAKE        = make

VPATH=.:$(srcdir)
### End User configurable options ###
.SUFFIXES:	.cc .C .f90

CFLAGS	  = $(OPTFLAGS) 
CFLAGSMPE = $(CFLAGS) -I$(MPE_DIR)/include 
CCFLAGS	  = $(CFLAGS)
#FFLAGS	  = '-qdpc=e' 
FFLAGS	  = $(OPTFLAGS)
EXECS	  = matrix # BINARY NAME HERE
ALL_EXECS = ${EXECS} 

all: matrix  # add the target here

default: $(EXECS)

matrix: matrix.o
	$(CLINKER) $(OPTFLAGS) -o matrix matrix.o -lm

clean:
	/bin/rm -f *.o *~ PI* $(ALL_EXECS) upshot rdb.* startup.* core

.c.o:
	$(CC) $(CFLAGS) -c $<

