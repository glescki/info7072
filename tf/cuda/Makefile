CXX ?= nvcc
CXX_FLAGS = #-std=c++11 -Wall -Wextra -g -O2
INCLUDES = #-I include/ -I /usr/local/include
LIBS =

all: NeedlemanWunsch

NeedlemanWunsch: 
	nvcc NeedlemanWunsch.cu -o NeedlemanWunsch 

clean:
	rm -rf NeedlemanWunsch

test:
	./NeedlemanWunsch GGATCGA -r GAATTCAGTTA
