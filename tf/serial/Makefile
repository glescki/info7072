CXX ?= g++
CXX_FLAGS = -std=c++11 -Wall -Wextra -g -O2
INCLUDES = #-I include/ -I /usr/local/include
LIBS =

all: NeedlemanWunsch

NeedlemanWunsch: 
	$(CXX) NeedlemanWunsch.cpp -o NeedlemanWunsch $(CXX_FLAGS) $(INCLUDES) 

clean:
	rm -rf NeedlemanWunsch

test:
	./NeedlemanWunsch AAAAAAAAAA -r AAAGAATTAAAAAACAGTTA
