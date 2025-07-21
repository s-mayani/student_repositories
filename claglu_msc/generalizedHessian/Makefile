BINARY = main
LIBS = -I/usr/include/eigen3
CXX = g++
CXXFLAGS = -std=c++17 -O3

.PHONY : all
	all: $(BINARY)

$(BINARY) : main.cpp hessian.h field.h convergence.h
	    $(CXX) $(LIBS) -o $@ $^ $(CXXFLAGS)

run:
	    ./hessian

.PHONY: clean
clean:
	    rm $(BINARY)

.PHONY: all clean
