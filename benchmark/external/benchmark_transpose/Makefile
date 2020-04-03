OPT = -O3 -march=native -mfma -DNDEBUG
CXX_FLAGS = -std=c++17 $(OPT)
CXX_FLAGS += -I.

EIGENROOT = /Users/roman/Downloads/eigen-master/
BLAZEROOT = /Users/roman/Downloads/blaze-3.7/
FASTORROOT = ../../../

RUN_SINGLE =

CXX_FLAGS += $(RUN_SINGLE) -DFASTOR_TRANS_INNER_BLOCK_SIZE=1 -DFASTOR_TRANS_OUTER_BLOCK_SIZE=1

all:
	$(CXX) benchmark_transpose.cpp -o benchmark_transpose.exe $(CXX_FLAGS) -I$(EIGENROOT) -I$(BLAZEROOT) -I$(FASTORROOT)

run:
	./benchmark_transpose.exe

clean:
	rm -rf *.exe