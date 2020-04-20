OPT = -O3 -march=native -DNDEBUG
CXX_FLAGS = -std=c++17 $(OPT)
CXX_FLAGS += -I../

BLAZEROOT = /Users/roman/Downloads/blaze-3.7/
FASTORROOT = ../../../
MKLROOT = /glob/development-tools/parallel-studio/mkl/

MKLLIB = -L$(MKLROOT)/lib/intel64/ -lmkl_rt

RUN_SINGLE =

CXX_FLAGS += $(RUN_SINGLE)

all:
	$(CXX) benchmark_mkl_jit.cpp -o benchmark_mkl_jit.exe $(CXX_FLAGS) -I$(BLAZEROOT) -I$(FASTORROOT) $(MKLLIB)

run:
	./benchmark_mkl_jit.exe

clean:
	rm -rf *.exe