OPT = -O3 -march=native -mfma -DNDEBUG
CXX_FLAGS = -std=c++17 $(OPT)
CXX_FLAGS += -I.

EIGENROOT = /Users/roman/Downloads/eigen-master/
FASTORROOT = ../../../

RUN_SINGLE =

CXX_FLAGS += $(RUN_SINGLE)

all:
	$(CXX) benchmark_inverse.cpp -o benchmark_inverse.exe $(CXX_FLAGS) -I$(EIGENROOT) -I$(FASTORROOT)

run:
	./benchmark_inverse.exe

clean:
	rm -rf *.exe