CXX_GCC = g++
CXX_CLANG = clang++
CXX_FLAGS = -std=c++11 -O3 -mavx -funroll-loops 
GCC_COMPAT = -fabi-version=6 -finline-functions
INCLUDES = -I../

all: tests_gcc tests_clang

tests_gcc: gcc_test_basics gcc_test_linalg
tests_clang: clang_test_basics clang_test_linalg

gcc_test_basics:
	$(CXX_GCC) test_basics.cpp -o test_basics_gcc.exe $(CXX_FLAGS) $(GCC_COMPAT) $(INCLUDES)	
clang_test_basics:
	$(CXX_CLANG) test_basics.cpp -o test_basics_clang.exe $(CXX_FLAGS) $(INCLUDES)

gcc_test_linalg:
	$(CXX_GCC) test_linalg.cpp -o test_linalg_gcc.exe $(CXX_FLAGS) $(GCC_COMPAT) $(INCLUDES)	
clang_test_linalg:
	$(CXX_CLANG) test_linalg.cpp -o test_linalg_clang.exe $(CXX_FLAGS) $(INCLUDES)