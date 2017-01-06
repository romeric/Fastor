
INC = ../../../../
CXX_FLAGS = -O3 -mavx -std=c++11 -DFASTOR_PRINT_COST

ifeq "$(OPFLAG)" "NOP"
    CXX_FLAGS += -DFASTOR_DONT_PERFORM_OP_MIN
endif

ifeq "$(CXX)" "clang++-3.9"
    CXX_FLAGS += -fconstexpr-steps=16000000
endif
ifeq "$(CXX)" "clang++"
    CXX_FLAGS += -fconstexpr-steps=16000000
endif

ifndef $(TIME_RES)
    TIME_RES = out.txt 
endif

ifeq "$(CXX)" "g++"
    GXX = g++
    CXX_FLAGS += -fabi-version=6
endif
ifeq "$(CXX)" "g++-6"
    GXX = g++-6
    CXX_FLAGS += -fabi-version=6
endif

TIME = /usr/bin/time -f "%e Elapsed Real Time (secs), %M memory used" -o $(TIME_RES) --append
SIZE = du -hs a.out


.SILENT:

all: bench


bench:
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_0 -DTHREE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_1 -DTHREE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_2 -DTHREE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_3 -DTHREE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_4 -DTHREE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_5 -DTHREE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_6 -DTHREE_TENSOR
	$(SIZE)
	./a.out
	
	# $(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
	# 	-DSPAN_7 -DTHREE_TENSOR
	# $(SIZE)
	# ./a.out
	# $(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
	# 	-DSPAN_8 -DTHREE_TENSOR
	# $(SIZE)
	# ./a.out



	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_0 -DFOUR_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_1 -DFOUR_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_2 -DFOUR_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_3 -DFOUR_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_4 -DFOUR_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_5 -DFOUR_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_6 -DFOUR_TENSOR
	$(SIZE)
	./a.out
	# $(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
	# 	-DSPAN_7 -DFOUR_TENSOR
	# $(SIZE)
	# ./a.out
	# $(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
	# 	-DSPAN_8 -DFOUR_TENSOR
	# $(SIZE)
	# ./a.out


	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_0 -DFIVE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_1 -DFIVE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_2 -DFIVE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_3 -DFIVE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_4 -DFIVE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_5 -DFIVE_TENSOR
	$(SIZE)
	./a.out
	$(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
		-DSPAN_6 -DFIVE_TENSOR
	$(SIZE)
	./a.out
	# $(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
	# 	-DSPAN_7 -DFIVE_TENSOR
	# $(SIZE)
	# ./a.out
	# $(TIME) $(CXX) $(CXX_FLAGS) depthfirst_compilation.cpp -I$(INC) \
	# 	-DSPAN_8 -DFIVE_TENSOR
	# $(SIZE)
	# ./a.out


clean:
	rm -rf a.out out.txt

all_clean:
	rm -rf a.out out.txt compilation_results_* binary_results_*