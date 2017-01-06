
INC = ../../../../
CXX_FLAGS = -O3 -mavx -std=c++11

ifeq "$(CXX)" "clang++-3.9"
    CXX_FLAGS += -fconstexpr-steps=16000000
endif
ifeq "$(CXX)" "clang++"
    CXX_FLAGS += -fconstexpr-steps=16000000
endif
ifeq "$(bench_test)" "runtime_bench"
    CXX_FLAGS += -DTEST_TIME
endif

ifndef $(TIME_RES)
    TIME_RES = out.txt 
endif

ifeq "$(CXX)" "g++"
    GXX = g++
endif
ifeq "$(CXX)" "g++-6"
    GXX = g++-6
endif

TIME = /usr/bin/time -f "%e Elapsed Real Time (secs), %M memory used" -o $(TIME_RES) --append
SIZE = du -hs a.out

ifeq "$(bench_test)" "profiler"
    CXX_FLAGS += -fconstexpr-steps=16000000 -Xtemplight -profiler -Xtemplight -memory -Xtemplight -ignore-system -Xtemplight -output=
endif


.SILENT:

all: $(bench_test)


compile_time_bench:

	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=0
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=1 
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=2 
	$(SIZE)


	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=0
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=1 
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=2 
	$(SIZE)


	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=0
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=1 
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=2 
	$(SIZE)


	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=0
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=1 
	$(SIZE)
ifneq "$(CXX)" "$(GXX)"
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=2 
	$(SIZE)
endif 


	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=0
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=1 
	$(SIZE)
ifneq "$(CXX)" "$(GXX)"
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=2 
	$(SIZE)
endif 


	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=0
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=1 
	$(SIZE)
ifneq "$(CXX)" "$(GXX)"
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=2 
	$(SIZE)
endif 

	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=0
	$(SIZE)
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=1 
	$(SIZE)
ifneq "$(CXX)" "$(GXX)"
	$(TIME) $(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS) -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=2 
	$(SIZE)
endif 


runtime_bench:

	$(CXX) compilation_pair_contraction.cpp -o out_7_0 $(CXX_FLAGS) -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp -o out_7_1 $(CXX_FLAGS) -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=1 
	$(CXX) compilation_pair_contraction.cpp -o out_7_2 $(CXX_FLAGS) -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=2 
	
	$(CXX) compilation_pair_contraction.cpp -o out_6_0 $(CXX_FLAGS) -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp -o out_6_1 $(CXX_FLAGS) -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=1 
	$(CXX) compilation_pair_contraction.cpp -o out_6_2 $(CXX_FLAGS) -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=2 

	$(CXX) compilation_pair_contraction.cpp -o out_5_0 $(CXX_FLAGS) -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp -o out_5_1 $(CXX_FLAGS) -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=1 
	$(CXX) compilation_pair_contraction.cpp -o out_5_2 $(CXX_FLAGS) -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=2 

	$(CXX) compilation_pair_contraction.cpp -o out_4_0 $(CXX_FLAGS) -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp -o out_4_1 $(CXX_FLAGS) -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=1 
ifneq "$(CXX)" "$(GXX)"
	$(CXX) compilation_pair_contraction.cpp -o out_4_2 $(CXX_FLAGS) -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=2 
endif 

	$(CXX) compilation_pair_contraction.cpp -o out_3_0 $(CXX_FLAGS) -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp -o out_3_1 $(CXX_FLAGS) -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=1 
ifneq "$(CXX)" "$(GXX)"
	$(CXX) compilation_pair_contraction.cpp -o out_3_2 $(CXX_FLAGS) -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=2 
endif 

	$(CXX) compilation_pair_contraction.cpp -o out_2_0 $(CXX_FLAGS) -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp -o out_2_1 $(CXX_FLAGS) -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=1 
ifneq "$(CXX)" "$(GXX)"
	$(CXX) compilation_pair_contraction.cpp -o out_2_2 $(CXX_FLAGS) -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=2 
endif

	$(CXX) compilation_pair_contraction.cpp -o out_1_0 $(CXX_FLAGS) -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp -o out_1_1 $(CXX_FLAGS) -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=1 
ifneq "$(CXX)" "$(GXX)"
	$(CXX) compilation_pair_contraction.cpp -o out_1_2 $(CXX_FLAGS) -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=2 
endif 

make run:
	./out_7_0
	./out_7_1
	./out_7_2
	./out_6_0
	./out_6_1
	./out_6_2
	./out_5_0
	./out_5_1
	./out_5_2
	./out_4_0
	./out_4_1
ifneq "$(CXX)" "$(GXX)"
	./out_4_2
endif 
	./out_3_0
	./out_3_1
ifneq "$(CXX)" "$(GXX)"
	./out_3_2
endif 	
	./out_2_0
	./out_2_1
ifneq "$(CXX)" "$(GXX)"
	./out_2_2
endif 
	./out_1_0
	./out_1_1
ifneq "$(CXX)" "$(GXX)"
	./out_1_2
endif 



profiler:

	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_7_0 -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_7_1 -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=1 
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_7_2 -I$(INC) \
		-DSEVEN_INDEX -DCONTRACT_OPT=2 
	
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_6_0 -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_6_1 -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=1 
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_6_2 -I$(INC) \
		-DSIX_INDEX -DCONTRACT_OPT=2 

	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_5_0 -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_5_1 -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=1 
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_5_2 -I$(INC) \
		-DFIVE_INDEX -DCONTRACT_OPT=2 

	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_4_0 -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_4_1 -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=1 
ifneq "$(CXX)" "$(GXX)"
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_4_2  -I$(INC) \
		-DFOUR_INDEX -DCONTRACT_OPT=2 
endif 

	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_3_0 -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_3_1 -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=1 
ifneq "$(CXX)" "$(GXX)"
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_3_2 -I$(INC) \
		-DTHREE_INDEX -DCONTRACT_OPT=2 
endif 

	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_2_0 -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_2_1 -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=1 
ifneq "$(CXX)" "$(GXX)"
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_2_2 -I$(INC) \
		-DTWO_INDEX -DCONTRACT_OPT=2 
endif

	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_1_0 -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=0
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_1_1 -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=1 
ifneq "$(CXX)" "$(GXX)"
	$(CXX) compilation_pair_contraction.cpp $(CXX_FLAGS)out_1_2 -I$(INC) \
		-DONE_INDEX -DCONTRACT_OPT=2 
endif 


clean:
	rm -rf a.out out_* out.txt

all_clean:
	rm -rf a.out out_* out.txt runtime_results_* binary_results_* compilation_results_*