#!/bin/bash

# CALL THIS SCRIPT WITH ARGUMENTS LIKE SO
# ./run_benchmark.sh gcc compile_time_bench
# ./run_benchmark.sh icc compile_time_bench
# ./run_benchmark.sh clang runtime_bench

# FOR RUNNING TEMPLATE INSTANTIATION PROFILER (YOU NEED TO HAVE TEMPLIGHT INSTALLED)
# ./run_benchmark.sh profile


read GCC CLANG ICC TEMPLIGHT < <(../../../detect_compiler $1)


if [ "$1" == "gcc" ] || [ "$1" == "g++" ]; then
	COMPILER=$GCC
	CRES=compilation_results_gcc
	BRES=binary_results_gcc
	RRES=runtime_results_gcc
elif [ "$1" == "clang" ] || [ "$1" == "clang++" ]; then
	COMPILER=$CLANG
	CRES=compilation_results_clang
	BRES=binary_results_clang
	RRES=runtime_results_clang
elif [ "$1" == "icc" ] || [ "$1" == "icpc" ]; then
	COMPILER=$ICC
	CRES=compilation_results_icc
	BRES=binary_results_icc
	RRES=runtime_results_icc
elif [ "$1" == "profile" ]; then
	templight=$TEMPLIGHT
else
	echo "Don't understand the compiler"
	exit 1
fi

# Test to run
if [ "$2" != "runtime_bench" ] && [ "$2" != "compile_time_bench" ]; then
	if [ "$1" != "profile" ]; then
		echo "Don't understand which benchmark test to run. I am going to assume you meant compile time benchmark"
		bench=compile_time_bench
	fi
else
	bench=$2
fi



if [ "$bench" == "compile_time_bench" ]; then
	make CXX=$COMPILER bench_test=$bench TIME_RES=$CRES > $BRES
elif [ "$bench" == "runtime_bench" ]; then
	make CXX=$COMPILER bench_test=$bench
	make run CXX=$COMPILER > $RRES
fi

if [ "$1" == "profile" ]; then
	make CXX=$templight bench_test=profiler
fi
