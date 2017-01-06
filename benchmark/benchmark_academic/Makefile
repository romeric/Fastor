all:
	echo "Running compilation benchmark for Fastor's optimisation levels"
	cd compilation_process/bypair_optimisation/
	./run_benchmark.sh gcc compile_time_bench
	./run_benchmark.sh clang compile_time_bench
	./run_benchmark.sh icc compile_time_bench
	./run_benchmark.sh gcc runtime_bench
	./run_benchmark.sh clang runtime_bench
	./run_benchmark.sh icc runtime_bench
	cd ../..

	echo "Running compliation benchmark for operation minimisation scheme"
	cd compilation_process/depthfirst/
	./run_benchmark.sh gcc
	./run_benchmark.sh clang
	./run_benchmark.sh icc
	cd ../..

	echo "Running run time benchmark for operation minimisation scheme"
	cd depthfirst/
	make 
	make run
	cd ..

	echo "Running SIMD tensor contraction benchmark for nonisomorphic products"
	cd contraction/
	make
	make run
	cd ..

	echo "Running SIMD tensor contraction benchmark for isomorphic products"
	cd outerproduct/
	make
	make run
	cd ..


clean:
	cd compilation_process/bypair_optimisation/
	make all_clean
	cd ../..
	cd compilation_process/depthfirst/
	make all_clean
	cd ../..
	cd depthfirst/
	make clean
	cd ..
	cd contraction/
	make clean
	cd ..
	cd outerproduct/
	make clean
	cd ..


