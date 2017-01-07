To run the backend benchmarks you typically do `make && make run`
For academic benchmarks, some Makefiles are hard-wired at the moment (`contraction` and `outer product` SIMD benchmarks). 
The other benchmarks should build and run fine.

Compile time profiling results can be seen if `templight` and `templar` are installed.