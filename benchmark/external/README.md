To compile the benchmarks you need to specifiy your library paths

~~~
make all EIGENROOT=... BLAZEROOT=... LIBXSMMROOT... FASTORROOT=...
~~~

Single and double precision benchmarks are run one at a time you need to define `make RUN_SINGL=-DRUN_SINGLE` for single precision benchmarks. The default is double precision

To run the benchmarks

~~~
make run >> benchmark_results_double.txt
~~~

Note that you will have to run the benchmark a few times to get a sense of where each library stands.

To plot the the results, you need python with numpy and matplotlib

~~~
python benchmark_plot.py
~~~