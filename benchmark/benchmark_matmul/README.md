To compile the benchmarks you need to specifiy your library paths

~~~
make all EIGENROOT=... BLAZEROOT=... LIBXSMMROOT... FASTORROOT=...
~~~

Single and double precision benchmarks are run one at a time you need to define `-DRUN_SINGLE`

To run the benchmarks

~~~
make run >> benchmark_results_double.txt
~~~

To plot the the results, you need python with numpy and matplotlib

~~~
python benchmark_plot.py
~~~