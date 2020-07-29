To run the tests do

~~~
cd tests && mkdir build && cd build && cmake .. && make -j 4 && ctest -V
~~~

To control verbosity of cmake build and ctest run processes you can specify

~~~
cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON ../ && make && ctest -V
~~~

To run the tests with optimisation ON and optimisation OFF you can specify

~~~
cmake -DCMAKE_BUILD_TYPE=Debug
cmake -DCMAKE_BUILD_TYPE=Release
~~~
