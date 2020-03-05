The tests can be ran with different compiler flags

~~~
make all && make run && make clean
make all CXX_FLAGS="-std=c++11" && make run && make clean
make all CXX_FLAGS="-std=c++11 -msse2" && make run && make clean
~~~

The first two combinations should always be tested as they cover optimisation ON and optimisation OFF
