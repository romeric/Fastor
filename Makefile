gcc_484:
	g++ Fastor.cpp -o Fastor -std=c++11 -fabi-version=0 \
	-I/home/roman/Dropbox/eigen-devel/ -I/home/roman/Dropbox/zdump/ \
	-I/home/roman/Dropbox/zdump/vdt \
	-O3 -funroll-loops -march=native -DNDEBUG -I. # -fopenmp -pthread -lpthread 

gcc_521:
	/opt/gcc-5.2/bin/g++ Fastor.cpp -o Fastor -std=c++14 -fabi-version=0 \
	-I/home/roman/Dropbox/eigen-devel/ -I/home/roman/Dropbox/zdump/ \
	-I/home/roman/Dropbox/zdump/vdt \
	-O3 -funroll-loops -march=native -I. -DNDEBUG # -fno-tree-vectorize # -fopenmp -pthread -lpthread 

gcc_530:
	/opt/gcc-5.3.0/bin/g++ Fastor.cpp -o Fastor -std=c++11 -fabi-version=0 \
	-I/home/roman/Dropbox/eigen-devel/ -I/home/roman/Dropbox/zdump/ \
	-I/home/roman/Dropbox/zdump/vdt \
	-O3 -funroll-loops -march=native -DNDEBUG -I. # -fopenmp -pthread -lpthread 

clang_361:
	clang++ Fastor.cpp -o Fastor -std=c++11 \
	-I/home/roman/Dropbox/eigen-devel/ -I/home/roman/Dropbox/zdump/ \
	-I/home/roman/Dropbox/zdump/vdt \
	-O3 -funroll-loops -march=native -DNDEBUG -I. # -fopenmp -pthread -lpthread 

clang_380:
	/home/roman/Downloads/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-14.04/bin/clang++ Fastor.cpp -o Fastor -std=c++14 \
	-I/home/roman/Dropbox/eigen-devel/ -I/home/roman/Dropbox/zdump/ \
	-I/home/roman/Dropbox/zdump/vdt \
	-O3 -funroll-loops -march=native -DNDEBUG -I. # -fopenmp -pthread -lpthread 

icc_1602:
	/media/MATLAB/intel/bin/icpc Fastor.cpp -o Fastor -std=c++11 \
	-I/home/roman/Dropbox/eigen-devel/ -I/home/roman/Dropbox/zdump/ \
	-I/home/roman/Dropbox/zdump/vdt \
	-O3 -funroll-loops -march=native -DNDEBUG -I. # -fopenmp -pthread -lpthread 

gcc_484_a:
	g++ -S Fastor.cpp -o Fastor.asm -std=c++11 -fabi-version=0 \
	-I/home/roman/Dropbox/eigen-devel/ -I/home/roman/Dropbox/zdump/ \
	-I/home/roman/Dropbox/zdump/vdt \
	-O3 -funroll-loops -march=native -DNDEBUG -I. # -fopenmp -pthread -lpthread 