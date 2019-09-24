#include <Fastor.h>

using namespace Fastor;


int main()
{

    enum{i,j,k};
    Tensor<double, 2, 2> a  = {{1, 2}, {3, 4}};
    Tensor<double, 2>    w  = {1, 1};
    // Tensor<double,2> e3 = einsum<Index<i, j>, Index <i> >(a, w);
    // Tensor<double,2> e3 = einsum<Index<i, j>, Index <j> >(a, w);

    // Tensor<double,2> e3 = einsum<Index<i>, Index <i,j> >(w, a);
    Tensor<double,2> e3 = einsum<Index<j>, Index <i,j> >(w, a);


    print(a,w,e3);

    return 0;
}