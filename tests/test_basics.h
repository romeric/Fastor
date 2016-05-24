#ifndef TEST_BASICS_H
#define TEST_BASICS_H

#include "tensor.h"
using namespace Fastor;

template<typename T>
void test_basics() {

    T number = 12.67;
    Tensor<T> a1 = 12.67;
    assert(a1 - number && "failed");

    Tensor<T,3> a2 = number;
    assert();

}

#endif // TEST_BASICS_H

