#ifndef FASTOR_H
#define FASTOR_H

#define HAS_SSE
#define HAS_AVX

#include "commons/utils.h"
#include "simd_vector/SIMDVector.h"
#include "tensor/Tensor.h"
#include "tensor/tensor_print.h"
#include "tensor/tensor_funcs.h"
#include "tensor_algebra/einsum.h"
#include "expressions/expressions.h"
#include "backend/voigt.h"

//using Fastor::details::contraction;

#endif // FASTOR_H

