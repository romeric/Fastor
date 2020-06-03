#ifndef FASTOR_ALL_INCLUDE_H
#define FASTOR_ALL_INCLUDE_H

#if defined(_MSC_VER)
// Disable MSVC warnings for macros
#pragma warning (disable: 4003)
// Disable MSVC warnings for conversion
#pragma warning (disable: 4244)
// Disable MSVC unreferenced parameters
#pragma warning (disable: 4100)
// Disable MSVC narrowing conversion
#pragma warning (disable: 4267)
#endif

#include "config/config.h"
#include "util/util.h"
#include "simd_vector/SIMDVector.h"
#include "simd_math/simd_math.h"
#include "tensor/Tensor.h"
#include "tensor/TensorMap.h"
#include "tensor/TensorIO.h"
#include "tensor/TensorFunctions.h"
#include "tensor/AbstractTensorFunctions.h"
#include "tensor_algebra/einsum.h"
#include "tensor_algebra/network_einsum.h"
#include "tensor_algebra/einsum_explicit.h"
#include "tensor_algebra/abstract_contraction.h"
#include "expressions/expressions.h"
#include "backend/voigt.h"

#if defined(_MSC_VER)
#pragma warning (default: 4003)
#pragma warning (default: 4244)
#pragma warning (default: 4100)
#pragma warning (default: 4267)
#endif

#endif // FASTOR_ALL_INCLUDE_H

