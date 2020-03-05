#ifndef TENSOR_VIEWS_H
#define TENSOR_VIEWS_H

#include "tensor_views_1d.h"
#include "tensor_views_2d.h"
#ifdef FASTOR_USE_OLD_NDVIEWS
#include "tensor_views_nd_idivmod.h"
#else
#include "tensor_views_nd.h"
#endif

#endif // TENSOR_VIEWS_H