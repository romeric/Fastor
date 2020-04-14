#ifndef TENSOR_VIEWS_H
#define TENSOR_VIEWS_H

#include "Fastor/expressions/views/tensor_views_1d.h"
#include "Fastor/expressions/views/tensor_views_2d.h"
#ifdef FASTOR_USE_OLD_NDVIEWS
#include "Fastor/expressions/views/tensor_views_nd_idivmod.h"
#else
#include "Fastor/expressions/views/tensor_views_nd.h"
#endif

#include "Fastor/expressions/views/tensor_views_assignment.h"

#endif // TENSOR_VIEWS_H
