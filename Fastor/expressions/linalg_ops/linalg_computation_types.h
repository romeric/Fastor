#ifndef LINALG_COMPUTATION_TYPES_H
#define LINALG_COMPUTATION_TYPES_H

namespace Fastor {

// QR factorisation computation types
enum class QRCompType : int
{
    MGSR = 0,       /* Modified Gram-Schmidt Row-wise            */
    MGSRPivot,      /* Modified Gram-Schmidt Row-wise with pivot */
    HHR             /* House Holder Reflections                  */
};

// Determinant computation type
enum class DetCompType : int
{
    Simple = 0,   /* Using simple hand-optimised calculations    */
    RREF,         /* Using Reduced Row Echelon Form              */
    RREFPivot,    /* Using Reduced Row Echelon Form with pivot   */
    QR            /* Using QR factorisation                      */
};

// Solve computation type
enum class SolveCompType : int
{
    Inverse = 0, /* Using optimised inversion                    */
    Chol,        /* Using Cholesky factorisation                 */
    LU,          /* Using LU factorisation                       */
    QR           /* Using QR factorisation                       */
};


} // end of namespace Fastor

#endif // LINALG_COMPUTATION_TYPES_H