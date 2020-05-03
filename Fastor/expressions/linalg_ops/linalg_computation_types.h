#ifndef LINALG_COMPUTATION_TYPES_H
#define LINALG_COMPUTATION_TYPES_H

namespace Fastor {

// Pivot types
enum class PivType : int
{
    V = 0,          /* The permutation vector                    */
    M,              /* The permutatoin matrix                    */
};


// QR factorisation computation types
enum class QRCompType : int
{
    MGSR = 0,       /* Modified Gram-Schmidt Row-wise            */
    MGSRPiv,        /* Modified Gram-Schmidt Row-wise with pivot */
    HHR             /* House Holder Reflections                  */
};

// LU factorisation computation types
enum class LUCompType : int
{
    BlockLU = 0,    /* Block LU decomposition wiht no pivot      */
    BlockLUPiv,     /* Block LU decomposition wiht pivot         */
    SimpleLU,       /* Simple LU decomposition wiht no pivot     */
    SimpleLUPiv,    /* Simple LU decomposition wiht pivot        */
};

// Determinant computation type
enum class DetCompType : int
{
    Simple = 0,   /* Using simple hand-optimised calculations    */
    LU,           /* Using LU factorisation                      */
    LUPiv,        /* Using LU factorisation with pivot           */
    QR,           /* Using QR factorisation                      */
    QRPiv,        /* Using QR factorisation with pivot           */
    RREF,         /* Using Reduced Row Echelon Form              */
    RREFPiv,      /* Using Reduced Row Echelon Form with pivot   */
};

// Solve computation type
enum class SolveCompType : int
{
    Inverse = 0, /* Using optimised inversion                    */
    InversePiv,  /* Using optimised inversion with pivot         */
    BlockLU,     /* Using block LU factorisation                 */
    BlockLUPiv,  /* Using block LU factorisation with pivot      */
    SimpleLU,    /* Using block LU factorisation with pivot      */
    SimpleLUPiv, /* Using block LU factorisation with pivot      */
    QR,          /* Using QR factorisation                       */
    Chol,        /* Using Cholesky factorisation                 */
};


} // end of namespace Fastor

#endif // LINALG_COMPUTATION_TYPES_H