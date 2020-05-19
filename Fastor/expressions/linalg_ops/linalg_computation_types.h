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
    QR,           /* Using QR factorisation                      */
};

// Inverse computation type
enum class InvCompType : int
{
    SimpleInv = 0,/* Using simple hand-optimised calculations    */
    SimpleInvPiv, /* Simple with pivot                           */
    BlockLU,      /* Using block LU factorisation                */
    BlockLUPiv,   /* Using block LU factorisation with pivot     */
    SimpleLU,     /* Using simple LU factorisation               */
    SimpleLUPiv,  /* Using simple LU factorisation with pivot    */
};

// Solve computation type
enum class SolveCompType : int
{
    SimpleInv = 0, /* Using optimised inversion                    */
    SimpleInvPiv,  /* Using optimised inversion with pivot         */
    BlockLU,       /* Using block LU factorisation                 */
    BlockLUPiv,    /* Using block LU factorisation with pivot      */
    SimpleLU,      /* Using simple LU factorisation                */
    SimpleLUPiv,   /* Using simple LU factorisation with pivot     */
    QR,            /* Using QR factorisation                       */
    Chol,          /* Using Cholesky factorisation                 */
};


} // end of namespace Fastor

#endif // LINALG_COMPUTATION_TYPES_H
