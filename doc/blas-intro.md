# Linear Algebra operations classification

Linear Algebra kernels belong either to **BLAS** (stands for **B**asic **L**inear **A**lgebra **S**ubprograms) - basic operations\ 
or **LAPACK** (**L**inear **A**lgebra **PACK**age) - larger operations.


## BLAS

There are the following dense or sparse matrix operations grouped by the <b>BLAS</b> levels:

### BLAS Level 1 - vector operations

```math #level1
\boldsymbol{y} \leftarrow \alpha \boldsymbol{x} + \boldsymbol{y} \!
```

These operations include:

    * DOT (inner product)
    * SUM (sum)
    * AXPBY (Scaled vector accumulation)
    * WAXPBY (Scaled vector addition) 
    
### BLAS Level 2 - matrix-vector operations

```math #level2
\boldsymbol{y} \leftarrow \alpha A \boldsymbol{x} + \beta \boldsymbol{y}, \quad \boldsymbol{y} \leftarrow \alpha A^T \boldsymbol{x} + \beta \boldsymbol{y} \!
```
These operations include:

    * GEMV (General matrix vector product)
    * GBMV (Banded matrix vector product)
    * SYMV (Symmetric matrix vector product)
    * SPMV (Symmetric matrix vector product, packed format)
    * HEMV (Hermitian matrix vector product)
    * HPMV (Hermitian matrix vector product, packed format)
    * TRSV (Triangular solve) 

  dtrsv, strsv, ztrsv and ctrsv solve one of the systems of equations

```math #level2_trsv
A \boldsymbol{x} = \boldsymbol{b}, \quad A^T \boldsymbol{x} = \boldsymbol{b} \!
```

### BLAS Level 3 - matrix-matrix operation

```math #level3_gemm
C \leftarrow \alpha A B + \beta C \
```


for

    - GEMM (General matrix matrix product)
    - SYMM, HEMM (Symmetric or Hermitian matrix matrix product)
    - TRMM (triangular matrix multiply)
    - GBMM (general banded matrix multiply)

and

    - SYRK, SYR2K, HERK, HER2K (symmetric or Hermitian rank-k and rank-2k updates)


## BLAS calls in Linpack

    * Level 1: idamax, dcopy, dscal, dswap
        * iamax (idamax) - find index of maximum element in a vector
        * copy (dcopy) - copy vector
        * scal (dscal) - scale a vector by a constant
        * swap (dswap) - swap two vectors

    * Level 2:
        * DGER - ``A\leftarrow{A}+\alpha{x}y^T``
 
    * Level 3:
        * DTRSM - solve one of the matrix equations  ``op(A)*X=\alpha*B, X*op(A) = \alpha*B``


## LAPACK

LAPACK operations include:

    * GTSV - solve the equation ``Ax=b``, where ``A`` is a tridiagonal matrix, ``x`` and ``b`` - vectors
    * GETRF - compute an LU factorization of a general M-by-N matrix A using partial pivoting with row interchanges


## Sparse Linear Algebra

### overview

Sparse matrix-vector multiplication (SpMV) is one of the most important operation in sparse matrix computations. Iterative
methods for solving large linear systems (``Ax=b``) and eigenvalue problems (``Ax = \lambda x``) require huge number of these operations.

Compared to dense linear algebra kernels, sparse kernels suffer from higher instruction and storage overheads per ﬂop, as well as indirect and irregular memory access patterns. Achieving higher performance on these platforms requires choosing a compact data structure and code transformations that best exploit properties of both the sparse matrix – which may be known only at run-time – and the underlying machine architecture. This need for optimization and tuning at run-time is a major distinction from the dense case.

The general form for the SpMV operation is ``y \leftarrow Ax+y`` where A is large and sparse and x and y are dense column vectors.
There are many sparse formats for storing sparse matrices. The most popular is Compressed Sparse Row (CSR) format.
The CSR format stores an <math>m\times n</math> sparse matrix having k non-zero elements using three one-dimensional arrays: the arrays
val and col ind, each of size k, to store the non-zeros values and column indices, respectively; and an array row ptr of size m + 1
to store pointers to the ﬁrst element of each row in the val and col ind arrays. 
Basic reference SpMV kernel implementation for Compressed Sparse Row (CSR) matrix where:
* the column indices within a given row are sorted in increasing order (i.e., the last element of each row is the diagonal non-zero value),
* ``x`` is an array with unit-stride

```c++
// y <- y + A*x, where A is in CSR format
for (size_t i = 0; i < m; i++) {
  float acc = y[i];
  for (size_t k = ptr[i]; k < ptr[i+1]; k++) {
    acc = acc + val[k] * x[ind[k]];
  }
  y[i] = acc;
}
```

*Fortran 90* introduced vector notation, vector intrinsics such as dot product and triplet notation for vector slices,
thus an equivalent code is shorter than C one:

```fortran
DO I=1,M
   K1 = IPTR(I)
   K2 = IPTR(I+1)-1
   Y(I) = DOTPRODUCT(VAL(K1:K2),X(IND(K1:K2)))
ENDDO
```


Consider the internal loop in the above C code excerpt - dot product. For single precision SpMV pumping of 12 bytes is required for processing one non-zero matrix entry (value from A, value from vector x, column index) and 2 flops (multiplication and addition - FFMA/DFMA is a natural choice). The outer loop can be very large - proportional to the number of matrix rows, internal loop workload is proportional to the number of non-zeros of this row.
Thus imbalance in the number of non-zeros might lead to workload imbalance.

Another interesting kernel: *SpTRSV*

Basic reference triangular solver kernel implementation for ``T\cdot x = b`` for ``T`` where:
``T`` is a lower triangular 
a full (all non-zero) diagonal exists,
the column indices within a given row are sorted in increasing order (i.e., the last element of each row is the diagonal non-zero value),
``x`` is an array with unit-stride


```c++
for (size_t i = 0; i < m; i++) {
  float xi = x[i];
  for (size_t k = ptr[i]; k < ptr[i+1]-1; k++) {
    xi -= val[k] * x[ind[k]];
  }
  xi /= val[ptr[i+1]-1];
  x[i] = xi;
}
```


## tridiagonal matrix solvers notes (*LAPACK* **GTSV** operation)

### Introduction

We wish to solve a system of n linear equations of the form
``Ax = d``, where ``A`` is a tridiagonal matrix:

```math 
\begin{pmatrix}
   {b_1} & {c_1} & {   } & {   } & { 0 } \\
   {a_2} & {b_2} & {c_2} & {   } & {   } \\
   {   } & {a_3} & {b_3} & \ddots & {   } \\
   {   } & {   } & \ddots & \ddots & {c_{n-1}}\\
   { 0 } & {   } & {   } & {a_n} & {b_n}\\
\end{pmatrix}
=
\begin{pmatrix}
   {d_1 }  \\
   {d_2 }  \\
   {d_3 }  \\
   \vdots   \\
   {d_n }  \\
\end{pmatrix}
.
```

Classical [Thomas algorithm](http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)) - a simplified form of Gaussian elimination - has asymptotic complexity ``O(N)`` instead of ``O(N^3)`` for Gaussian elimination. This algorithm is a base for sweep variant of solvers.

Sweep algorithm doesn't parallelize tridiagonal system solution, parallelism appear only during of large number of systems simultaneously. Cyclic reduction and parallel cyclic reduction from another hand parallelize tridiagonal system solution. Cyclic reduction has better asymptotic compxity ``O(N)`` compared to ``O(\log(N))`` for pcr, but it worse in terms of address divergence. An appropriate algorithm choice depends on the task.

+ References
    + [N. Bell and M. Garland. Efficient sparse matrix-vector multiplication on CUDA. NVIDIA Technical Report NVR-2008-004, December 2008.](http://mgarland.org/files/papers/nvr-2008-004.pdf)
    + [N. Bell and M. Garland. Implementing sparse matrix-vector multiplication on throughput-oriented processors. Proc. Supercomputing 2009, To appear, November 2009.](http://sg.nvidia.com/docs/IO/77944/sc09-spmv-throughput.pdf)
    + [Jee Whan Choi, Amik Singh, and Richard W. Vuduc. Model-driven autotuning of sparse matrix-vector multiply on GPUs. In Proc. ACM SIGPLAN Symp. Principles and Practice of Parallel Programming (PPoPP), Bangalore, India, January 2010.](http://vuduc.org/pubs/choi2010-gpu-spmv.pdf)
    + [Richard Vuduc, Aparna Chandramowlishwaran, Jee Whan Choi, Murat Efe Guney, and Aashay Shringarpure. On the limits of GPU acceleration. In Proc. USENIX Wkshp. Hot Topics in Parallelism (HotPar), Berkeley, CA, USA, June 2010.](http://vuduc.org/pubs/vuduc2010-hotpar-cpu-v-gpu.pdf)
