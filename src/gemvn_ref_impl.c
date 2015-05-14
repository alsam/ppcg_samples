
#ifdef BETA0
void ATL_dgemvN_a1_x1_b0_y1
#elif defined (BETA1)
void ATL_dgemvN_a1_x1_b1_y1
#else
void ATL_dgemvN_a1_x1_bX_y1
#endif
   (const int M, const int N, const double alpha, const double *A, 
    const int lda, const double *X, const int incX, const double beta,
    double *Y, const int incY)
{
   int i, j;
   double y0;

#pragma scop
   for (i=0; i != M; i++)
   {
      #ifdef BETA0
         y0 = 0.0;
      #elif defined(BETA1)
         y0 = Y[i];
      #elif defined(BETAX)
         y0 = Y[i] * beta;
      #endif
      for (j=0; j != N; j++) y0 += A[i+j*lda] * X[j];
      Y[i] = y0;
   }
#pragma endscop
}
