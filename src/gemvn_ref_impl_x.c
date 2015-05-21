#include <stdlib.h>

void ATL_dgemvN
   (int M, int N, int MN, double alpha, double A[restrict const static MN], 
    int lda, double X[restrict const static M], int incX, const double beta,
    double Y[restrict const static M], const int incY)
{
#pragma scop
    for (int i=0; i < M; i++) {
        int ind;
        double y0 = Y[i] * beta;
        for (int j=0; j < N; j++) {
            ind = i + j*lda;
            y0 = y0 + A[ind]  *  X[j];
        }
        Y[i] = y0;
    }
#pragma endscop
}


const int M = 8192;
const int N = M;
double AA[M][N];

void matvecmul(int MM, int NN, double A[restrict const static MM][N], double X[restrict const static MM], double Y[restrict const static MM])
{
#pragma scop
    for (int i=0; i < MM; i++) {
        Y[i] = 0.0;
        for (int j=0; j < NN; j++) {
            Y[j] = Y[j] + A[i][j] * X[j];
        }
    }
#pragma endscop
}

int main()
{
    //const int M = 8192;
    //const int N = M;
    const int lda = M;
    const int incX = 1, incY = 1;
    double *A = (double*)malloc(sizeof(double)*M*N);
    double *X = (double*)malloc(sizeof(double)*N);
    double *Y = (double*)malloc(sizeof(double)*N);
    double alpha = 0.5, beta = 2.0;

    ATL_dgemvN(M, N, M*N, alpha, A, lda, X, incX, beta, Y, incY);
    matvecmul(M, N, AA, X, Y);
    free(A);
    free(X);
    free(Y);
}
