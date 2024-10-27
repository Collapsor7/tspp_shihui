#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h> 

void print_m256d(__m256d vec, const char* name){
  double result[4];
  _mm256_storeu_pd(result,vec);
  printf("%s = [ %.6f , %.6f , %.6f , %.6f ] \n",name,result[0],result[1],result[2],result[3]);
}

void initialize_matrix(double* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (double)rand() / RAND_MAX;
        }
    }
}

void matrix_multiply_sequential(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void matrix_multiply_avx(double* A, double* B, double* C, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      __m256d c = _mm256_setzero_pd();
      int k = 0;
      for (; k <= N - 4; k += 4) {
        __m256d a = _mm256_loadu_pd(&A[i * N + k]);
        __m256d b = _mm256_set_pd(B[(k+3) * N + j],
                                  B[(k+2) * N + j],
                                  B[(k+1) * N + j],
                                  B[k * N + j]);
        c = _mm256_add_pd(c, _mm256_mul_pd(a, b)); // C += A*B
      }

      double sum = 0.0;
      for (; k < N; k++) {
          sum += A[i * N + k] * B[k * N + j];
        }
      double result[4];
      _mm256_storeu_pd(result, c);
      C[i * N + j] = result[0] + result[1] + result[2] + result[3] + sum;
    }
  }
}

double get_time_in_seconds(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
    int N[] = {512,1024,2048}; 

    for(int i =0;i<3;i++){
      printf("N = %d\n", N[i]);
      double* A = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));
      double* B = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));
      double* C_seq = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));
      double* C_avx = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));

      initialize_matrix(A, N[i]);
      initialize_matrix(B, N[i]);

      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      matrix_multiply_sequential(A, B, C_seq, N[i]);
      clock_gettime(CLOCK_MONOTONIC, &end);
      double seq_time = get_time_in_seconds(start, end);
      printf("Sequential version time: %.6f seconds\n", seq_time);

      clock_gettime(CLOCK_MONOTONIC, &start);
      matrix_multiply_avx(A, B, C_avx, N[i]);
      clock_gettime(CLOCK_MONOTONIC, &end);
      double avx_time = get_time_in_seconds(start, end);
      printf("AVX version time: %.6f seconds\n", avx_time);

      int correct = 1;
      for (int i = 0; i < N[i]; i++) {
          for (int j = 0; j < N[i]; j++) {
              if (C_seq[i * N[i] + j] - C_avx[i * N[i] + j] > 1e-6) {
                  correct = 0;
                  break;
              }
          }
          if (!correct) break;
      }
      if (correct) {
          printf("Results are correct!\n");
      } else {
          printf("Results are incorrect!\n");
      }

      free(A);
      free(B);
      free(C_seq);
      free(C_avx);

    }
   
    return 0;
}


