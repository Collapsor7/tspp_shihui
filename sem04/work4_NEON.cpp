#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h> 


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


void matrix_multiply_neon(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float64x4_t c = vdupq_n_f64(0.0);
            for (int k = 0; k < N; k += 2) { 
                float64x2_t a = vld1q_f64(&A[i * N + k]); 
                float64x2_t b = vld1q_f64(&B[k * N + j]); 
                c = vaddq_f64(c, vmulq_f64(a, b)); // C += A*B
            }
            double result[2];
            vst1q_f64(result, c); 
            C[i * N + j] = result[0] + result[1]; 
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
      double* C_neon = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));


      initialize_matrix(A, N[i]);
      initialize_matrix(B, N[i]);

      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      matrix_multiply_sequential(A, B, C_seq, N[i]);
      clock_gettime(CLOCK_MONOTONIC, &end);
      double seq_time = get_time_in_seconds(start, end);
      printf("Sequential version time: %.6f seconds\n", seq_time);

      clock_gettime(CLOCK_MONOTONIC, &start);
      matrix_multiply_neon(A, B, C_neon, N[i]);
      clock_gettime(CLOCK_MONOTONIC, &end);
      double neon_time = get_time_in_seconds(start, end);
      printf("NEON version time: %.6f seconds\n", neon_time);

      int correct = 1;
      for (int i = 0; i < N[i]; i++) {
          for (int j = 0; j < N[i]; j++) {
              if (C_seq[i * N[i] + j] != C_avx[i * N[i] + j] || C_seq[i * N[i] + j] != C_neon[i * N[i] + j]) {
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
      free(C_neon);

    }
   
    return 0;
}


