#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h> 

// 初始化矩阵，使用较小的数值 (< 1.0)
void initialize_matrix(double* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (double)rand() / RAND_MAX; // 生成小于 1.0 的浮点数
        }
    }
}

// 顺序矩阵乘法
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
            float64x4_t c = vdupq_n_f64(0.0); // 初始化 C 的向量值
            for (int k = 0; k < N; k += 2) { // 每次处理 2 个元素
                float64x2_t a = vld1q_f64(&A[i * N + k]); // 加载 A 矩阵的行
                float64x2_t b = vld1q_f64(&B[k * N + j]); // 加载 B 矩阵的列
                c = vaddq_f64(c, vmulq_f64(a, b)); // C += A*B
            }
            double result[2];
            vst1q_f64(result, c); // 将 NEON 结果存储回内存
            C[i * N + j] = result[0] + result[1]; // 累加得到最终结果
        }
    }
}

// 计时函数，返回执行时间（秒）
double get_time_in_seconds(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
    int N[] = {512,1024,2048}; // 可以根据任务要求修改为 512, 1024, 2048
    for(int i =0;i<3;i++){
      printf("N = %d\n", N[i]);
      double* A = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));
      double* B = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));
      double* C_seq = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));
      double* C_avx = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));
      double* C_neon = (double*)aligned_alloc(32, N[i] * N[i] * sizeof(double));

      // 初始化矩阵 A 和 B
      initialize_matrix(A, N[i]);
      initialize_matrix(B, N[i]);

      // 顺序矩阵乘法
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

      // 释放内存
      free(A);
      free(B);
      free(C_seq);
      free(C_avx);
      free(C_neon);

    }
   
    return 0;
}


