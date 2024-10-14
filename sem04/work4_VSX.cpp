#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <altivec.h> // 包含 VSX 指令集的头文件

// 初始化矩阵，使用较小的数值 (< 1.0)
void initialize_matrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (float)rand() / RAND_MAX; // 生成小于 1.0 的浮点数
        }
    }
}

// 顺序矩阵乘法
void matrix_multiply_sequential(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// 使用 VSX 进行向量化矩阵乘法
void matrix_multiply_vsx(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            vector float c = {0.0f, 0.0f, 0.0f, 0.0f}; // 初始化 C 的向量值
            for (int k = 0; k < N; k += 4) { // 每次处理 4 个元素
                vector float a = vec_ld(0, &A[i * N + k]); // 加载 A 矩阵的行
                vector float b = vec_ld(0, &B[k * N + j]); // 加载 B 矩阵的列
                c = vec_madd(a, b, c); // C += A*B
            }
            float result[4];
            vec_st(c, 0, result); // 将 VSX 结果存储回内存
            C[i * N + j] = result[0] + result[1] + result[2] + result[3]; // 累加得到最终结果
        }
    }
}

// 计时函数，返回执行时间（秒）
double get_time_in_seconds(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
    int N[] = {512,1024,2048}; 
    for(int i =0;i<3;i++){
      printf("N = %d\n", N[i]);
      float* A = (float*)aligned_alloc(32, N[i] * N[i] * sizeof(float));
      float* B = (float*)aligned_alloc(32, N[i] * N[i] * sizeof(float));
      float* C_seq = (float*)aligned_alloc(32, N[i] * N[i] * sizeof(float));
      float* C_vsx = (float*)aligned_alloc(32, N[i] * N[i] * sizeof(float));

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

      // VSX 矩阵乘法
      clock_gettime(CLOCK_MONOTONIC, &start);
      matrix_multiply_vsx(A, B, C_vsx, N[i]);
      clock_gettime(CLOCK_MONOTONIC, &end);
      double vsx_time = get_time_in_seconds(start, end);
      printf("VSX version time: %.6f seconds\n", vsx_time);

      // 结果验证
      int correct = 1;
      for (int i = 0; i < N[i]; i++) {
          for (int j = 0; j < N[i]; j++) {
              if (C_seq[i * N[i] + j] != C_vsx[i * N[i] + j]) {
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
      free(C_vsx);   
    }


    return 0;
}


