#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <ctime>

bool verify_result(const std::vector<int> &A, const std::vector<int> &B, const std::vector<int> &C, int N) {
    bool correct = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int expected_value = 0;
            for (int k = 0; k < N; k++) {
                expected_value += A[i * N + k] * B[k * N + j];
            }
            if (C[i * N + j] != expected_value) {
                printf("Mismatch at C[%d][%d]: expected %d, got %d\n", i, j, expected_value, C[i * N + j]);
                correct = false;
            }
        }
    }
    return correct;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank_world, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N, b;
    if (argc > 2) {
        N = atoi(argv[1]);
        b = atoi(argv[2]);
    } else {
        if (rank_world == 0) {
            printf("Usage: mpirun -np <P> %s <matrix_size N> <block_size b>\n", argv[0]);
            printf("Please provide the matrix size N and block size b.\n");
        }
        MPI_Finalize();
        return 0;
    }

    int sqrt_p = (int)sqrt(size);
    if (sqrt_p * sqrt_p != size) {
        if (rank_world == 0)
            printf("Error: The number of processes P must be a perfect square.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (N % sqrt_p != 0) {
        if (rank_world == 0)
            printf("Error: N must be divisible by sqrt(P).\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_L = N / sqrt_p;

    if (b > local_L || local_L % b != 0) {
        if (rank_world == 0)
            printf("Error: Block size b must be less than or equal to local matrix size L and L must be divisible by b.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dims[2] = {sqrt_p, sqrt_p};
    int periods[2] = {0, 0};
    int coords[2];
    MPI_Comm cart_comm;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(cart_comm, coords[1], coords[0], &col_comm);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    int local_size = local_L * local_L;

    std::vector<int> A(local_size);
    std::vector<int> B(local_size);
    std::vector<int> C(local_size, 0);

    std::vector<int> global_A, global_B, global_C;
    if (rank_world == 0) {
        global_A.resize(N * N);
        global_B.resize(N * N);
        global_C.resize(N * N, 0);

        srand(0); // 为了调试，使用固定的随机种子
        for (int i = 0; i < N * N; i++) {
            global_A[i] = rand() % 10;
            global_B[i] = rand() % 10;
        }
        // printf("Global Matrix A:\n");
        // for (int i = 0; i < N; i++) {
        //     printf("Row %d: ", i);
        //     for (int j = 0; j < N; j++) {
        //         printf("%d ", global_A[i * N + j]);
        //     }
        //     printf("\n");
        // }

        // printf("\nGlobal Matrix B:\n");
        // for (int i = 0; i < N; i++) {
        //     printf("Row %d: ", i);
        //     for (int j = 0; j < N; j++) {
        //         printf("%d ", global_B[i * N + j]);
        //     }
        //     printf("\n");
        // }
    }

    double start_time = MPI_Wtime();

    if (rank_world != 0) {
        global_A.resize(N * N);
        global_B.resize(N * N);
    }
    MPI_Bcast(global_A.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_L; i++) {
        int global_i = coords[0] * local_L + i;
        for (int j = 0; j < local_L; j++) {
            int global_j = coords[1] * local_L + j;
            A[i * local_L + j] = global_A[global_i * N + global_j];
            B[i * local_L + j] = global_B[global_i * N + global_j];
        }
    }

    // printf("Process %d (coords %d,%d): Local Matrix A:\n", rank, coords[0], coords[1]);
    // for (int i = 0; i < local_L; i++) {
    //     printf("Row %d: ", i);
    //     for (int j = 0; j < local_L; j++) {
    //         printf("%d ", A[i * local_L + j]);
    //     }
    //     printf("\n");
    // }
    // printf("Process %d (coords %d,%d): Local Matrix B:\n", rank, coords[0], coords[1]);
    // for (int i = 0; i < local_L; i++) {
    //     printf("Row %d: ", i);
    //     for (int j = 0; j < local_L; j++) {
    //         printf("%d ", B[i * local_L + j]);
    //     }
    //     printf("\n");
    // }

    int num_blocks = N / b;

    std::vector<int> temp_A(local_L * b);
    std::vector<int> temp_B(b * local_L);


    // SUMMA
    for (int kk = 0; kk < num_blocks; kk++) {
        int global_k = kk * b;

        int owner_col = (global_k / local_L);
        int owner_row = (global_k / local_L);

        int root_col = owner_col % sqrt_p; 
        int root_row = owner_row % sqrt_p; 

        if (coords[1] == root_col) {
            int local_k = global_k % local_L;
            for (int i = 0; i < local_L; i++) {
                for (int j = 0; j < b; j++) {
                    int source_col = local_k + j;
                    if (source_col < local_L) {
                        temp_A[i * b + j] = A[i * local_L + source_col];
                    } else {
                        temp_A[i * b + j] = 0;
                    }
                }
            }

            //printf("Process %d (coords %d,%d) broadcasting A block at kk=%d in row_comm\n", rank, coords[0], coords[1], kk);
        }


        MPI_Bcast(temp_A.data(), local_L * b, MPI_INT, root_col, row_comm);

        if (coords[0] == root_row) {
            int local_k = global_k % local_L;
            for (int i = 0; i < b; i++) {
                int source_row = local_k + i;
                if (source_row < local_L) {
                    for (int j = 0; j < local_L; j++) {
                        temp_B[i * local_L + j] = B[source_row * local_L + j];
                    }
                } else {
                    for (int j = 0; j < local_L; j++) {
                        temp_B[i * local_L + j] = 0;
                    }
                }
            }
            //printf("Process %d (coords %d,%d) broadcasting B block at kk=%d in col_comm\n", rank, coords[0], coords[1], kk);
        }

        MPI_Bcast(temp_B.data(), b * local_L, MPI_INT, root_row, col_comm);

        //printf("Process %d (coords %d,%d): Received temp_A at kk=%d:\n", rank, coords[0], coords[1], kk);
        // for (int i = 0; i < local_L; i++) {
        //     printf("Row %d: ", i);
        //     for (int j = 0; j < b; j++) {
        //         printf("%d ", temp_A[i * b + j]);
        //     }
        //     printf("\n");
        // }

       // printf("Process %d (coords %d,%d): Received temp_B at kk=%d:\n", rank, coords[0], coords[1], kk);
        // for (int i = 0; i < b; i++) {
        //     printf("Row %d: ", i);
        //     for (int j = 0; j < local_L; j++) {
        //         printf("%d ", temp_B[i * local_L + j]);
        //     }
        //     printf("\n");
        // }

        for (int i = 0; i < local_L; i++) {
            for (int j = 0; j < local_L; j++) {
                for (int k = 0; k < b; k++) {
                    C[i * local_L + j] += temp_A[i * b + k] * temp_B[k * local_L + j];
                }
            }
        }

    }

    std::vector<int> recvbuf_C;
    if (rank_world == 0) {
        recvbuf_C.resize(local_size * size);
    }
    MPI_Gather(C.data(), local_size, MPI_INT, recvbuf_C.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double local_execution_time = end_time - start_time;

    if (rank_world == 0) {
        for (int p = 0; p < size; p++) {
            int src_coords[2];
            MPI_Cart_coords(cart_comm, p, 2, src_coords);
            int src_row = src_coords[0];
            int src_col = src_coords[1];

            for (int i = 0; i < local_L; i++) {
                int global_i = src_row * local_L + i;
                for (int j = 0; j < local_L; j++) {
                    int global_j = src_col * local_L + j;
                    global_C[global_i * N + global_j] = recvbuf_C[p * local_size + i * local_L + j];
                }
            }
        }

        printf("Total execution time (T): %f seconds\n", local_execution_time);  
        // printf("\nGlobal Result Matrix C:\n");
        // for (int i = 0; i < N; i++) {
        //     printf("Row %d: ", i);
        //     for (int j = 0; j < N; j++) {
        //         printf("%d ", global_C[i * N + j]);
        //     }
        //     printf("\n");
        // }

        if (verify_result(global_A, global_B, global_C, N)) {
            printf("Matrix multiplication result is correct!\n");
        } else {
            printf("Matrix multiplication result is incorrect!\n");
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}