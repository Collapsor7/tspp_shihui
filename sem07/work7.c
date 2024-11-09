#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define N 1024        
#define MAX_ITER 1000
#define TOLERANCE 1e-6


void jacobi_update(double *local_grid, double *next_grid, int local_N, int size_N) {
    for (int i = 1; i <= local_N; i++) {
        for (int j = 1; j < size_N - 1; j++) {
            int idx = i * size_N + j;
            next_grid[idx] = 0.25 * (
                local_grid[(i - 1) * size_N + j] + 
                local_grid[(i + 1) * size_N + j] + 
                local_grid[i * size_N + (j - 1)] + 
                local_grid[i * size_N + (j + 1)] 
            );
        }
    }
}

double compute_norm(double *grid1, double *grid2, int local_N, int size_N) {
    double norm = 0.0;
    for (int i = 1; i <= local_N; i++) {
        for (int j = 0; j < size_N; j++) {
            int idx = i * size_N + j;
            norm += (grid1[idx] - grid2[idx])* (grid1[idx] - grid2[idx]);
        }
    }
    return sqrt(norm);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int local_N;
    int start_row;
    double *local_grid, *next_grid;
    double *temp_grid;
    double norm = 0.0;
    int iter;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base_local_N = N / size;
    int remainder = N % size;
    if (rank < remainder) {
        local_N = base_local_N + 1;
        start_row = rank * local_N;
    } else {
        local_N = base_local_N;
        start_row = rank * local_N + remainder;
    }

    
    local_grid = (double*)malloc((local_N + 2) * N * sizeof(double));
    next_grid = (double*)malloc((local_N + 2) * N * sizeof(double));

    if (local_grid == NULL || next_grid == NULL) {
        fprintf(stderr, "Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    
    for (int i = 1; i <= local_N; i++) {
        for (int j = 0; j < N; j++) {
            local_grid[i * N + j] = rand() / (double)RAND_MAX;
        }
    }

  
    memset(local_grid, 0, N * sizeof(double));
    memset(local_grid + (local_N + 1) * N, 0, N * sizeof(double)); 

    double start_time = MPI_Wtime();

    for (iter = 0; iter < MAX_ITER; iter++) {
        jacobi_update(local_grid, next_grid, local_N, N);

        norm = compute_norm(local_grid, next_grid, local_N, N);

        if (rank > 0) {
            MPI_Send(local_grid + N, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD );
            MPI_Recv(local_grid, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(local_grid + local_N * N, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(local_grid + (local_N + 1) * N, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        //MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        temp_grid = local_grid;
        local_grid = next_grid;
        next_grid = temp_grid;

        if (iter % 100 == 0 && rank == 0) {
            printf("Iteration %d, Norm: %f\n", iter, norm);
        }

        if (norm < TOLERANCE) {
            break;
        }
    }

    double end_time = MPI_Wtime();
    double local_execution_time = end_time - start_time;


    double max_execution_time;
    MPI_Reduce(&local_execution_time, &max_execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total execution time (T): %f seconds\n", max_execution_time);
        printf("Max norm after %d iterations: %f\n", iter, norm);
    }

    free(local_grid);
    free(next_grid);

    MPI_Finalize();

    return 0;
}



