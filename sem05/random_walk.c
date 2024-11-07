#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int random_walk(int a, int b, int x, double p, int *steps, unsigned int *seed);
void simulate_walks(int a, int b, int x, double p, int N, int P, double *reach_b_prob, double *avg_time);

int main() {
    int a = 0;
    int b = 1000;
    double p = 0.5;
    int x = 500;
    int N_sizes[3] = {1000, 10000, 100000};
    int P_sizes[5] = {1, 2, 4, 8, 16};
    double T_sizes[5];
    double S_sizes[5];
    double E_sizes[5];

    for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 5; n++) {
            printf("now N = %d , P = %d \n", N_sizes[m], P_sizes[n]);
            omp_set_num_threads(P_sizes[n]);

            double reach_b_prob_par = 0.0, avg_time_par = 0.0;
            
            double start_time = omp_get_wtime();
            simulate_walks(a, b, x, p, N_sizes[m], P_sizes[n], &reach_b_prob_par, &avg_time_par);
            double end_time = omp_get_wtime();
            double T = end_time - start_time;
            printf("Probability of reaching b (Sequential): %f\n", reach_b_prob_par);
            printf("Average time of walk (Sequential): %f\n", avg_time_par);
            printf("Parallel execution time (T): %f seconds\n", T);
            
            T_sizes[n] = T;
        }

        printf("---------------------------------------------\n");
        printf("Sum_List Execution time: ");
        for (int i = 0; i < 5; i++) {
            printf("%f ", T_sizes[i]);
        }
        printf("\n");

        for (int i = 0; i < 5; i++) {
            S_sizes[i] = T_sizes[0] / T_sizes[i];
            E_sizes[i] = S_sizes[i] / (double)(i + 1);
        }

        printf("Speed (S):");
        for (int i = 0; i < 5; i++) {
            printf("%f ", S_sizes[i]);
        }
        printf("\n");

        printf("Parallel efficiency (E): ");
        for (int i = 0; i < 5; i++) {
            printf("%f ", E_sizes[i]);
        }
        printf("\n");
    }

    return 0;
}

int random_walk(int a, int b, int x, double p, int *steps, unsigned int *seed) {
    *steps = 0;
    while (x > a && x < b) {
        x += (rand_r(seed) / (double)RAND_MAX) < p ? 1 : -1;
        (*steps)++;
    }
    return x >= b;
}

void simulate_walks(int a, int b, int x, double p, int N, int P, double *reach_b_prob, double *avg_time) {
    int count_b = 0;
    double total_time = 0.0;
    
    unsigned int time_global = (unsigned int)time(NULL);  // 全局时间戳

    #pragma omp parallel for num_threads(P) reduction(+:count_b, total_time)
    for (int i = 0; i < N; i++) {
        // unsigned int seed = time_global + omp_get_thread_num(); 
        int steps = 0;
        int reached_b = random_walk(a, b, x, p, &steps, &seed);
        total_time += steps;
        
        if (reached_b) {
            count_b++;
        }
    }

    *reach_b_prob = (double)count_b / N;
    *avg_time = total_time / N;
}
