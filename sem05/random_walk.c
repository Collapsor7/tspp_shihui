#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int random_walk(int a, int b, int x, double p, int *steps);
void simulate_walks(int a, int b, int x, double p, int N, int P, double *reach_b_prob, double *avg_time);

int main(int argc, char *argv[]) {
    
    if (argc < 7) {
        printf("Usage: %s <a> <b> <p> <x> <N> <P>\n", argv[0]);
        return 1;
    }

    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    double p = atof(argv[3]);
    int x = atoi(argv[4]);
    int N = atoi(argv[5]);
    int P = atoi(argv[6]);
     
    omp_set_num_threads(1);

    double reach_b_prob = 0.0, avg_time = 0.0;
    
    double start_time = omp_get_wtime();
    simulate_walks(a, b, x, p, N, 1, &reach_b_prob, &avg_time);
    double end_time = omp_get_wtime();
    double T_seq = end_time-start_time;
    omp_set_num_threads(P);

    reach_b_prob = 0.0, avg_time = 0.0;
    
    start_time = omp_get_wtime();
    simulate_walks(a, b, x, p, N, P, &reach_b_prob, &avg_time);
    end_time = omp_get_wtime();
    double T_par = end_time - start_time;    
    
    
    double S = T_seq / T_par;
    double E = S / P;

    printf("Probability of reaching b: %f\n", reach_b_prob);
    printf("Average time of walk: %f\n", avg_time);
    printf("Parallel execution time (T): %f seconds\n", T_par);
    printf("Speed (S): %f\n", S);
    printf("Parallel efficiency (E): %f\n", E);

    return 0;
}

int random_walk(int a, int b, int x, double p, int *steps) {
    *steps = 0;
    while (x > a && x < b) {
        x += (rand() / (double)RAND_MAX) < p ? 1 : -1;
        (*steps)++;
    }
    return x >= b;
}


void simulate_walks(int a, int b, int x, double p, int N, int P, double *reach_b_prob, double *avg_time) {
    int count_b = 0;
    double total_time = 0.0;
    
    #pragma omp parallel for num_threads(P) reduction(+:count_b, total_time)
    for (int i = 0; i < N; i++) {
        int steps = 0;
        int reached_b = random_walk(a, b, x, p, &steps);
        total_time += steps;
        
        if (reached_b) {
            count_b++;
        }
    }

    *reach_b_prob = (double)count_b / N;
    *avg_time = total_time / N;
}
