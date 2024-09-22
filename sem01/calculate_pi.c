#include <stdio.h> 
#include <stdlib.h> 
#include <pthread.h> 
#include <math.h> 
#include <sys/time.h> 

int num_intervals; 
int num_threads; 
double sum = 0.0; 
pthread_mutex_t mutex; 


void* calculate_pi(void* arg) { 
	int thread_id = *(int*)arg; 
	double local_sum = 0.0; 
	double width = 1.0 / num_intervals; 

	for (int i = thread_id; i < num_intervals; i += num_threads) { 
		double x = (i + 0.5) * width; 
		local_sum += 4.0 / (1.0 + x * x); 
	} 


	pthread_mutex_lock(&mutex); 
	sum += local_sum; 
	pthread_mutex_unlock(&mutex); 

	return NULL; 
} 

int main(int argc, char* argv[]) { 

	if (argc != 3) { 
		printf("Usage: %s <num_intervals> <num_threads>\n", argv[0]); 
		return 1; 
	} 

	num_intervals = atoi(argv[1]); 
	num_threads = atoi(argv[2]); 
	pthread_t threads[num_threads]; 
	int thread_ids[num_threads]; 

	pthread_mutex_init(&mutex, NULL); 

	struct timeval start, end; 
	gettimeofday(&start, NULL); 

	for (int i = 0; i < num_threads; i++) { 
		thread_ids[i] = i; 
		pthread_create(&threads[i], NULL, calculate_pi, &thread_ids[i]); 
	} 

	for (int i = 0; i < num_threads; i++) { 
		pthread_join(threads[i], NULL); 	
	} 

	double pi = sum * (1.0 / num_intervals); 

	gettimeofday(&end, NULL); 
	double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0; 

	printf("Estimated value of pi: %.15f\n", pi); 
	printf("Elapsed time: %.6f seconds\n", elapsed_time); 


	pthread_mutex_destroy(&mutex); 
	return 0; 
} 

