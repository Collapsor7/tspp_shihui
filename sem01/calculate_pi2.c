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

double* run_pi_calculate(int intervals,int threads){
	num_intervals=intervals;
	num_threads=threads;
	sum=0.0;
	pthread_t thread_ids[num_threads]; 
	int thread_nums[num_threads]; 
	
	pthread_mutex_init(&mutex, NULL); 
	
	struct timeval start, end; 
	gettimeofday(&start, NULL); 

	for (int i = 0; i < num_threads; i++) { 
		thread_nums[i] = i; 
		pthread_create(&thread_ids[i], NULL, calculate_pi, &thread_nums[i]); 
	} 

	for (int i = 0; i < num_threads; i++) { 
		pthread_join(thread_ids[i], NULL); 
	} 

	gettimeofday(&end, NULL); 
	double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0; 
	
	double pi = sum * (1.0 / num_intervals); 

	pthread_mutex_destroy(&mutex);
	
	double *arr = (double*)malloc(2*sizeof(double));
	arr[0]=pi;
	arr[1]=elapsed_time;
	
	return  arr;
}
int main() { 
	FILE *fptr = fopen("pi_results.csv","w");
	
	if(fptr == NULL){
		printf("Error open file!");
		return 1;
	}
	
	fprintf(fptr,"NumIntervals,NumThreads,PiValue,ElaspsedTime(s)\n");
	
	int choose_intervals[] = {100000000,200000000,300000000,400000000,500000000,600000000,700000000,800000000,900000000,1000000000};
	//int choose_intervals[] = {10000000};
	
	for(int i=0;i<10;i++){
		for (int threads=1;threads<=20;threads++){
			double *results= run_pi_calculate(choose_intervals[i],threads);
			printf("Intervals: %d,Threads:%d,pi:%.15f,Time:%.06f s\n",choose_intervals[i],threads,results[0],results[1]);
			fprintf(fptr,"%d,%d,%.15f,%.06f\n",choose_intervals[i],threads,results[0],results[1]);
			fflush(fptr);
			free(results);
		}
	}
	
	fclose(fptr);
	printf("Results are saved to file\n");
	return 0; 
} 

