#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THRESHOLD 10000

void merge(int *array, int left, int mid, int right) {
    int i = left, j = mid + 1, k = 0;
    int *temp = (int *)malloc((right - left + 1) * sizeof(int));

    while (i <= mid && j <= right) {
        if (array[i] < array[j]) {
            temp[k++] = array[i++];
        } else {
            temp[k++] = array[j++];
        }
    }
    while (i <= mid) temp[k++] = array[i++];
    while (j <= right) temp[k++] = array[j++];

    for (i = left, k = 0; i <= right; i++, k++) array[i] = temp[k];
    free(temp);
}
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}
void parallel_merge_sort(int *array, int left, int right) {
    if(right - left + 1 <= THRESHOLD){
      qsort(array + left,right - left + 1, sizeof(int), compare);
    }
    else{
        int mid = left + (right - left) / 2;

        #pragma omp task shared(array) if (right - left + 1 > THRESHOLD)
        parallel_merge_sort(array, left, mid);

        #pragma omp task shared(array) if (right - left + 1 > THRESHOLD)
        parallel_merge_sort(array, mid + 1, right);

        #pragma omp taskwait
        merge(array, left, mid, right);
    }
}
int is_sorted(int *array, int N){
  for(int i = 0; i < N-1;i++){
    if (array[i] > array[i+1]){
    return 0;}
  }
  return 1;
}


int main() {

    int N = 10000000;
    int *array = (int *)malloc(N * sizeof(int));
    int *array_copy = (int *)malloc(N * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        array[i] = rand() % 100;
    }
    for (int i = 0; i < N; i++) {
        array_copy[i] = array[i];
    }

    int P_sizes[5] = {1,2,4,8,16};
    double T_sizes[5];
    double S_sizes[5];
    double E_sizes[5];
    
    for(int m = 0; m < 5; m++){
      int P = P_sizes[m];
      omp_set_num_threads(P);
      double start_time = omp_get_wtime();
      #pragma omp parallel
      {
          #pragma omp single
          {
            int chunk_size = N / P;
            for (int n = 0; n < P ;n++){
              int left = n * chunk_size;
              int right = (n == P -1) ? (N - 1) : (left + chunk_size - 1);
              #pragma omp task shared(array)
              parallel_merge_sort(array, left, right);
            }
         
          }
          
      }
      double end_time = omp_get_wtime();
      double T = end_time - start_time;
      T_sizes[m] = T;
      
      if (is_sorted(array,N)){
        printf("Now P = %d,Sorting correct!\n",P);
      }
      else{
        printf("Now P = %d,Sorting incorrect!\n",P);
      }
      for (int i = 0; i < N ; i++){
        array[i] = array_copy[i];
      }
    }
    printf("Sum_List Execution time: ");
    for(int i = 0; i < 5;i++){
      printf("%f ",T_sizes[i]);
    }
    printf("\n");
    for( int i = 0 ;i < 5; i++){
      S_sizes[i]=T_sizes[0] / T_sizes[i];
      E_sizes[i]=S_sizes[i] / (double)(i+1);
    }
    printf("Speed (S):");
    for(int i = 0; i < 5;i++){
      printf("%f ",S_sizes[i]);
    }
    printf("\n");
    printf("Parallel efficiency (E): ");
    for(int i = 0; i < 5;i++){
      printf("%f ",E_sizes[i]);
    }
    printf("\n");

    free(array);
    free(array_copy);
    return 0;
}

