#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <stdlib.h>


// https://stackoverflow.com/questions/77122903/passing-multiple-arguments-to-pthread-including-vector-arguments
typedef struct {
    int from;
    int to;
} RangeArg;

bool is_prime(int a){
    if (a == 2) {return true;}
    if (a <= 1 | a % 2 == 0) {return false;}

    int i;
    int mid = sqrt(a);
    for (i = 3; i <= mid; i += 2){
        if (a % i == 0) {return false;}
    }

    return true;
}

void *check_for_prime(void* arg){
    RangeArg* args = (RangeArg*)arg;
    int from = args -> from;
    int to = args -> to;
    int i;
    FILE *fptr;
    fptr = fopen("task2.txt", "a");
    int* count = malloc(sizeof(int));
    *count = 0;
    for (i = from; i < to; i += 1){
        if (is_prime(i)){
            //printf("%d\n", i);
            fprintf(fptr, "%d\n", i);

            (*count)++;
        }
    }
    fclose(fptr); 
    return count;
}

void split(int from, int to, int count, int *arr) {
    int total = to - from;
    int base = total / count;
    int remainder = total % count;

    arr[0] = from;
    for (int i = 1; i <= count; i++) {
        arr[i] = arr[i - 1] + base + (i <= remainder ? 1 : 0);
    }
}

int main() {
    FILE *fptr;
    fptr = fopen("task2.txt", "w");
    fclose(fptr);

    // https://stackoverflow.com/questions/2962785/c-using-clock-to-measure-time-in-multi-threaded-programs
    struct timespec start, finish;
    double elapsed; 

    clock_gettime(CLOCK_MONOTONIC, &start);

    int a = 0;
    int b = 10000000;

    int thread_count = 16; // how many threads we want
    int arr[thread_count+1];
    split(a, b, thread_count, arr);
    
    int i;
        pthread_t tid[thread_count];
    RangeArg args[thread_count];  // Array of structs
    for (i = 0; i < thread_count; i++){
        args[i].from = arr[i];
        args[i].to = arr[i + 1];
        pthread_create(&tid[i], NULL, check_for_prime, &args[i]);
    }

    int total_count = 0;

    for (int i = 0; i < thread_count; i++) {
        void* ret;
        pthread_join(tid[i], &ret);
        if (ret) {
            total_count += *(int*)ret;
            free(ret);
        }
    }
    

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("\nCOUNT: %d", total_count);
    printf("\n %f seconds", elapsed);
    printf("\nTHREADS: %d\n", thread_count);



    fptr = fopen("task2.txt", "a");
    fprintf(fptr, "\n %f seconds", elapsed);
    fprintf(fptr, "\nCOUNT: %d", total_count);
    fprintf(fptr, "\nTHREADS: %d", thread_count);
    fclose(fptr);
}