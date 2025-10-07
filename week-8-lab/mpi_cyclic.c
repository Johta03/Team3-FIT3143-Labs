#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

int comp(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

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

int main(int argc, char **argv){
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n = 0;
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (rank == 0) {
        n = atoi(argv[1]);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_primes = malloc((n/size + size) * sizeof(int)); // Allocate enough space
    int index = 0;
    
    // Handle the case for 2 separately
    if (n >= 2 && rank == 0) {
        local_primes[index++] = 2; // Include 2 if in range
    }

    // round robin distribution of work
    // Starts from 3, checks only odd numbers (even numbers besides 2 are not prime trivially)
    // Eg. 3 processes: P0 -> 3,9,15,...; P1 -> 5,11,17,...; P2 -> 7,13,19,...
    for (int i = 3 + 2 * rank; i < n; i += 2 * size) {
        if (is_prime(i)) {
            local_primes[index++] = i;
        }
    }
    int local_count = index;
    int *all_counts = NULL;
    int *displacements = NULL;
    int total_primes = 0;
    int *global_primes = NULL;

    if (rank == 0) {
        all_counts = malloc(size * sizeof(int));
        displacements = malloc(size * sizeof(int));
    }
    
    // Gather counts of primes from all processes
    MPI_Gather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements and total primes on root process
    if (rank == 0) {
        displacements[0] = 0;
        total_primes = all_counts[0];
        for (int i = 1; i < size; i++) {
            displacements[i] = displacements[i - 1] + all_counts[i - 1];
            total_primes += all_counts[i];
        }
        global_primes = malloc(total_primes * sizeof(int));
    }

    // Gather all primes to root process
    MPI_Gatherv(local_primes, local_count, MPI_INT, 
        global_primes, all_counts, displacements, MPI_INT, 
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        qsort(global_primes, total_primes, sizeof(int), comp);
        FILE *fptr = fopen("coretask.txt", "w");
        for (int i = 0; i < total_primes; i++) {
            fprintf(fptr, "%d\n", global_primes[i]);
        }
        fclose(fptr);
        free(all_counts);
        free(displacements);
        free(global_primes);

        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        printf("Total primes: %d\n", total_primes);
        printf("Elapsed time: %f seconds\n", elapsed);
    }
    free(local_primes);
    MPI_Finalize();

    return 0;
}