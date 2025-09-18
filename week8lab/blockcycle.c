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
    if (a == 2) return true;
    if (a <= 1 || a % 2 == 0) return false;

    int mid = sqrt(a);
    for (int i = 3; i <= mid; i += 2) {
        if (a % i == 0) return false;
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

    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <n>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    // Broadcast n to everyone
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Block size for block-cyclic distribution
    const int block_size = 100;

    int *local_primes = malloc((n/size + block_size) * sizeof(int));
    int index = 0;

    // Handle 2 separately
    if (rank == 0 && n >= 2) {
        local_primes[index++] = 2;
    }

    // Assign chunks of size `block_size` in a round-robin manner
    for (int start_num = 3 + rank * block_size;
         start_num < n;
         start_num += block_size * size) {

        int end_num = start_num + block_size;
        if (end_num > n) end_num = n;

        // Only check odd numbers
        for (int i = (start_num % 2 == 0 ? start_num + 1 : start_num);
             i < end_num;
             i += 2) {
            if (is_prime(i)) {
                local_primes[index++] = i;
            }
        }
    }

    int local_count = index;
    int *all_counts = NULL, *displacements = NULL;
    int total_primes = 0;
    int *global_primes = NULL;

    if (rank == 0) {
        all_counts = malloc(size * sizeof(int));
        displacements = malloc(size * sizeof(int));
    }

    // Gather counts from all processes
    MPI_Gather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displacements[0] = 0;
        total_primes = all_counts[0];
        for (int i = 1; i < size; i++) {
            displacements[i] = displacements[i-1] + all_counts[i-1];
            total_primes += all_counts[i];
        }
        global_primes = malloc(total_primes * sizeof(int));
    }

    // Gather actual primes
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

        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1e9;

        printf("Total primes: %d\n", total_primes);
        printf("Elapsed time: %f seconds\n", elapsed);

        free(all_counts);
        free(displacements);
        free(global_primes);
    }

    free(local_primes);
    MPI_Finalize();
    return 0;
}
