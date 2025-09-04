#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[]) {
    // Initialise MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialise the number of iterations
    long N = 0;

    if (rank == 0) {
        printf("Enter the number of iterations: ");
        fflush(stdout);
        scanf("%lu", &N);
    }

    // Broadcast the number of iterations to all processors
    MPI_Bcast(&N, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // Partition to work into different chunks
    int start_index = (int) N * rank / size;
    int end_index = (int) N * (rank + 1) / size;

    int i;
    double sum = 0.0;
    double local_sum = 0.0; // Count the pi for each process/chunk
    double piVal;
    struct timespec start, end;
    double time_taken;

    // Sync all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Get current clock time.
    clock_gettime(CLOCK_MONOTONIC, &start);

    // For loop to main algorithm to calculate pi at each chunk
    for (i = start_index; i < end_index; i++) {
        local_sum += 4.0 / (1 + pow((2.0 * i + 1.0) / (2.0 * N), 2));
    }

    // Combine all local sums to the global sum
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    piVal = sum / (double) N;

    // Get the clock current time again
    // Subtract end from start to get the CPU time used.
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Only print the output from the main process
    if (rank == 0) {
        time_taken = (double) (end.tv_sec - start.tv_sec) * 1e9;
        time_taken = (time_taken + (double) (end.tv_nsec - start.tv_nsec)) *
                     1e-9;
        printf("Calculated Pi value (Serial-AlgoI) = %12.9f\n",
               piVal);
        printf("Overall time (s): %lf\n", time_taken); // ts
    }

    return MPI_Finalize();
}