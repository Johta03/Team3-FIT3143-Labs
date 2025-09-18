// MPI Block Prime Search - Lab 4 Week 8
// Each process works on equal distributed chunks/blocks

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Check if number is prime using trial division
int is_prime(int n) {
    if (n < 2) return 0;        // 0 and 1 are not prime
    if (n == 2) return 1;       // 2 is prime
    if (n % 2 == 0) return 0;   // Even numbers > 2 are not prime
    
    int mid = sqrt(n);
    for (int i = 3; i <= mid; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get n from command line (process 0 only)
    int n = 0;
    if (rank == 0) {
        if (argc < 2) {
            printf("Usage: mpirun -np <procs> ./prime_mpi <n>\n");;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
    }
    
    // Send n to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Start timing
    double start = MPI_Wtime();
    
    // BLOCK DISTRIBUTION
	// Pattern: Each processor evenly distributes workload by chunking the load.
    
    int start_range = 2+(rank * n) / size;
    int end_range  = 2+((rank + 1) * n) / size;
    int range = end_range - start_range;

    int *my_primes = malloc(range * sizeof(int));
    int my_count = 0;
    
    for (int i = start_range; i < end_range; i++) {
        if (is_prime(i)) {
            my_primes[my_count] = i;
            my_count++;

        }
    }
    
    // Gather counts from all processes
    int *counts = NULL;
    if (rank == 0) counts = malloc(size * sizeof(int));

    MPI_Gather(&my_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate total and displacements
    int total = 0;
    int *displs = NULL;
    if (rank == 0) {
        displs = malloc(size * sizeof(int));
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            if (i > 0) displs[i] = displs[i-1] + counts[i-1];
            total += counts[i];
        }
    }
   
    // Gather all primes
    int *all_primes = NULL;
    if (rank == 0) all_primes = malloc(total * sizeof(int));
    MPI_Gatherv(my_primes, my_count, MPI_INT, all_primes, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Process 0 writes results
    if (rank == 0) {
        double end = MPI_Wtime();
        double time = end - start; 
        char filepath[256];
        snprintf(filepath, sizeof(filepath), "/primes.txt");
        FILE *f = fopen("primes.txt", "w");
        if (!f) {
            printf("Error: Cannot open output file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Write all primes to file (already in sorted order due to cyclic distribution)
        for (int i = 0; i < total; i++) {
            fprintf(f, "%d\n", all_primes[i]);
        }
        fclose(f);
        
        printf("n=%d, processes=%d, primes=%d, time=%.6f s\n", 
               n, size, total, time);
    }
    
    // Cleanup
    free(my_primes);
    if (rank == 0) {
        free(counts);
        free(displs);
        free(all_primes);
    }
    
    MPI_Finalize();
    return 0;
}