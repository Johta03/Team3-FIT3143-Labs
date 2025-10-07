// MPI Block Prime Search - Lab 4 Week 8
// Each process works on an equally distributed contiguous block of [2, n)

#include <mpi.h>   // MPI_Init, MPI_Comm_rank/size, MPI_Bcast, MPI_Gather/Gatherv
#include <stdio.h> // printf, FILE*, fopen, fprintf, fclose
#include <stdlib.h>// atoi, malloc, free
#include <math.h>  // sqrt

// Check if number is prime using trial division up to floor(sqrt(n))
int is_prime(int n) {
    if (n < 2) return 0;        // reject 0 and 1
    if (n == 2) return 1;       // 2 is prime
    if (n % 2 == 0) return 0;   // reject even numbers > 2
    
    int mid = sqrt(n);          // largest divisor worth testing
    for (int i = 3; i <= mid; i += 2) { // test odd divisors only
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);                 // start MPI runtime
    
    int rank, size;                         // this process id, number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get n from command line (process 0 only)
    int n = 0;
    if (rank == 0) {
        if (argc < 2) {
            printf("Usage: mpirun -np <procs> ./mpi_block <n>\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);                 // parse upper bound
    }
    
    // Send n to all processes so everyone knows the upper bound
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Start timing (includes local compute and gather on root)
    double start = MPI_Wtime();
    
    // BLOCK DISTRIBUTION
	// Each rank gets a contiguous slice of [2, n) (no overlaps, no gaps)
    
    // Compute slice for this rank (inclusive start, exclusive end)
    int total = n - 2;                              // numbers 2..n-1
    int start_range = 2 + (rank * total) / size;    // inclusive
    int end_range   = 2 + ((rank + 1) * total) / size; // exclusive
    int range = end_range - start_range;

    int *my_primes = malloc(range * sizeof(int));   // worst-case capacity
    int my_count = 0;                                // number of local primes
    
    for (int i = start_range; i < end_range; i++) {  // scan contiguous slice
        if (is_prime(i)) {
            my_primes[my_count] = i;
            my_count++;

        }
    }
    
    // Gather counts from all processes so root can size the receive buffer
    int *counts = NULL;
    if (rank == 0) counts = malloc(size * sizeof(int));

    MPI_Gather(&my_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate total size and displacements for Gatherv on root
    int total = 0;
    int *displs = NULL;
    if (rank == 0) {
        displs = malloc(size * sizeof(int)); // starting offsets per rank
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            if (i > 0) displs[i] = displs[i-1] + counts[i-1];
            total += counts[i];
        }
    }
   
    // Gather all primes into root; blocks concatenate in global ascending order
    int *all_primes = NULL;
    if (rank == 0) all_primes = malloc(total * sizeof(int));
    MPI_Gatherv(my_primes, my_count, MPI_INT, all_primes, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Process 0 writes results and reports timing
    if (rank == 0) {
        double end = MPI_Wtime();
        double time = end - start; 
        FILE *f = fopen("primes.txt", "w");
        if (!f) {
            printf("Error: Cannot open output file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Write all primes to file (already globally sorted by block concatenation)
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
    
    MPI_Finalize(); // finalize MPI runtime
    return 0;
}