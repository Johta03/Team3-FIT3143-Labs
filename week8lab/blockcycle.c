#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
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

    // === Timing variables ===
    double total_start, total_end;
    double comp_time = 0.0, comm_time = 0.0, ser_time = 0.0;

    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <n>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
    }

    total_start = MPI_Wtime(); // start total timing

    // Broadcast n (communication)
    double t0 = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    const int block_size = 100;
    int *local_primes = malloc((n/size + block_size) * sizeof(int));
    int index = 0;

    // Handle 2 separately (serial, but negligible)
    if (rank == 0 && n >= 2) {
        local_primes[index++] = 2;
    }

    // === Compute portion ===
    double comp_start = MPI_Wtime();
    for (int start_num = 3 + rank * block_size;
         start_num < n;
         start_num += block_size * size) {

        int end_num = start_num + block_size;
        if (end_num > n) end_num = n;

        for (int i = (start_num % 2 == 0 ? start_num + 1 : start_num);
             i < end_num;
             i += 2) {
            if (is_prime(i)) {
                local_primes[index++] = i;
            }
        }
    }
    comp_time = MPI_Wtime() - comp_start;

    int local_count = index;
    int *all_counts = NULL, *displacements = NULL;
    int total_primes = 0;
    int *global_primes = NULL;

    if (rank == 0) {
        all_counts = malloc(size * sizeof(int));
        displacements = malloc(size * sizeof(int));
    }

    // === Communication: Gather counts ===
    t0 = MPI_Wtime();
    MPI_Gather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    if (rank == 0) {
        displacements[0] = 0;
        total_primes = all_counts[0];
        for (int i = 1; i < size; i++) {
            displacements[i] = displacements[i-1] + all_counts[i-1];
            total_primes += all_counts[i];
        }
        global_primes = malloc(total_primes * sizeof(int));
    }

    // === Communication: Gather primes ===
    t0 = MPI_Wtime();
    MPI_Gatherv(local_primes, local_count, MPI_INT,
                global_primes, all_counts, displacements, MPI_INT,
                0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    // === Root serial work (sorting + writing file) ===
    if (rank == 0) {
        double ser_start = MPI_Wtime();

        qsort(global_primes, total_primes, sizeof(int), comp);

        FILE *fptr = fopen("coretask.txt", "w");
        for (int i = 0; i < total_primes; i++) {
            fprintf(fptr, "%d\n", global_primes[i]);
        }
        fclose(fptr);

        ser_time = MPI_Wtime() - ser_start;
    }

    total_end = MPI_Wtime();

    // === Reduce times to root ===
    double comp_max, comm_max;
    MPI_Reduce(&comp_time, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_time = total_end - total_start;

        printf("\n=== Timing Breakdown (p=%d) ===\n", size);
        printf("Total primes found: %d\n", total_primes);
        printf("Total time    : %.6f s\n", total_time);
        printf("Compute time  : %.6f s (max across ranks)\n", comp_max);
        printf("Comm time     : %.6f s (max across ranks)\n", comm_max);
        printf("Serial (root) : %.6f s\n", ser_time);
    }

    free(local_primes);
    if (rank == 0) {
        free(all_counts);
        free(displacements);
        free(global_primes);
    }

    MPI_Finalize();
    return 0;
}
