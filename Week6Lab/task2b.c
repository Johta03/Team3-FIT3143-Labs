// Task 2b: Use MPI_Bcast to send a message (in)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size, value;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get the rank and size of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Main loop - keep reading until negative input
    while (1) {
        if (rank == 0) {
            // Process 0 reads input from user
            printf("Enter an integer (negative to exit): ");
            fflush(stdout);
            scanf("%d", &value);
        }
        
        // Broadcast the value from process 0 to all processes
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Check if we should exit
        if (value < 0) {
            break;
        }
        
        // All processes print their rank and received value 
        printf("Process Rank: %d received value: %d\n", rank, value);
        fflush(stdout);
        
        // Add barrier to synchronize all processes before next iteration
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
