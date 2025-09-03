#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int size, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the name of the processor
    MPI_Get_processor_name(processor_name, &namelen);
    
    // Print a hello world message from each process
    printf("Hello world from process %d of %d\n",
           rank, size);

    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}