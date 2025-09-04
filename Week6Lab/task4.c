#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
int main(){
    int my_rank;
    struct timespec ts = {0, 50000000L}; /* wait 0 sec and 5^8 nanosec */
    int a; double b;
    char *buffer; int buf_size, buf_size_int, buf_size_double, position =
    0;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Determine buffer size
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &buf_size_int);
    MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &buf_size_double);

    buf_size = buf_size_int + buf_size_double; // Increase the buffer size

    // Allocate memory to the buffer
    buffer = (char *) malloc((unsigned) buf_size);
    do{
        if(my_rank == 0){
        nanosleep (&ts, NULL);
        printf("Enter an round number (>0) & a real number: ");
        fflush(stdout);
        scanf("%d %lf", &a, &b);
        position = 0; // Reset the position in buffer

        // Pack the integer a into the buffer
        // Please insert one line of code here (Hint: a MPI_Pack function call)
        MPI_Pack(&a, 1, MPI_INT, buffer, buf_size, &position, MPI_COMM_WORLD);

        // Pack the double b into the buffer
        // Please insert one line of code here (Hint: another MPI_Pack function call)
        MPI_Pack(&b, 1, MPI_DOUBLE, buffer, buf_size, &position, MPI_COMM_WORLD);

        }
    // Broadcast the buffer to all processes
        MPI_Bcast(buffer, buf_size, MPI_PACKED, 0, MPI_COMM_WORLD);
        position = 0; // Reset the position in buffer in each iteration
        // Unpack the integer a from the buffer
        // Please insert one line of code here (Hint: a MPI_Unpack function call)
        MPI_Unpack(buffer, buf_size, &position, &a, 1, MPI_INT, MPI_COMM_WORLD);

        // Unpack the double b from the buffer
        // Please insert one line of code here (Hint: another MPI_Unpack function call)
        MPI_Unpack(buffer, buf_size, &position, &b, 1, MPI_DOUBLE, MPI_COMM_WORLD);


        printf("[Process %d] Received values: values.a = %d, values.b = %lf\n", my_rank, a, b);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        } while(a > 0);
    /* Clean up */
    free(buffer);
    MPI_Finalize();
    return 0;
}