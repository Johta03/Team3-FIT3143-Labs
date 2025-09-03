// Task 3: Use an MPI derived datatype to broadcast a struct containing
// an int and a double from rank 0 to all ranks using a single MPI_Bcast.

#include <stdio.h>
#include <mpi.h>

// This struct holds the data we want to share in one go.
// We'll create a matching MPI datatype so we can broadcast the entire struct
// with a single MPI_Bcast call.
struct valuestruct {
    int a;      // integer field (e.g., control value; negative to exit)
    double b;   // double-precision field (e.g., associated real value)
} ;

int main(int argc, char** argv)
{
    struct valuestruct values;        // the instance we send/receive each loop
    int myrank;                       // this process's rank ID
    MPI_Datatype Valuetype;           // derived MPI datatype matching valuestruct
    MPI_Datatype type[2] = { MPI_INT, MPI_DOUBLE }; // base MPI types for each field
    int blocklen[2] = { 1, 1};        // one element for each field
    MPI_Aint disp[2];                 // byte displacements of each field

    MPI_Init(&argc, &argv);                 // start the MPI runtime
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // get this process's rank
    MPI_Get_address(&values.a, &disp[0]);   // address of first field
    MPI_Get_address(&values.b, &disp[1]);   // address of second field

    // Make displacements relative to the first field to handle any padding
    disp[1] = disp[1] - disp[0];
    disp[0] = 0;

    // Create an MPI derived datatype that mirrors 'struct valuestruct'.
    MPI_Type_create_struct(2, blocklen, disp, type, &Valuetype);
    MPI_Type_commit(&Valuetype); // make the new type ready for use

    // Main loop: rank 0 reads input, then we broadcast the struct to everyone.
    do{
        if (myrank == 0){
            // Prompt and read: an integer then a double.
            // A negative integer signals all processes to exit the loop.
            printf("Enter a round number (>0) & a real number: ");
            fflush(stdout);
            scanf("%d%lf", &values.a, &values.b);
        }
        // Broadcast the entire struct from rank 0 to all ranks (including 0).
        MPI_Bcast(&values, 1, Valuetype, 0, MPI_COMM_WORLD);
        // Every rank now has the same 'values'; print to verify.
        printf("Rank: %d. values.a = %d. values.b = %lf\n",
               myrank, values.a, values.b);
        fflush(stdout);
    } while(values.a > 0);

    /* Clean up the type */
    MPI_Type_free(&Valuetype); // free the derived datatype resources
    MPI_Finalize();            // shut down MPI

    return 0;
}
