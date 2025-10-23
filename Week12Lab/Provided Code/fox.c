//////////////////////////////////////////////////////////////////////////////////////
// fox.c
// ----------------------------------------------------------------------------------
//
// Multiplies two square matrices (NxN) and writes the resultant multiplication into a text file.
// Applies a parallel design following Fox Algorithm.
//
// Initial version by: Shageenderan Sapai
//
// Last updated by: Shageenderan Sapai
//
// Last updated date: 31th August 2021
//////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <memory.h>

#define MASTER 0
#define DEBUG 0 // controls whether to print debug lines (1) or not (0)

int n = 4; // Matrix Size

// Helper function to print array
void printarr(int *arr, int n, char *name, int process)
{
    for (int i = 0; i < n; i++)
    {
        if (i == 0)
        {
            printf("[Rank %d] %s: [%d, ", process, name, arr[i]);
            fflush(stdout);
        }
        else if (i == n - 1)
        {
            printf("%d]", arr[i]);
            fflush(stdout);
        }

        else
        {
            printf("%d, ", arr[i]);
            fflush(stdout);
        }
    }
    printf("\n");
}

// Helper function to print a NxN matrix
void printmatrix(int *arr, int n, FILE *out)
{
    for (int row = 0; row < n * n; row++)
    {

        if (row % n == 0 && row != 0)
        {                       //If at end of row, increment row
            fprintf(out, "\n"); //and print new line
        }

        //Print each character
        fprintf(out, "%d ", arr[row]);
        fflush(out);
    }

    fprintf(out, "\n\n");
    fflush(out);
    return;
}

int main(int argc, char *argv[])
{
    int my_rank, size;

    int i = 0, j = 0;

    int *pMatrix_A = NULL;
    int *pMatrix_B = NULL;
    unsigned long long *pMatrix_C = NULL;

    // Read new different matrix size if needed
    if (argc > 1)
        n = strtol(argv[1], NULL, 10);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (my_rank == MASTER)
    {
        if (DEBUG)
        {
            printf("DEBUG is on\n");
        }
        else
        {
            printf("DEBUG is off\n");
        }
        
        // 1. Read Matrix A
        printf("Reading Matrix A - Start\n");

        FILE *pFileA = fopen("MA.txt", "r");

        pMatrix_A = (int *)malloc((n * n) * sizeof(int));
        int offset = 0;
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                offset = (i * n) + j;
                fscanf(pFileA, "%d", &pMatrix_A[offset]);
            }
        }
        fclose(pFileA);

        printf("Reading Matrix A - Done\n");

        // 2. Read Matrix B
        printf("Reading Matrix B - Start\n");

        FILE *pFileB = fopen("MB.txt", "r");

        pMatrix_B = (int *)malloc((n * n) * sizeof(int));
        offset = 0;
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                offset = (i * n) + j;
                fscanf(pFileB, "%d", &pMatrix_B[offset]);
            }
        }
        fclose(pFileB);

        printf("Reading Matrix B - Done\n");

        if (DEBUG)
        {
            printf("A:\n");
            printmatrix(pMatrix_A, n, stdout);
            printf("B:\n");
            printmatrix(pMatrix_B, n, stdout);
        }
    }

    int q = (int)pow(size, 0.5); // Process grid size (sqrt(p))
    int nr = n / q;              // Size of each submatrix

    // define a & b
    int *a = (int *)malloc(n * n * sizeof(int));
    int *b = (int *)malloc(n * n * sizeof(int));

    // If n is not divisible by root p, end program
    if (n % q != 0)
    {
        MPI_Finalize();
        fprintf(stderr, "Square Root of Processes does not divide Number of elements.\n");
        return 0;
    }

    // Reshape Matrix A and B
    if (my_rank == MASTER)
    {
        /* Reshape matrix so each block can be sent to the correct process.
        ----------------------------------------------------------------
         0  1  |  2  3          P1:  0  1  4  5
        _4__5__|__6__7_   ==>   P2:  2  3  6  7
         8  9  | 10 11    ==>   P3:  8  9 12 13
        12 13  | 14 15          P4: 10 11 14 15
        ----------------------------------------------------------------
        */
        int i = 0;
        int k = 0;
        for (int row_num = 0; row_num < q; row_num++)
        {
            for (int col_num = 0; col_num < q; col_num++)
            {
                k = col_num * nr + row_num * n * nr;
                for (int j = 0; j < nr * nr; j++)
                {
                    if (j % nr == 0 && j != 0)
                    {
                        k = k + (n - nr);
                    }

                    // Element Extraction
                    a[i] = pMatrix_A[k];

                    k++;
                    i++;
                }
            }
        }

        /* Reshape matrix so each block can be sent to the correct process.
        ----------------------------------------------------------------
        0  1  |  2  3          P1:  0  1  4  5
        _4__5__|__6__7_   ==>   P2:  2  3  6  7
        8  9  | 10 11    ==>   P3:  8  9 12 13
        12 13  | 14 15          P4: 10 11 14 15
        ----------------------------------------------------------------
        */
        i = 0;
        k = 0;
        for (int row_num = 0; row_num < q; row_num++)
        {
            for (int col_num = 0; col_num < q; col_num++)
            {
                k = col_num * nr + row_num * n * nr;
                for (int j = 0; j < nr * nr; j++)
                {
                    if (j % nr == 0 && j != 0)
                    {
                        k = k + (n - nr);
                    }

                    //Element Extraction
                    b[i] = pMatrix_B[k];

                    k++;
                    i++;
                }
            }
        }
        if (DEBUG)
        {
            printf("Reordered A:\n");
            printmatrix(a, n, stdout);
            printf("Reordered B:\n");
            printmatrix(b, n, stdout);
        }
    }

    // define local A, local B & local C
    int *local_a = (int *) malloc(nr * nr * sizeof(int));
    int *local_b = (int *) malloc(nr * nr * sizeof(int));
    int *local_c = (int *)calloc(nr * nr, sizeof(int)); 

    // Scatter the reshape matrices across the comm world
    MPI_Scatter(a, (nr * nr), MPI_INT, local_a, (nr * nr), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, (nr * nr), MPI_INT, local_b, (nr * nr), MPI_INT, 0, MPI_COMM_WORLD);

    double start, end;

    start = MPI_Wtime();

    int rank = 0;
    if (DEBUG)
    {
        while (rank != size)
        {
            if (my_rank == rank)
            {
                printf("Rank %d Partitions:\n", my_rank);
                printf("A:\n");
                printmatrix(local_a, nr, stdout);
                printf("B:\n");
                printmatrix(local_b, nr, stdout);
            }
            rank++;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    int my_col_rank, my_row_rank;
    int dims[2] = {q, q};           // no. processes in each dimension
    int periods[2] = {1, 1};        // periodic so they loop around

    int reorder = 1;

    // Create a NxN GRID, ROW and COL communicators.
    MPI_Comm GRID_COMM, ROW_COMM, COL_COMM;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &GRID_COMM);

    int grid_rank;
    MPI_Comm_rank(GRID_COMM, &grid_rank);

    // Get my coordinates in the new communicator
    int my_coords[2];
    MPI_Cart_coords(GRID_COMM, grid_rank, 2, my_coords);

    int my_row = my_coords[0];  // row coords
    int my_col = my_coords[1];  // col coords
    
    int free_coords[2];
    // Create ROW Communicator
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(GRID_COMM, free_coords, &ROW_COMM);

    // Create COL Communicator
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(GRID_COMM, free_coords, &COL_COMM);

    MPI_Comm_rank(ROW_COMM, &my_row_rank);  // get rank of process in ROW communicator
    MPI_Comm_rank(COL_COMM, &my_col_rank);  // get rank of process in COL communicator

    if (DEBUG)
    {
        rank = 0;
        while (rank != size)
        {
            if (my_rank == rank)
            {
                printf("[MPI process %d|row rank: %d|col rank: %d]\n", my_rank, my_row_rank, my_col_rank);
                if (rank == size-1) printf("\n");
                fflush(stdout);
            }
            rank++;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // FOX ALGORITHM
    for (int i = 0; i < q; i++)
    {
        // Broadcast Ai block to all processors in the same row group
        int count = nr * nr;
        int *new_block = (int *)malloc(count * sizeof(int));

        if (my_rank == my_row * q + (my_row + i) % q)
        {
            memcpy(new_block, local_a, count * sizeof(int));
        }

        int root = (my_row + i % q) % q;
        MPI_Bcast(new_block, count, MPI_INT, root, ROW_COMM);

        if (DEBUG)
        {
            rank = 0;
            while (rank != size)
            {
                if (my_rank == rank)
                {
                    printf("Process: %d Ai-Block:\n", my_rank);
                    fflush(stdout);
                    printmatrix(new_block, nr, stdout);
                }
                rank++;
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        // Local Matrix Multiplication
        for (int j = 0; j < nr; j++)
        {
            for (int k = 0; k < nr; k++) 
            {
                for (int h = 0; h < nr; h++)
                {
                    local_c[j * nr + k] = local_c[j * nr + k] + new_block[h + j * nr] * local_b[k + h * nr];
                }
            }
        }

        if (DEBUG)
        {
            rank = 0;
            while (rank != size)
            {
                if (my_rank == rank)
                {
                    printf("Process: %d After Local Multiplication\n", my_rank);
                    printmatrix(local_c, nr, stdout);
                }
                rank++;
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        // Shift resident B block up
        int incoming = (q + my_col_rank + 1) % q;   // rank of process to receive the new B block from
        int outgoing = (q + my_col_rank - 1) % q;   // rank of process to send my current B block to

        MPI_Status status;

        MPI_Sendrecv_replace(local_b, nr * nr, MPI_INT, outgoing, 0, incoming, 0, COL_COMM, &status);

        if (DEBUG)
        {
            rank = 0;
            while (rank != size)
            {
                if (my_rank == rank)
                {
                    printf("Process: %d Local B\n", my_rank);
                    printmatrix(local_b, nr, stdout);
                }
                rank++;
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
    }

    // Gather All
    int *row_result = (int *)calloc(sizeof(int), n * nr);
    int *result = (int *)calloc(sizeof(int), n * n);

    // Gather Row Results
    for (int i = 0; i < nr; i++)
    {
        MPI_Gather(local_c + i * nr, nr, MPI_INT, row_result + i * n, nr, MPI_INT, 0, ROW_COMM);
    }

    // Gather Column Results
    MPI_Gather(row_result, n * nr, MPI_INT, result, n * nr, MPI_INT, 0, COL_COMM);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == MASTER)
    printf("Fox Matrix Multiplication - Process time (s): %lf\n", end-start);

    // Output result either to stdout or output file (output)
    if (my_rank == 0)
    {
        FILE *output = fopen("Fox_MC.txt", "w");  
        printf("DONE\n");
        printmatrix(result, n, output);
        fclose(output);
    }

    MPI_Finalize();
}