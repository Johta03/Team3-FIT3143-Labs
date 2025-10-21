///////////////////////////////////////////////////////////////////////////////////////////
// MatrixMul_1D_TransposeParRead_bin.c
// --------------------------------------------------------------------------------------
// 
// Multiplies two matrices and writes the resultant multiplication into a binary file.
// Parallel read from binary files
///////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <fcntl.h>

#define NUM_THREADS 2

void transpose(int *pInMatrix, int rows, int cols, int *pTransposedMatrix);
void read_matrix_parallel(const char *pFilename, int *pRow, int *pCol, int **ppMatrix);

int main(int argc, char *argv[])
{
    if (argc != 4) {
        printf("Usage: %s <MatrixA_File> <MatrixB_File> <MatrixC_Output_File>\n", argv[0]);
        printf("Example: %s MA_6x7.bin MB_7x6.bin MC_6x6.bin\n", argv[0]);
        return 0;
    }

    char *matAName = argv[1];
    char *matBName = argv[2];
    char *matCName = argv[3];

    int i, j, k;

    /* Clock information */
    struct timespec start, end; 
    double time_taken;
    clock_gettime(CLOCK_MONOTONIC, &start); 

    printf("Matrix Multiplication using 1-Dimension Arrays - Start\n\n");

    // 1. Read Matrix A
    int rowA = 0, colA = 0;
	int *pMatrixA = NULL;
	int *pMatrixB = NULL;
    printf("Reading Matrix A (%s) - Start\n", matAName);
	read_matrix_parallel(matAName, &rowA, &colA, &pMatrixA);
    printf("Reading Matrix A - Done\n");

    // 2. Read Matrix B
    int rowB = 0, colB = 0;
    printf("Reading Matrix B (%s) - Start\n", matBName);
	read_matrix_parallel(matBName, &rowB, &colB, &pMatrixB);
    printf("Reading Matrix B - Done\n");

	// Transpose matrix B
    int *pMatrixBT = (int*)malloc((rowB * colB) * sizeof(int));
    if (pMatrixBT == NULL) {
        printf("Error: Memory allocation failed for Transposed Matrix B.\n");
        free(pMatrixA);
        free(pMatrixB);
        return 0;
    }
    transpose(pMatrixB, rowB, colB, pMatrixBT);

    // 3. Perform matrix multiplication 
    printf("Matrix Multiplication - Start\n");
        
    int rowC = rowA, colC = colB;
    unsigned long long *pMatrixC = (unsigned long long*)calloc((rowC * colC), sizeof(unsigned long long));
    if (pMatrixC == NULL) {
        printf("Error: Memory allocation failed for Matrix C.\n");
        free(pMatrixA);
        free(pMatrixB);
        free(pMatrixBT);
        return 0;
    }

    int commonPoint = colA;
    for (i = 0; i < rowC; i++) {
        for (j = 0; j < colC; j++) {
            for (k = 0; k < commonPoint; k++) {
                pMatrixC[(i * colC) + j] += (pMatrixA[(i * colA) + k] * pMatrixBT[(j * rowB) + k]);
            }
        }
    }
    printf("Matrix Multiplication - Done\n");

    // 4. Write results to a new file
    printf("Write Resultant Matrix C (%s) to File - Start\n", matCName);

    FILE *pFileC = fopen(matCName, "wb");
    if (pFileC == NULL) {
        printf("Error: Unable to create output file %s.\n", matCName);
        free(pMatrixA);
        free(pMatrixB);
        free(pMatrixBT);
        free(pMatrixC);
        return 0;
    }

    fwrite(&rowC, sizeof(int), 1, pFileC); 
    fwrite(&colC, sizeof(int), 1, pFileC); 
    for (i = 0; i < rowC; i++) {
        fwrite(&pMatrixC[i * colC], sizeof(unsigned long long), colC, pFileC);
    }
    fclose(pFileC);
    printf("Write Resultant Matrix C - Done\n");

    clock_gettime(CLOCK_MONOTONIC, &end); 
    time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9; 
    printf("Overall time (Including read, multiplication and write)(s): %lf\n", time_taken);

    // Clean up
    free(pMatrixA);
    free(pMatrixB);
    free(pMatrixBT);
    free(pMatrixC);

    printf("Matrix Multiplication using 1-Dimension Arrays - Done\n");
    return 0;
}

void transpose(int *pInMatrix, int rows, int cols, int *pTransposedMatrix)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pTransposedMatrix[(j * rows) + i] = pInMatrix[(i * cols) + j];
        }
    }
}

// Read a matrix from a file in parallel
void read_matrix_parallel(const char *pFilename, int *pRow, int *pCol, int **ppMatrix) 
{
    int fd = open(pFilename, O_RDONLY);
    // check the file discriptor for error
    if (fd == -1) {
        fprintf(stderr, "Unable to open file %s\n", pFilename);
        return;
    }

    // Read matrix dimensions
    pread(fd, pRow, sizeof(int), 0);
    pread(fd, pCol, sizeof(int), sizeof(int));

    // Allocate memory for the matrix
    *ppMatrix = (int *)malloc((*pRow) * (*pCol) * sizeof(int));

    // Parallel read using OpenMP
    int num_threads = NUM_THREADS;
    #pragma omp parallel num_threads(num_threads)
    {
        // Compute the start and end rows for the thread
        int thread_id = omp_get_thread_num();
        int rows_per_thread = (*pRow + num_threads - 1) / num_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = start_row + rows_per_thread;
        if (end_row > *pRow){
			end_row = *pRow;
		}

        // Get size and offset for reading
        size_t row_size_bytes = (*pCol) * sizeof(int);
        off_t data_offset = 2 * sizeof(int); // Skip the row and col dimensions 

        for (int i = start_row; i < end_row; i++) {
            off_t offset = data_offset + i * row_size_bytes;
            // Read the row from the file into the matrix
            ssize_t bytes_read = pread(fd, &(*ppMatrix)[i * (*pCol)], row_size_bytes, offset);
            if (bytes_read != row_size_bytes) {
                fprintf(stderr, "Thread %d: Error reading file %s at row %d\n", thread_id, pFilename, i);
                break;
            }
        }
    }

    close(fd);
}
