///////////////////////////////////////////////////////////////////////////////////////////
// MatrixMul_1D_bin.c
// --------------------------------------------------------------------------------------
// 
// Multiplies two matrices and writes the resultant multiplication into a binary file.
///////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h> 

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
    printf("Reading Matrix A (%s) - Start\n", matAName);

    FILE *pFileA = fopen(matAName, "rb");
    if (pFileA == NULL) {
        printf("Error: File %s doesn't exist.\n", matAName);
        return 0;
    }

    fread(&rowA, sizeof(int), 1, pFileA); 
    fread(&colA, sizeof(int), 1, pFileA); 

    int *pMatrixA = (int*)malloc((rowA * colA) * sizeof(int));
    if (pMatrixA == NULL) {
        printf("Error: Memory allocation failed for Matrix A.\n");
        fclose(pFileA);
        return 0;
    }

    for (i = 0; i < rowA; i++) {
        fread(&pMatrixA[i * colA], sizeof(int), colA, pFileA);
    }
    fclose(pFileA);
    printf("Reading Matrix A - Done\n");

    // 2. Read Matrix B
    int rowB = 0, colB = 0;
    printf("Reading Matrix B (%s) - Start\n", matBName);

    FILE *pFileB = fopen(matBName, "rb");
    if (pFileB == NULL) {
        printf("Error: File %s doesn't exist.\n", matBName);
        free(pMatrixA);
        return 0;
    }

    fread(&rowB, sizeof(int), 1, pFileB); 
    fread(&colB, sizeof(int), 1, pFileB); 

    int *pMatrixB = (int*)malloc((rowB * colB) * sizeof(int));
    if (pMatrixB == NULL) {
        printf("Error: Memory allocation failed for Matrix B.\n");
        fclose(pFileB);
        free(pMatrixA);
        return 0;
    }

    for (i = 0; i < rowB; i++) {
        fread(&pMatrixB[i * colB], sizeof(int), colB, pFileB); 
    }
    fclose(pFileB);
    printf("Reading Matrix B - Done\n");

    // 3. Perform matrix multiplication 
    printf("Matrix Multiplication - Start\n");   
        
    int rowC = rowA, colC = colB;
    unsigned long long *pMatrixC = (unsigned long long*)calloc((rowC * colC), sizeof(unsigned long long));
    if (pMatrixC == NULL) {
        printf("Error: Memory allocation failed for Matrix C.\n");
        free(pMatrixA);
        free(pMatrixB);
        return 0;
    }

    int commonPoint = colA;
    for (i = 0; i < rowC; i++) {
        for (j = 0; j < colC; j++) {
            for (k = 0; k < commonPoint; k++) {
                pMatrixC[(i * colC) + j] += (pMatrixA[(i * colA) + k] * pMatrixB[(k * colB) + j]);
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
    free(pMatrixC);

    printf("Matrix Multiplication using 1-Dimension Arrays - Done\n");
    return 0;
}