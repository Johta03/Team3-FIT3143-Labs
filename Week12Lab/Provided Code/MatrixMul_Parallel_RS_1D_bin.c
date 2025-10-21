//////////////////////////////////////////////////////////////////////////////////////
// MatrixMul_Parallel_RS_1D_bin.c
// ----------------------------------------------------------------------------------
//
// Multiplies two matrices and writes the resultant multiplication into a binary file.
// Applies a parallel design based on row segmentation with MPI.
//
// Initial version by: Chin-Kit Ng
//
// Last updated by: Vishnu Monn
//
// Last updated date: 14th October 2025
//////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <memory.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>

#define NUM_THREADS_READ 2

void transpose(int *pInMatrix, int rows, int cols, int *pTransposedMatrix);
void read_matrix_parallel(const char *pFilename, int *pRow, int *pCol, int **ppMatrix);

int main(int argc, char* argv[])
{
	int my_rank, processors;
	int i=0, j=0, k=0;
	int *pRowBuff = NULL;
	unsigned long long *pResBuff = NULL;
	int rows_per_procs, element_offset = 0, result_offset = 0;
	int ref_point, row_remain, end_val = 100;
	MPI_Status stat;

	int *pMatrix_A = NULL;
	int *pMatrix_BT = NULL;
	unsigned long long *pMatrix_C = NULL;
	int ma_row = 0, ma_col = 0;
	int mb_row = 0, mb_col = 0;
	int mc_row = 0, mc_col = 0;
	double startTime, endTime;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processors);

	if (argc != 4) {
		if(my_rank == 0){
			printf("Usage: %s <MatrixA_File> <MatrixB_File> <MatrixC_Output_File>\n", argv[0]);
			printf("Example: %s MA_6x7.bin MB_7x6.bin MC_6x6.bin\n", argv[0]);
			MPI_Finalize();
		}
		return 0;
	}
		
	if(my_rank == 0){
		printf("Parallel Matrix Multiplication using 1-Dimension Arrays - Start\n\n");
	
		char *matAName = argv[1];
		char *matBName = argv[2];
		
		// 1. Read Matrix A
		printf("Reading Matrix A (%s) - Start\n", matAName);
		read_matrix_parallel(matAName, &ma_row, &ma_col, &pMatrix_A);
		printf("Reading Matrix A - Done\n");

		// 2. Read Matrix B
		int *pMatrix_B = NULL;
		printf("Reading Matrix B (%s) - Start\n", matBName);
		read_matrix_parallel(matBName, &mb_row, &mb_col, &pMatrix_B);
		printf("Reading Matrix B - Done\n");
		
		pMatrix_BT = (int*)malloc((mb_row * mb_col) * sizeof(int));
		transpose(pMatrix_B, mb_row, mb_col, pMatrix_BT);
		free(pMatrix_B);

		// Start the computation time (which covers the communication time)
		startTime = MPI_Wtime();
	}

	//broadcast row and col number for Matrix A & B to all processors
	MPI_Bcast(&ma_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ma_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&mb_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&mb_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if(my_rank == 0){
		rows_per_procs = ma_row / processors;
		row_remain = ma_row % processors;
		element_offset = rows_per_procs * ma_col;

		//Distribute rows of Matrix A to all processors, remainder rows taken by last processor 
		for(i= 1; i < processors; i++){
			if(i == processors -1){
				MPI_Send(pMatrix_A + element_offset, (rows_per_procs + row_remain) * ma_col, MPI_INT, i, 0, MPI_COMM_WORLD);
			}else{
				MPI_Send(pMatrix_A + element_offset, rows_per_procs * ma_col, MPI_INT, i, 0, MPI_COMM_WORLD);
				element_offset += rows_per_procs * ma_col;
			}
		}
	}else{
		mc_row = ma_row; 
		mc_col = mb_col;
		rows_per_procs = ma_row / processors;
		row_remain = ma_row % processors;
		
		pMatrix_BT = (int*)malloc((mb_row*mb_col)*sizeof(int));

		//Receive rows of Matrix A from node 0, last node receives more element due to remainder rows
		if(my_rank == processors-1){
			pRowBuff = (int*)malloc(((rows_per_procs + row_remain) * ma_col)  * sizeof(int));
			pResBuff = (unsigned long long*)calloc(((rows_per_procs + row_remain) * mc_col), sizeof(unsigned long long)); 
			MPI_Recv(pRowBuff, (rows_per_procs + row_remain) * ma_col, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
		}else{
			pRowBuff = (int*)malloc((rows_per_procs * ma_col)  * sizeof(int));
			pResBuff = (unsigned long long*)calloc((rows_per_procs * mc_col), sizeof(unsigned long long)); 
			MPI_Recv(pRowBuff, rows_per_procs * ma_col, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
		}
		
	}

	//Broadcast the entire Matrix B to all processors
	MPI_Bcast(pMatrix_BT, (mb_row*mb_col), MPI_INT, 0, MPI_COMM_WORLD);

	if(my_rank == 0){
		mc_row = ma_row; mc_col = mb_col;
		pMatrix_C = (unsigned long long*)calloc((mc_row * mc_col), sizeof(unsigned long long));
		
		//Start matrix multiplication for node 0
		ref_point = ma_col;
		for(i=0; i<rows_per_procs; i++){
			for(j=0; j<mc_col; j++){
				for(k=0; k<ref_point; k++){
					pMatrix_C[(i * mc_col) + j] += pMatrix_A[(i * ma_col) + k] * pMatrix_BT[(j * mb_row) + k];
				}
			}
		}

		//Receive multiplication results from other processors into the Matrix C  
		result_offset = rows_per_procs * mc_col;
		for(i=1; i<processors; i++){
			if(i == processors-1){
				MPI_Recv(pMatrix_C + result_offset, (rows_per_procs + row_remain) * mc_col, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, &stat);
			}else{
				MPI_Recv(pMatrix_C + result_offset, rows_per_procs * mc_col, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, &stat);
				result_offset += rows_per_procs * mc_col;
			}
		}
	}else{
		ref_point = ma_col;

		//Matrix Multiplication at last node
		if(my_rank == processors-1){
			for(i=0; i<(rows_per_procs + row_remain); i++){
				for(j=0; j<mc_col; j++){
					for(k=0; k<ref_point; k++){
						pResBuff[(i * mc_col) + j] += pRowBuff[(i * ma_col) + k] * pMatrix_BT[(j * mb_row) + k];
					}
				}
			}
			//Send the results back to node 0
			MPI_Send(pResBuff, (rows_per_procs + row_remain) * mc_col, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
		}else{
			//Matrix Multiplication at other nodes
			for(i=0; i<rows_per_procs; i++){
				for(j=0; j<mc_col; j++){
					for(k=0; k<ref_point; k++){
						pResBuff[(i * mc_col) + j] += pRowBuff[(i * ma_col) + k] * pMatrix_BT[(j * mb_row) + k];
					}
				}
			}

			//Send results back to node 0
			MPI_Send(pResBuff, rows_per_procs * mc_col, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
		}

		free(pRowBuff);
		free(pResBuff);
	}

	if(my_rank == 0){
		printf("Matrix Multiplication - Done\n");
		endTime = MPI_Wtime();
		printf("Matrix Multiplication - End. Process time (s): %lf\n", endTime - startTime);


		//Write the results of Matrix Multiplication into file 
		char *matCName = argv[3];
		FILE *pfile_MC = fopen(matCName, "wb");
		fwrite(&mc_row, sizeof(int), 1, pfile_MC); //write the row number of Matrix C
		fwrite(&mc_col, sizeof(int), 1, pfile_MC); //write the col number of Matrix C
		for(i=0; i<mc_row; i++){
			fwrite(&pMatrix_C[i * mc_col], sizeof(unsigned long long), mc_col, pfile_MC); //write an entire row of cols
		}
		fclose(pfile_MC);

		printf("Write Matrix Files - Done\n");
	}
	
	free(pMatrix_A);
	free(pMatrix_BT);
	free(pMatrix_C);
	MPI_Finalize();
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
    int num_threads = NUM_THREADS_READ;
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
