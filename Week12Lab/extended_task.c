///////////////////////////////////////////////////////////////////////////////////////////
// extended_task.c
// --------------------------------------------------------------------------------------
//
// DISTRIBUTED MATRIX MULTIPLICATION USING FOX ALGORITHM
// 
// This implementation uses the Fox algorithm for distributed matrix multiplication,
// combining MPI for inter-process communication and OpenMP for intra-process parallelism.
// The Fox algorithm is particularly efficient for square matrices on 2D process grids.
//
// KEY FEATURES:
//  - Fox algorithm with row broadcasting and column shifting
//  - Cartesian process grid topology for optimal communication patterns
//  - Parallel file I/O using POSIX threads for reading binary matrices
//  - OpenMP parallelization within each MPI process
//  - Memory-efficient tile-based computation
//
// ALGORITHM OVERVIEW:
//  - Processes arranged in q×q grid where q = sqrt(p)
//  - Each process holds tiles of matrices A and B
//  - Fox algorithm performs q iterations with:
//    1. Row broadcast of A tiles
//    2. Local matrix multiplication
//    3. Column shift of B tiles
//
// COMPILATION:
//   mpicc -fopenmp -lm -O3 extended_task.c -o extended_task
//
// EXECUTION:
//   export OMP_NUM_THREADS=4
//   mpirun -np 4 ./extended_task matrixA.bin matrixB.bin result.bin
///////////////////////////////////////////////////////////////////////////////////////////

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <omp.h>
#include "mpi.h"

// Configuration constants for Fox algorithm implementation
#define DEFAULT_FILE_IO_THREADS 2     // Number of threads for parallel file I/O
#define DEFAULT_COMPUTATION_THREADS 4  // Number of threads for matrix computation
#define MAX_PROCESS_GRID_SIZE 100      // Maximum supported process grid dimension
#define FOX_ALGORITHM_ITERATIONS 10   // Maximum Fox algorithm iterations

static int read_matrix_parallel(const char *filename, int *rows, int *cols, int **matrix_data);
static void copy_tile_int(const int *source_matrix, int source_cols, int row_start, int col_start,
                          int tile_height, int tile_width, int *destination_matrix);
static void copy_tile_ull(const unsigned long long *tile_data, unsigned long long *global_matrix,
                          int global_cols, int row_start, int col_start,
                          int tile_height, int tile_width);
static void multiply_add_tile(const int *matrix_a_tile, const int *matrix_b_tile, unsigned long long *result_tile,
                              int tile_rows, int tile_cols, int tile_k, int thread_count);

/**
 * MAIN FUNCTION: Fox Algorithm Matrix Multiplication
 * 
 * This function implements the Fox algorithm for distributed matrix multiplication:
 * 1. Validates that the number of processes is a perfect square
 * 2. Reads input matrices using parallel I/O (root process only)
 * 3. Creates Cartesian process grid topology
 * 4. Distributes matrix tiles to processes
 * 5. Executes Fox algorithm with row broadcasts and column shifts
 * 6. Gathers results and writes output file (root process only)
 */
int main(int argc, char *argv[])
{
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get process information
    int process_rank = 0, total_processes = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    // Validate command line arguments
    if (argc != 4) {
        if (process_rank == 0) {
            printf("Usage: %s <MatrixA_File> <MatrixB_File> <MatrixC_Output_File>\n", argv[0]);
            printf("Example: %s MA_1000x1000.bin MB_1000x1000.bin MC_1000x1000_fox.bin\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    // Parse command line arguments
    const char *input_matrix_a_path = argv[1];
    const char *input_matrix_b_path = argv[2];
    const char *output_matrix_c_path = argv[3];

    // Matrix dimension variables
    int matrix_a_rows = 0, matrix_a_cols = 0;
    int matrix_b_rows = 0, matrix_b_cols = 0;
    
    // Matrix data pointers
    int *matrix_a_data = NULL;
    int *matrix_b_data = NULL;
    unsigned long long *result_matrix_c = NULL;
    
    // Timing variables
    double computation_start_time = 0.0, computation_end_time = 0.0;

    // Validate that total processes is a perfect square for Fox algorithm
    int process_grid_dimension = 1;
    while (process_grid_dimension * process_grid_dimension < total_processes) {
        process_grid_dimension++;
    }
    if (process_grid_dimension * process_grid_dimension != total_processes) {
        if (process_rank == 0) {
            fprintf(stderr, "ERROR: Total processes (%d) is not a perfect square. Fox algorithm requires p = q^2.\n",
                    total_processes);
        }
        MPI_Finalize();
        return 0;
    }

    // Root process handles file I/O and matrix preparation
    if (process_rank == 0) {
        printf("=== FOX ALGORITHM MATRIX MULTIPLICATION ===\n\n");

        // Read Matrix A using parallel I/O
        printf("[ROOT] Reading Matrix A from %s...\n", input_matrix_a_path);
        if (read_matrix_parallel(input_matrix_a_path, &matrix_a_rows, &matrix_a_cols, &matrix_a_data) != 0) {
            fprintf(stderr, "[ROOT] ERROR: Failed to read Matrix A from %s\n", input_matrix_a_path);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        printf("[ROOT] Matrix A loaded successfully (%d x %d)\n", matrix_a_rows, matrix_a_cols);

        // Read Matrix B using parallel I/O
        printf("[ROOT] Reading Matrix B from %s...\n", input_matrix_b_path);
        if (read_matrix_parallel(input_matrix_b_path, &matrix_b_rows, &matrix_b_cols, &matrix_b_data) != 0) {
            fprintf(stderr, "[ROOT] ERROR: Failed to read Matrix B from %s\n", input_matrix_b_path);
            free(matrix_a_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        printf("[ROOT] Matrix B loaded successfully (%d x %d)\n", matrix_b_rows, matrix_b_cols);

        // Validate matrix dimensions for Fox algorithm
        if (matrix_a_rows != matrix_a_cols || matrix_b_rows != matrix_b_cols || matrix_a_cols != matrix_b_rows) {
            fprintf(stderr, "[ROOT] ERROR: Fox algorithm requires square matrices with matching dimensions.\n");
            free(matrix_a_data);
            free(matrix_b_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Validate that matrix dimensions are divisible by process grid dimension
        if (matrix_a_rows % process_grid_dimension != 0) {
            fprintf(stderr, "[ROOT] ERROR: Matrix dimensions (%d) must be divisible by sqrt(p) = %d.\n",
                    matrix_a_rows, process_grid_dimension);
            free(matrix_a_data);
            free(matrix_b_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Allocate result matrix
        result_matrix_c = (unsigned long long *)calloc((size_t)matrix_a_rows * (size_t)matrix_b_cols,
                                                       sizeof(unsigned long long));
        if (!result_matrix_c) {
            fprintf(stderr, "[ROOT] ERROR: Memory allocation failed for result matrix.\n");
            free(matrix_a_data);
            free(matrix_b_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Start timing the Fox algorithm computation
        computation_start_time = MPI_Wtime();
        printf("[ROOT] Starting Fox algorithm matrix multiplication...\n");
    }

    // Broadcast matrix dimensions and process grid configuration to all processes
    MPI_Bcast(&matrix_a_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_a_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_b_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_b_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&process_grid_dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Validate matrix dimensions
    if (matrix_a_rows == 0 || matrix_a_cols == 0 || matrix_b_rows == 0 || matrix_b_cols == 0) {
        MPI_Finalize();
        return 0;
    }

    // Calculate tile dimensions for Fox algorithm
    const int tile_rows = matrix_a_rows / process_grid_dimension;
    const int tile_cols = matrix_b_cols / process_grid_dimension;
    const int tile_k = matrix_a_cols / process_grid_dimension;

    // Calculate tile element counts for memory allocation
    const size_t tile_a_elements = (size_t)tile_rows * (size_t)tile_k;
    const size_t tile_b_elements = (size_t)tile_k * (size_t)tile_cols;
    const size_t tile_c_elements = (size_t)tile_rows * (size_t)tile_cols;

    // Allocate local tile matrices for Fox algorithm
    int *local_matrix_a_tile = (int *)malloc(tile_a_elements * sizeof(int));
    int *local_matrix_b_tile = (int *)malloc(tile_b_elements * sizeof(int));
    unsigned long long *local_result_tile = (unsigned long long *)calloc(tile_c_elements, sizeof(unsigned long long));
    int *broadcast_matrix_a_tile = (int *)malloc(tile_a_elements * sizeof(int));

    if (!local_matrix_a_tile || !local_matrix_b_tile || !local_result_tile || !broadcast_matrix_a_tile) {
        fprintf(stderr, "[PROCESS %d] ERROR: Memory allocation failed for local tiles.\n", process_rank);
        free(local_matrix_a_tile);
        free(local_matrix_b_tile);
        free(local_result_tile);
        free(broadcast_matrix_a_tile);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Root process distributes matrix tiles to all processes
    if (process_rank == 0) {
        for (int row_block = 0; row_block < process_grid_dimension; ++row_block) {
            for (int col_block = 0; col_block < process_grid_dimension; ++col_block) {
                int target_process = row_block * process_grid_dimension + col_block;
                int row_offset = row_block * tile_rows;
                int col_offset = col_block * tile_cols;

                if (target_process == 0) {
                    // Root process copies its own tiles directly
                    copy_tile_int(matrix_a_data, matrix_a_cols, row_offset, col_block * tile_k,
                                  tile_rows, tile_k, local_matrix_a_tile);
                    copy_tile_int(matrix_b_data, matrix_b_cols, row_block * tile_k, col_offset,
                                  tile_k, tile_cols, local_matrix_b_tile);
                } else {
                    // Allocate temporary buffers for sending tiles
                    size_t bytes_a = tile_a_elements * sizeof(int);
                    size_t bytes_b = tile_b_elements * sizeof(int);
                    int *temp_matrix_a = (int *)malloc(bytes_a);
                    int *temp_matrix_b = (int *)malloc(bytes_b);
                    if (!temp_matrix_a || !temp_matrix_b) {
                        fprintf(stderr, "[ROOT] ERROR: Memory allocation failed during tile distribution.\n");
                        free(temp_matrix_a);
                        free(temp_matrix_b);
                        free(matrix_a_data);
                        free(matrix_b_data);
                        free(result_matrix_c);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }

                    // Copy tiles to temporary buffers
                    copy_tile_int(matrix_a_data, matrix_a_cols, row_offset, col_block * tile_k,
                                  tile_rows, tile_k, temp_matrix_a);
                    copy_tile_int(matrix_b_data, matrix_b_cols, row_block * tile_k, col_offset,
                                  tile_k, tile_cols, temp_matrix_b);

                    // Send tiles to target process
                    MPI_Send(temp_matrix_a, (int)tile_a_elements, MPI_INT, target_process, 0, MPI_COMM_WORLD);
                    MPI_Send(temp_matrix_b, (int)tile_b_elements, MPI_INT, target_process, 1, MPI_COMM_WORLD);

                    free(temp_matrix_a);
                    free(temp_matrix_b);
                }
            }
        }

        // Clean up global matrices (tiles are now distributed)
        free(matrix_a_data);
        free(matrix_b_data);
        matrix_a_data = NULL;
        matrix_b_data = NULL;
    } else {
        // Non-root processes receive their tiles
        MPI_Recv(local_matrix_a_tile, (int)tile_a_elements, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_matrix_b_tile, (int)tile_b_elements, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Create Cartesian topology for Fox algorithm
    int process_coordinates[2] = {0, 0};
    int grid_dimensions[2] = {process_grid_dimension, process_grid_dimension};
    int periodic_boundaries[2] = {1, 1};  // Periodic for Fox algorithm shifts
    MPI_Comm cartesian_communicator;
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid_dimensions, periodic_boundaries, 1, &cartesian_communicator);
    
    int cartesian_rank = 0;
    MPI_Comm_rank(cartesian_communicator, &cartesian_rank);
    MPI_Cart_coords(cartesian_communicator, cartesian_rank, 2, process_coordinates);

    // Create row and column communicators for Fox algorithm
    MPI_Comm row_communicator;
    MPI_Comm column_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, process_coordinates[0], process_coordinates[1], &row_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, process_coordinates[1], process_coordinates[0], &column_communicator);

    int row_rank = 0, column_rank = 0;
    MPI_Comm_rank(row_communicator, &row_rank);
    MPI_Comm_rank(column_communicator, &column_rank);

    // Determine neighbors for column shifting in Fox algorithm
    int neighbor_receive = MPI_PROC_NULL;
    int neighbor_send = MPI_PROC_NULL;
    MPI_Cart_shift(cartesian_communicator, 0, -1, &neighbor_receive, &neighbor_send);

    // Configure OpenMP threads for computation
    int num_compute_threads = DEFAULT_COMPUTATION_THREADS;
    char *omp_threads_env = getenv("OMP_NUM_THREADS");
    if (omp_threads_env != NULL) {
        int env_threads = atoi(omp_threads_env);
        if (env_threads > 0) {
            num_compute_threads = env_threads;
        }
    }

    // Execute Fox algorithm iterations
    for (int fox_step = 0; fox_step < process_grid_dimension; ++fox_step) {
        // Determine which process in the row should broadcast its A tile
        int broadcast_root_column = (process_coordinates[0] + fox_step) % process_grid_dimension;
        
        // Prepare broadcast tile if this process is the root for this step
        if (process_coordinates[1] == broadcast_root_column) {
            memcpy(broadcast_matrix_a_tile, local_matrix_a_tile, tile_a_elements * sizeof(int));
        }

        // Broadcast A tile along the row
        MPI_Bcast(broadcast_matrix_a_tile, (int)tile_a_elements, MPI_INT, broadcast_root_column, row_communicator);

        // Perform local matrix multiplication and accumulation
        multiply_add_tile(broadcast_matrix_a_tile, local_matrix_b_tile, local_result_tile,
                          tile_rows, tile_cols, tile_k, num_compute_threads);

        // Shift B tiles along columns for next iteration
        MPI_Sendrecv_replace(local_matrix_b_tile, (int)tile_b_elements, MPI_INT,
                             neighbor_send, 101,
                             neighbor_receive, 101,
                             cartesian_communicator, MPI_STATUS_IGNORE);
    }

    // Root process gathers results from all processes
    unsigned long long *receive_buffer = NULL;
    if (process_rank == 0) {
        receive_buffer = (unsigned long long *)malloc(tile_c_elements * sizeof(unsigned long long));
        if (!receive_buffer) {
            fprintf(stderr, "[ROOT] ERROR: Memory allocation failed while gathering results.\n");
            free(local_matrix_a_tile);
            free(local_matrix_b_tile);
            free(local_result_tile);
            free(broadcast_matrix_a_tile);
            free(result_matrix_c);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (process_rank == 0) {
        // Copy root process's own result tile to global matrix
        copy_tile_ull(local_result_tile, result_matrix_c, matrix_b_cols,
                      process_coordinates[0] * tile_rows, process_coordinates[1] * tile_cols,
                      tile_rows, tile_cols);

        // Gather result tiles from other processes
        for (int source_process = 1; source_process < total_processes; ++source_process) {
            int source_coordinates[2] = {0, 0};
            MPI_Recv(source_coordinates, 2, MPI_INT, source_process, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(receive_buffer, (int)tile_c_elements, MPI_UNSIGNED_LONG_LONG,
                     source_process, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            copy_tile_ull(receive_buffer, result_matrix_c, matrix_b_cols,
                          source_coordinates[0] * tile_rows, source_coordinates[1] * tile_cols,
                          tile_rows, tile_cols);
        }

        computation_end_time = MPI_Wtime();
        printf("[ROOT] Fox algorithm matrix multiplication completed in %.6f seconds\n", 
               computation_end_time - computation_start_time);

        // Write result matrix to binary file
        FILE *output_file = fopen(output_matrix_c_path, "wb");
        if (!output_file) {
            fprintf(stderr, "[ROOT] ERROR: Unable to open output file %s for writing: %s\n",
                    output_matrix_c_path, strerror(errno));
        } else {
            // Write matrix dimensions
            fwrite(&matrix_a_rows, sizeof(int), 1, output_file);
            fwrite(&matrix_b_cols, sizeof(int), 1, output_file);
            
            // Write matrix data row by row
            for (int row = 0; row < matrix_a_rows; ++row) {
                fwrite(&result_matrix_c[(size_t)row * (size_t)matrix_b_cols],
                       sizeof(unsigned long long), (size_t)matrix_b_cols, output_file);
            }
            fclose(output_file);
            printf("[ROOT] Result matrix written to %s\n", output_matrix_c_path);
        }
    } else {
        // Non-root processes send their results to root
        MPI_Send(process_coordinates, 2, MPI_INT, 0, 200, MPI_COMM_WORLD);
        MPI_Send(local_result_tile, (int)tile_c_elements, MPI_UNSIGNED_LONG_LONG,
                 0, 201, MPI_COMM_WORLD);
    }

    // Clean up local memory allocations
    free(local_matrix_a_tile);
    free(local_matrix_b_tile);
    free(local_result_tile);
    free(broadcast_matrix_a_tile);
    free(receive_buffer);
    free(result_matrix_c);

    // Free MPI communicators
    MPI_Comm_free(&row_communicator);
    MPI_Comm_free(&column_communicator);
    MPI_Comm_free(&cartesian_communicator);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}

/**
 * IMPLEMENTATION: Parallel Matrix File Reader for Fox Algorithm
 * 
 * Reads a binary matrix file using parallel I/O with OpenMP threads.
 * This function is optimized for the Fox algorithm which requires square matrices.
 */
static int read_matrix_parallel(const char *filename, int *rows, int *cols, int **matrix_data)
{
    // Open the binary file for reading
    int file_descriptor = open(filename, O_RDONLY);
    if (file_descriptor == -1) {
        fprintf(stderr, "ERROR: Unable to open file %s: %s\n", filename, strerror(errno));
        return -1;
    }

    // Read matrix dimensions from file header
    if (pread(file_descriptor, rows, sizeof(int), 0) != sizeof(int) ||
        pread(file_descriptor, cols, sizeof(int), sizeof(int)) != sizeof(int)) {
        fprintf(stderr, "ERROR: Failed to read matrix dimensions from %s\n", filename);
        close(file_descriptor);
        return -1;
    }

    // Validate matrix dimensions
    if (*rows <= 0 || *cols <= 0) {
        fprintf(stderr, "ERROR: Invalid matrix dimensions (%d x %d) in %s\n", *rows, *cols, filename);
        close(file_descriptor);
        return -1;
    }

    // Allocate memory for matrix data
    size_t total_elements = (size_t)(*rows) * (size_t)(*cols);
    *matrix_data = (int *)malloc(total_elements * sizeof(int));
    if (!*matrix_data) {
        fprintf(stderr, "ERROR: Memory allocation failed for matrix %s\n", filename);
        close(file_descriptor);
        return -1;
    }

    // Calculate I/O parameters
    size_t bytes_per_row = (size_t)(*cols) * sizeof(int);
    off_t data_start_offset = 2 * (off_t)sizeof(int);
    
    // Determine number of threads for parallel I/O
    int num_io_threads = DEFAULT_FILE_IO_THREADS;
    char *io_threads_env = getenv("MATRIX_READ_THREADS");
    if (io_threads_env != NULL) {
        int env_threads = atoi(io_threads_env);
        if (env_threads > 0) {
            num_io_threads = env_threads;
        }
    }

    // Parallel I/O using OpenMP threads
#pragma omp parallel num_threads(num_io_threads)
    {
        int thread_id = omp_get_thread_num();
        int rows_per_thread = (*rows + num_io_threads - 1) / num_io_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = start_row + rows_per_thread;
        if (end_row > *rows) {
            end_row = *rows;
        }

        // Each thread reads its assigned rows
        for (int row = start_row; row < end_row; ++row) {
            off_t row_offset = data_start_offset + (off_t)row * (off_t)bytes_per_row;
            ssize_t bytes_read = pread(file_descriptor, 
                                      &(*matrix_data)[(size_t)row * (size_t)(*cols)],
                                      bytes_per_row, row_offset);
            if (bytes_read != (ssize_t)bytes_per_row) {
                fprintf(stderr, "Thread %d: Failed to read row %d from %s\n",
                        thread_id, row, filename);
            }
        }
    }

    close(file_descriptor);
    return 0;
}

/**
 * IMPLEMENTATION: Tile Copy Function for Fox Algorithm
 * 
 * Copies a tile from source matrix to destination matrix.
 * This function is used during tile distribution in the Fox algorithm.
 */
static void copy_tile_int(const int *source_matrix, int source_cols, int row_start, int col_start,
                          int tile_height, int tile_width, int *destination_matrix)
{
    for (int row = 0; row < tile_height; ++row) {
        const int *source_row = source_matrix + ((size_t)(row_start + row) * (size_t)source_cols + (size_t)col_start);
        memcpy(&destination_matrix[(size_t)row * (size_t)tile_width], source_row, (size_t)tile_width * sizeof(int));
    }
}

/**
 * IMPLEMENTATION: Result Tile Copy Function for Fox Algorithm
 * 
 * Copies computed tile data back to the global result matrix.
 * This function is used during result gathering in the Fox algorithm.
 */
static void copy_tile_ull(const unsigned long long *tile_data, unsigned long long *global_matrix,
                          int global_cols, int row_start, int col_start,
                          int tile_height, int tile_width)
{
    for (int row = 0; row < tile_height; ++row) {
        unsigned long long *destination_row = global_matrix + ((size_t)(row_start + row) * (size_t)global_cols + (size_t)col_start);
        const unsigned long long *source_row = tile_data + (size_t)row * (size_t)tile_width;
        memcpy(destination_row, source_row, (size_t)tile_width * sizeof(unsigned long long));
    }
}

/**
 * IMPLEMENTATION: Fox Algorithm Matrix Multiplication
 * 
 * Performs matrix multiplication and accumulation for the Fox algorithm.
 * This function multiplies two tile blocks and adds the result to the output tile.
 * The accumulation is essential for the Fox algorithm's iterative nature.
 */
static void multiply_add_tile(const int *matrix_a_tile, const int *matrix_b_tile, unsigned long long *result_tile,
                              int tile_rows, int tile_cols, int tile_k, int thread_count)
{
    // Parallel matrix multiplication with accumulation using OpenMP
#pragma omp parallel for collapse(2) num_threads(thread_count)
    for (int row = 0; row < tile_rows; ++row) {
        for (int col = 0; col < tile_cols; ++col) {
            unsigned long long accumulator = 0ULL;
            
            // Get pointers to current row and column
            const int *a_row_ptr = matrix_a_tile + (size_t)row * (size_t)tile_k;
            const int *b_col_ptr = matrix_b_tile + (size_t)col;
            
            // Perform dot product with accumulation
            for (int k = 0; k < tile_k; ++k) {
                accumulator += (unsigned long long)a_row_ptr[k] * (unsigned long long)b_col_ptr[(size_t)k * (size_t)tile_cols];
            }
            
            // Accumulate result (essential for Fox algorithm)
            result_tile[(size_t)row * (size_t)tile_cols + (size_t)col] += accumulator;
        }
    }
}
