///////////////////////////////////////////////////////////////////////////////////////////
// coretask.c
// --------------------------------------------------------------------------------------
//
// DISTRIBUTED MATRIX MULTIPLICATION WITH HYBRID PARALLELIZATION
// 
// This implementation uses a hybrid approach combining MPI for distributed computing
// and OpenMP for shared-memory parallelization. The algorithm partitions matrices
// into 2D tiles and distributes them across MPI processes for parallel computation.
//
// KEY FEATURES:
//  - Parallel file I/O using POSIX threads for reading binary matrix files
//  - Dynamic process grid optimization based on matrix dimensions
//  - Memory-efficient tile-based distribution avoiding full matrix broadcasts
//  - OpenMP parallelization within each MPI process for fine-grained parallelism
//  - Optimized matrix transpose operations for improved cache performance
//
// COMPILATION:
//   mpicc -fopenmp -lm -O3 core_task.c -o core_task
//
// EXECUTION:
//   export OMP_NUM_THREADS=4
//   mpirun -np 4 ./core_task matrixA.bin matrixB.bin result.bin
//
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
#include <stdint.h>

// Configuration constants for parallel execution
#define DEFAULT_IO_THREADS 2        // Number of threads for parallel file I/O
#define DEFAULT_COMPUTE_THREADS 4   // Number of threads for matrix computation
#define MAX_MATRIX_DIM 10000        // Maximum supported matrix dimension
#define TILE_SIZE_THRESHOLD 64      // Minimum tile size for efficient computation


static void transpose_matrix(const uint32_t *source_matrix, size_t num_rows, size_t num_cols, uint32_t *transposed_matrix);
static void transpose_matrix_in_place(uint32_t *matrix, size_t dimension);
static int read_matrix_parallel(const char *filename, uint32_t *rows, uint32_t *cols, uint32_t **matrix_data);
static void choose_process_grid(int total_processes, uint32_t matrix_rows, uint32_t matrix_cols,
                                uint32_t *grid_rows, uint32_t *grid_cols);
static void multiply_tile_block(const uint32_t *matrix_a, const uint32_t *matrix_b_transposed,
                                uint64_t *result_matrix,
                                uint32_t tile_rows, uint32_t tile_cols, uint32_t common_dimension, int thread_count);
static void copy_tile_into_global(const uint64_t *tile_data, uint64_t *global_matrix,
                                  uint32_t global_cols, uint32_t row_start, uint32_t col_start,
                                  uint32_t tile_height, uint32_t tile_width);

/**
 * MAIN FUNCTION: Distributed Matrix Multiplication with Hybrid Parallelization
 * 
 * This function orchestrates the entire parallel matrix multiplication process:
 * 1. Initializes MPI and validates command line arguments
 * 2. Reads input matrices using parallel I/O (root process only)
 * 3. Determines optimal process grid layout
 * 4. Distributes matrix tiles to MPI processes
 * 5. Performs parallel computation using OpenMP threads
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
            printf("Example: %s MA_6x7.bin MB_7x6.bin MC_6x6.bin\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    // Parse command line arguments
    const char *input_matrix_a_path = argv[1];
    const char *input_matrix_b_path = argv[2];
    const char *output_matrix_c_path = argv[3];

    // Matrix dimension variables
    uint32_t matrix_a_rows = 0, matrix_a_cols = 0;
    uint32_t matrix_b_rows = 0, matrix_b_cols = 0;
    
    // Matrix data pointers
    uint32_t *matrix_a_data = NULL;
    uint32_t *matrix_b_data = NULL;
    uint32_t *matrix_b_transposed = NULL;
    uint64_t *result_matrix_c = NULL;
    
    // Timing variables
    double computation_start_time = 0.0, computation_end_time = 0.0;

    // Process grid configuration
    uint32_t process_grid_rows = 1, process_grid_cols = total_processes;

    // Root process handles file I/O and matrix preparation
    if (process_rank == 0) {
        printf("=== DISTRIBUTED MATRIX MULTIPLICATION WITH HYBRID PARALLELIZATION ===\n\n");

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

        // Validate matrix dimensions for multiplication
        if (matrix_a_cols != matrix_b_rows) {
            fprintf(stderr, "[ROOT] ERROR: Dimension mismatch - A columns (%d) != B rows (%d)\n", 
                    matrix_a_cols, matrix_b_rows);
            free(matrix_a_data);
            free(matrix_b_data);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Optimize Matrix B by transposing for better cache performance
        printf("[ROOT] Optimizing Matrix B by transposition...\n");
        if (matrix_b_rows != matrix_b_cols) {
            // Non-square matrix: create separate transposed copy
            matrix_b_transposed = malloc((size_t)matrix_b_rows * (size_t)matrix_b_cols * sizeof(uint32_t));
            if (!matrix_b_transposed) {
                fprintf(stderr, "[ROOT] ERROR: Memory allocation failed for transposed Matrix B\n");
                free(matrix_a_data);
                free(matrix_b_data);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            transpose_matrix(matrix_b_data, matrix_b_rows, matrix_b_cols, matrix_b_transposed);
            free(matrix_b_data);
        } else {
            // Square matrix: transpose in-place for memory efficiency
            transpose_matrix_in_place(matrix_b_data, matrix_b_rows);
            matrix_b_transposed = matrix_b_data;
        }
        matrix_b_data = NULL; // Prevent accidental access to original B

        // Determine optimal process grid layout
        choose_process_grid(total_processes, matrix_a_rows, matrix_b_cols, &process_grid_rows, &process_grid_cols);
        printf("[ROOT] Process grid configuration: %d x %d (rows x cols)\n", process_grid_rows, process_grid_cols);

        // Allocate result matrix
        result_matrix_c = calloc((size_t)matrix_a_rows * (size_t)matrix_b_cols, sizeof(uint64_t));
        if (!result_matrix_c) {
            fprintf(stderr, "[ROOT] ERROR: Memory allocation failed for result matrix\n");
            free(matrix_a_data);
            free(matrix_b_transposed);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Start timing the computation
        computation_start_time = MPI_Wtime();
        printf("[ROOT] Starting parallel matrix multiplication...\n");
    }

    // Broadcast matrix dimensions and process grid configuration to all processes
    MPI_Bcast(&matrix_a_rows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_a_cols, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_b_rows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix_b_cols, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&process_grid_rows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&process_grid_cols, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Validate matrix dimensions
    if (matrix_a_rows == 0 || matrix_a_cols == 0 || matrix_b_rows == 0 || matrix_b_cols == 0) {
        if (process_rank == 0) {
            fprintf(stderr, "[ROOT] ERROR: Invalid matrix dimensions detected.\n");
            free(matrix_a_data);
            free(matrix_b_transposed);
            free(result_matrix_c);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Root process prepares tile distribution metadata for all processes
    unsigned *tile_row_offsets = NULL;
    unsigned *tile_row_counts = NULL;
    unsigned *tile_col_offsets = NULL;
    unsigned *tile_col_counts = NULL;

    if (process_rank == 0) {
        // Allocate memory for tile distribution metadata
        tile_row_offsets = malloc((size_t)total_processes * sizeof(unsigned));
        tile_row_counts  = malloc((size_t)total_processes * sizeof(unsigned));
        tile_col_offsets = malloc((size_t)total_processes * sizeof(unsigned));
        tile_col_counts  = malloc((size_t)total_processes * sizeof(unsigned));

        if (!tile_row_offsets || !tile_row_counts || !tile_col_offsets || !tile_col_counts) {
            fprintf(stderr, "[ROOT] ERROR: Memory allocation failed for tile distribution metadata\n");
            free(matrix_a_data);
            free(matrix_b_transposed);
            free(result_matrix_c);
            free(tile_row_offsets);
            free(tile_row_counts);
            free(tile_col_offsets);
            free(tile_col_counts);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Allocate prefix arrays for efficient tile distribution calculation
        unsigned *row_prefix_sums = malloc((size_t)(process_grid_rows + 1) * sizeof(unsigned));
        unsigned *col_prefix_sums = malloc((size_t)(process_grid_cols + 1) * sizeof(unsigned));

        if (!row_prefix_sums || !col_prefix_sums) {
            fprintf(stderr, "[ROOT] ERROR: Memory allocation failed for prefix sum arrays\n");
            free(matrix_a_data);
            free(matrix_b_transposed);
            free(result_matrix_c);
            free(tile_row_offsets);
            free(tile_row_counts);
            free(tile_col_offsets);
            free(tile_col_counts);
            free(row_prefix_sums);
            free(col_prefix_sums);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Calculate row distribution using prefix sums
        row_prefix_sums[0] = 0;
        unsigned base_rows_per_process = matrix_a_rows / process_grid_rows;
        unsigned extra_rows = matrix_a_rows % process_grid_rows;
        for (unsigned r = 0; r < process_grid_rows; ++r) {
            unsigned rows_for_this_process = base_rows_per_process + (r < extra_rows ? 1 : 0);
            row_prefix_sums[r + 1] = row_prefix_sums[r] + rows_for_this_process;
        }

        // Calculate column distribution using prefix sums
        col_prefix_sums[0] = 0;
        unsigned base_cols_per_process = matrix_b_cols / process_grid_cols;
        unsigned extra_cols = matrix_b_cols % process_grid_cols;
        for (unsigned c = 0; c < process_grid_cols; ++c) {
            unsigned cols_for_this_process = base_cols_per_process + (c < extra_cols ? 1 : 0);
            col_prefix_sums[c + 1] = col_prefix_sums[c] + cols_for_this_process;
        }

        // Assign tile coordinates and dimensions to each process
        for (unsigned process_id = 0; process_id < total_processes; ++process_id) {
            unsigned tile_row_index = process_id / process_grid_cols;
            unsigned tile_col_index = process_id % process_grid_cols;
            
            tile_row_offsets[process_id] = row_prefix_sums[tile_row_index];
            tile_row_counts[process_id]  = row_prefix_sums[tile_row_index + 1] - row_prefix_sums[tile_row_index];
            tile_col_offsets[process_id] = col_prefix_sums[tile_col_index];
            tile_col_counts[process_id]  = col_prefix_sums[tile_col_index + 1] - col_prefix_sums[tile_col_index];
        }

        // Clean up prefix sum arrays
        free(row_prefix_sums);
        free(col_prefix_sums);

        // Distribute tiles to non-root processes
        for (int target_process = 1; target_process < total_processes; ++target_process) {
            // Send tile metadata (offsets and dimensions)
            unsigned tile_metadata[4] = {
                tile_row_offsets[target_process], 
                tile_row_counts[target_process], 
                tile_col_offsets[target_process], 
                tile_col_counts[target_process]
            };
            MPI_Send(tile_metadata, 4, MPI_UNSIGNED, target_process, 0, MPI_COMM_WORLD);

            unsigned tile_rows = tile_row_counts[target_process];
            unsigned tile_cols = tile_col_counts[target_process];

            // Calculate number of elements to send
            unsigned matrix_a_elements = tile_rows * matrix_a_cols;
            unsigned matrix_b_transposed_elements = tile_cols * matrix_b_rows;

            // Send Matrix A tile data
            if (matrix_a_elements > 0) {
                MPI_Send(matrix_a_data + ((size_t)tile_row_offsets[target_process] * (size_t)matrix_a_cols),
                         matrix_a_elements, MPI_UINT32_T, target_process, 1, MPI_COMM_WORLD);
            }

            // Send Matrix B transposed tile data
            if (matrix_b_transposed_elements > 0) {
                MPI_Send(matrix_b_transposed + ((size_t)tile_col_offsets[target_process] * (size_t)matrix_b_rows),
                         matrix_b_transposed_elements, MPI_UINT32_T, target_process, 2, MPI_COMM_WORLD);
            }
        }
    }

    // Each process receives its tile metadata
    unsigned local_tile_metadata[4] = {0, 0, 0, 0};
    if (process_rank == 0) {
        // Root process uses its own metadata
        local_tile_metadata[0] = tile_row_offsets[0];
        local_tile_metadata[1] = tile_row_counts[0];
        local_tile_metadata[2] = tile_col_offsets[0];
        local_tile_metadata[3] = tile_col_counts[0];
    } else {
        // Non-root processes receive metadata from root
        MPI_Recv(local_tile_metadata, 4, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Extract local tile dimensions and offsets
    unsigned local_tile_row_offset = local_tile_metadata[0];
    unsigned local_tile_rows = local_tile_metadata[1];
    unsigned local_tile_col_offset = local_tile_metadata[2];
    unsigned local_tile_cols = local_tile_metadata[3];

    // Local matrix data pointers for each process
    uint32_t *local_matrix_a_tile = NULL;
    uint32_t *local_matrix_b_transposed_tile = NULL;

    if (process_rank == 0) {
        // Root process uses direct pointers to global matrices
        local_matrix_a_tile = matrix_a_data + ((size_t)local_tile_row_offset * (size_t)matrix_a_cols);
        local_matrix_b_transposed_tile = matrix_b_transposed + ((size_t)local_tile_col_offset * (size_t)matrix_b_rows);
    } else {
        // Non-root processes allocate memory and receive data
        if (local_tile_rows > 0) {
            local_matrix_a_tile = malloc((size_t)local_tile_rows * (size_t)matrix_a_cols * sizeof(uint32_t));
            if (!local_matrix_a_tile) {
                fprintf(stderr, "[PROCESS %d] ERROR: Memory allocation failed for local Matrix A tile\n", process_rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            MPI_Recv(local_matrix_a_tile, local_tile_rows * matrix_a_cols, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (local_tile_cols > 0) {
            local_matrix_b_transposed_tile = malloc((size_t)local_tile_cols * (size_t)matrix_b_rows * sizeof(uint32_t));
            if (!local_matrix_b_transposed_tile) {
                fprintf(stderr, "[PROCESS %d] ERROR: Memory allocation failed for local Matrix B transposed tile\n", process_rank);
                free(local_matrix_a_tile);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            MPI_Recv(local_matrix_b_transposed_tile, local_tile_cols * matrix_b_rows, MPI_UINT32_T, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Allocate and initialize result matrix for this process
    uint64_t *local_result_matrix = NULL;
    if (local_tile_rows > 0 && local_tile_cols > 0) {
        local_result_matrix = calloc((size_t)local_tile_rows * (size_t)local_tile_cols, sizeof(uint64_t));
        if (!local_result_matrix) {
            fprintf(stderr, "[PROCESS %d] ERROR: Memory allocation failed for local result matrix\n", process_rank);
            if (process_rank != 0) {
                free(local_matrix_a_tile);
                free(local_matrix_b_transposed_tile);
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Determine number of OpenMP threads for computation
        int num_compute_threads = DEFAULT_COMPUTE_THREADS;
        char *omp_threads_env = getenv("OMP_NUM_THREADS");
        if (omp_threads_env != NULL) {
            int env_threads = atoi(omp_threads_env);
            if (env_threads > 0) {
                num_compute_threads = env_threads;
            }
        }

        // Perform parallel matrix multiplication on local tiles
        multiply_tile_block(local_matrix_a_tile, local_matrix_b_transposed_tile, local_result_matrix,
                            local_tile_rows, local_tile_cols, matrix_a_cols, num_compute_threads);
    }

    // Root process gathers results from all processes
    if (process_rank == 0) {
        // Copy root process's own result tile to global matrix
        if (local_tile_rows > 0 && local_tile_cols > 0) {
            copy_tile_into_global(local_result_matrix, result_matrix_c, matrix_b_cols,
                                  local_tile_row_offset, local_tile_col_offset,
                                  local_tile_rows, local_tile_cols);
        }

        // Gather result tiles from other processes
        for (int source_process = 1; source_process < total_processes; ++source_process) {
            unsigned received_metadata[4] = {0, 0, 0, 0};
            MPI_Recv(received_metadata, 4, MPI_UNSIGNED, source_process, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            unsigned received_row_offset = received_metadata[0];
            unsigned received_rows = received_metadata[1];
            unsigned received_col_offset = received_metadata[2];
            unsigned received_cols = received_metadata[3];

            if (received_rows == 0 || received_cols == 0) {
                continue;
            }

            // Allocate buffer for receiving tile data
            size_t tile_elements = (size_t)received_rows * (size_t)received_cols;
            uint64_t *receive_buffer = malloc(tile_elements * sizeof(uint64_t));
            if (!receive_buffer) {
                fprintf(stderr, "[ROOT] ERROR: Memory allocation failed while receiving tile from process %d\n", source_process);
                free(matrix_a_data);
                free(matrix_b_transposed);
                free(result_matrix_c);
                free(tile_row_offsets);
                free(tile_row_counts);
                free(tile_col_offsets);
                free(tile_col_counts);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            MPI_Recv(receive_buffer, tile_elements, MPI_UINT64_T,
                     source_process, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Copy received tile to global result matrix
            copy_tile_into_global(receive_buffer, result_matrix_c, matrix_b_cols,
                                  received_row_offset, received_col_offset,
                                  received_rows, received_cols);
            free(receive_buffer);
        }
    } else {
        // Non-root processes send their results to root
        MPI_Send(local_tile_metadata, 4, MPI_UNSIGNED, 0, 3, MPI_COMM_WORLD);
        if (local_tile_rows > 0 && local_tile_cols > 0) {
            MPI_Send(local_result_matrix, local_tile_rows * local_tile_cols, MPI_UINT64_T,
                     0, 4, MPI_COMM_WORLD);
        }
    }

    // Root process writes the final result to output file
    if (process_rank == 0) {
        computation_end_time = MPI_Wtime();
        printf("[ROOT] Matrix multiplication completed in %.6f seconds\n", 
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
            for (size_t row = 0; row < matrix_a_rows; ++row) {
                fwrite(&result_matrix_c[row * (size_t)matrix_b_cols],
                       sizeof(uint64_t), (size_t)matrix_b_cols, output_file);
            }
            fclose(output_file);
            printf("[ROOT] Result matrix written to %s\n", output_matrix_c_path);
        }
    }

    // Clean up local memory allocations
    if (process_rank != 0) {
        free(local_matrix_a_tile);
        free(local_matrix_b_transposed_tile);
    }
    free(local_result_matrix);

    // Root process cleans up global matrices and metadata
    if (process_rank == 0) {
        free(matrix_a_data);
        free(matrix_b_transposed);
        free(result_matrix_c);
        free(tile_row_offsets);
        free(tile_row_counts);
        free(tile_col_offsets);
        free(tile_col_counts);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}

/**
 * IMPLEMENTATION: Matrix Transpose Function
 * 
 * Transposes a matrix from source to destination buffer.
 * This function creates a transposed copy of the input matrix,
 * which is useful for optimizing cache performance during matrix multiplication.
 */
static void transpose_matrix(const uint32_t *source_matrix, size_t num_rows, size_t num_cols, uint32_t *transposed_matrix)
{
    for (size_t row = 0; row < num_rows; ++row) {
        for (size_t col = 0; col < num_cols; ++col) {
            transposed_matrix[col * num_rows + row] = source_matrix[row * num_cols + col];
        }
    }
}

/**
 * IMPLEMENTATION: In-Place Matrix Transpose
 * 
 * Performs in-place transpose for square matrices to save memory.
 * This function swaps elements across the diagonal, requiring only
 * half the matrix to be processed to avoid double-swapping.
 */
static void transpose_matrix_in_place(uint32_t *matrix, size_t dimension)
{
    // Process only upper triangle to avoid double-swapping elements
    for (size_t row = 1; row < dimension; ++row) {
        for (size_t col = 0; col < row; ++col) {
            uint32_t temp = matrix[row * dimension + col];
            matrix[row * dimension + col] = matrix[col * dimension + row];
            matrix[col * dimension + row] = temp;
        }
    }
}

/**
 * IMPLEMENTATION: Parallel Matrix File Reader
 * 
 * Reads a binary matrix file using parallel I/O with OpenMP threads.
 * This function uses POSIX pread() for thread-safe parallel reading,
 * where each thread reads a different set of rows simultaneously.
 */
static int read_matrix_parallel(const char *filename, uint32_t *rows, uint32_t *cols, uint32_t **matrix_data)
{
    // Open the binary file for reading
    int file_descriptor = open(filename, O_RDONLY);
    if (file_descriptor == -1) {
        fprintf(stderr, "ERROR: Unable to open file %s: %s\n", filename, strerror(errno));
        return -1;
    }

    // Read matrix dimensions from file header
    if (pread(file_descriptor, rows, sizeof(uint32_t), 0) != sizeof(uint32_t) ||
        pread(file_descriptor, cols, sizeof(uint32_t), sizeof(uint32_t)) != sizeof(uint32_t)) {
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
    *matrix_data = malloc(total_elements * sizeof(uint32_t));
    if (!*matrix_data) {
        fprintf(stderr, "ERROR: Memory allocation failed for matrix %s\n", filename);
        close(file_descriptor);
        return -1;
    }

    // Calculate I/O parameters
    size_t bytes_per_row = (size_t)(*cols) * sizeof(uint32_t);
    off_t data_start_offset = 2 * (off_t)sizeof(uint32_t);
    
    // Determine number of threads for parallel I/O
    int num_io_threads = DEFAULT_IO_THREADS;
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
                fprintf(stderr, "Thread %d: Failed to read row %d from %s\n", thread_id, row, filename);
                // Note: Cannot safely abort from within parallel region
            }
        }
    }

    close(file_descriptor);
    return 0;
}

/**
 * IMPLEMENTATION: Process Grid Layout Optimizer
 * 
 * Determines the optimal 2D process grid layout based on matrix dimensions.
 * This function finds the best factorization of the total number of processes
 * that most closely matches the aspect ratio of the matrices for balanced workload.
 */
static void choose_process_grid(int total_processes, uint32_t matrix_rows, uint32_t matrix_cols,
                                uint32_t *grid_rows, uint32_t *grid_cols)
{
    // Handle edge cases
    if (total_processes <= 0) {
        *grid_rows = 1;
        *grid_cols = 1;
        return;
    }

    // Calculate target aspect ratio for optimal load balancing
    double target_aspect_ratio = (double)matrix_rows / (double)matrix_cols;
    if (matrix_cols == 0) {
        target_aspect_ratio = 1.0;
    }

    // Initialize with default 1D layout
    int optimal_rows = 1;
    int optimal_cols = total_processes;
    double best_ratio_difference = fabs(((double)optimal_rows / (double)optimal_cols) - target_aspect_ratio);

    // Test all possible factorizations of total_processes
    int max_factor = 1;
    while (max_factor * max_factor <= total_processes) {
        max_factor++;
    }
    max_factor--; // Get the largest factor <= sqrt(total_processes)
    
    for (int rows = 1; rows <= max_factor; ++rows) {
        if (total_processes % rows != 0) {
            continue;
        }
        int cols = total_processes / rows;

        // Test both orientations: rows x cols and cols x rows
        double ratio_rc = fabs(((double)rows / (double)cols) - target_aspect_ratio);
        if (ratio_rc < best_ratio_difference) {
            best_ratio_difference = ratio_rc;
            optimal_rows = rows;
            optimal_cols = cols;
        }

        double ratio_cr = fabs(((double)cols / (double)rows) - target_aspect_ratio);
        if (ratio_cr < best_ratio_difference) {
            best_ratio_difference = ratio_cr;
            optimal_rows = cols;
            optimal_cols = rows;
        }
    }

    *grid_rows = optimal_rows;
    *grid_cols = optimal_cols;
}

/**
 * IMPLEMENTATION: Parallel Tile Matrix Multiplication
 * 
 * Performs matrix multiplication on tile blocks using OpenMP parallelization.
 * This function uses a 2D loop collapse for optimal thread utilization
 * and processes the transposed matrix B for better cache performance.
 */
static void multiply_tile_block(const uint32_t *matrix_a, const uint32_t *matrix_b_transposed,
                                uint64_t *result_matrix,
                                uint32_t tile_rows, uint32_t tile_cols, uint32_t common_dimension, int thread_count)
{
    // Handle edge cases
    if (tile_rows == 0 || tile_cols == 0) {
        return;
    }

    // Parallel matrix multiplication using OpenMP
#pragma omp parallel for collapse(2) num_threads(thread_count)
    for (size_t row = 0; row < tile_rows; ++row) {
        for (size_t col = 0; col < tile_cols; ++col) {
            uint64_t accumulator = 0;
            
            // Get pointers to current row and column
            const uint32_t *a_row_ptr = matrix_a + row * common_dimension;
            const uint32_t *bt_col_ptr = matrix_b_transposed + col * common_dimension;
            
            // Perform dot product with vectorization-friendly loop
            for (size_t k = 0; k < common_dimension; ++k) {
                accumulator += (uint64_t)a_row_ptr[k] * (uint64_t)bt_col_ptr[k];
            }
            
            // Store result in output matrix
            result_matrix[row * tile_cols + col] = accumulator;
        }
    }
}

/**
 * IMPLEMENTATION: Tile-to-Global Matrix Copy
 * 
 * Copies computed tile data back to the global result matrix.
 * This function efficiently copies tile data using memcpy for each row,
 * ensuring proper placement in the global matrix coordinate system.
 */
static void copy_tile_into_global(const uint64_t *tile_data, uint64_t *global_matrix,
                                  uint32_t global_cols, uint32_t row_start, uint32_t col_start,
                                  uint32_t tile_height, uint32_t tile_width)
{
    // Handle edge cases
    if (tile_height == 0 || tile_width == 0) {
        return;
    }

    // Copy each row of the tile to the corresponding position in global matrix
    for (size_t row = 0; row < tile_height; ++row) {
        uint64_t *destination = global_matrix + ((size_t)(row_start + row) * (size_t)global_cols + (size_t)col_start);
        const uint64_t *source = tile_data + (size_t)row * (size_t)tile_width;
        memcpy(destination, source, (size_t)tile_width * sizeof(uint64_t));
    }
}
