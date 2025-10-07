#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#define SHIFT_ROW 0
#define SHIFT_COL 1
#define DISP 1

int main(int argc, char *argv[])
{
    int ndims = 2, size, my_rank, reorder = 1, my_cart_rank, ierr;
    int nrows, ncols;
    int nbr_i_lo, nbr_i_hi;  /* top, bottom */
    int nbr_j_lo, nbr_j_hi;  /* left, right */
    MPI_Comm comm2D = MPI_COMM_NULL;
    int dims[2], coord[2];
    int wrap_around[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* process command line arguments */
    if (argc == 3) {
        nrows = atoi(argv[1]);
        ncols = atoi(argv[2]);
        dims[0] = nrows; /* number of rows */
        dims[1] = ncols; /* number of columns */
        if (nrows <= 0 || ncols <= 0 || (nrows * ncols) != size) {
            if (my_rank == 0) {
                printf("ERROR: nrows*ncols = %d * %d = %d != %d\n",
                       nrows, ncols, nrows * ncols, size);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        /* Let MPI choose a good decomposition */
        dims[0] = dims[1] = 0;
    }

    /* create cartesian topology for processes */
    MPI_Dims_create(size, ndims, dims);

    if (my_rank == 0) {
        printf("Root Rank: %d. Comm Size: %d: Grid Dimension = [%d x %d]\n",
               my_rank, size, dims[0], dims[1]);
    }

    /* create cartesian mapping (non-periodic in both dims) */
    wrap_around[0] = 0; /* rows */
    wrap_around[1] = 0; /* cols */

    ierr = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, wrap_around, reorder, &comm2D);
    if (ierr != MPI_SUCCESS || comm2D == MPI_COMM_NULL) {
        if (my_rank == 0) printf("ERROR creating CART, ierr=%d\n", ierr);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    /* find my rank in the cartesian communicator group */
    MPI_Comm_rank(comm2D, &my_cart_rank);

    /* find my coordinates in the cartesian communicator group */
    MPI_Cart_coords(comm2D, my_cart_rank, ndims, coord);

    /* alternatively: from coords -> rank
       MPI_Cart_rank(comm2D, coord, &my_cart_rank); */

    /* get my neighbors */
    MPI_Cart_shift(comm2D, SHIFT_ROW, DISP, &nbr_i_lo, &nbr_i_hi);  /* top, bottom */
    MPI_Cart_shift(comm2D, SHIFT_COL, DISP, &nbr_j_lo, &nbr_j_hi);  /* left, right */

    printf("Global rank: %d. Cart rank: %d. Coord: (%d, %d). "
           "Left: %d. Right: %d. Top: %d. Bottom: %d\n",
           my_rank, my_cart_rank, coord[0], coord[1],
           nbr_j_lo, nbr_j_hi, nbr_i_lo, nbr_i_hi);
    fflush(stdout);

    MPI_Comm_free(&comm2D);
    MPI_Finalize();
    return 0;
}
