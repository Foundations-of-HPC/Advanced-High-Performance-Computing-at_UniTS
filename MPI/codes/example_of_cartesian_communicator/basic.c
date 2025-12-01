
#include <stdio.h>
#include <stdio.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
  int world_size, world_rank;
  MPI_Comm cart_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // We want a 2D grid
  int ndims = 2;
  int dims[2] = {0, 0};        // 0 means "please choose"
  int periods[2] = {0, 0};     // non-periodic in both dimensions
  int reorder = 1;             // allow MPI to reorder ranks

  // Let MPI figure out a good grid decomposition
  MPI_Dims_create(world_size, ndims, dims);

  if (world_rank == 0) {
    printf("Using a %d x %d process grid\n", dims[0], dims[1]);
  }

  // Create the Cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);

  // If the size is incompatible (very rare if dims[] is valid), cart_comm may be MPI_COMM_NULL
  if (cart_comm == MPI_COMM_NULL) {
    if (world_rank == 0) {
      fprintf(stderr, "Failed to create Cartesian communicator\n");
    }
    MPI_Finalize();
    return 1;
  }

  int cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);

  int coords[2];
  MPI_Cart_coords(cart_comm, cart_rank, ndims, coords);
  int row = coords[0];
  int col = coords[1];

  // Find neighbors in each direction
  int up, down, left, right;

  // direction 0 = rows (up/down), direction 1 = columns (left/right)
  MPI_Cart_shift(cart_comm, 0, 1, &up,   &down);
  MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

  // Simple halo exchange: send our cart_rank to each neighbor and receive from them.
  // We'll store -1 where there is no neighbor.
  int from_up   = -1;
  int from_down = -1;
  int from_left = -1;
  int from_right= -1;

  MPI_Request reqs[8];
  int nreq = 0;

  // Post receives first
  if (up != MPI_PROC_NULL) {
    MPI_Irecv(&from_up, 1, MPI_INT, up,   0, cart_comm, &reqs[nreq++]);
  }
  if (down != MPI_PROC_NULL) {
    MPI_Irecv(&from_down, 1, MPI_INT, down, 1, cart_comm, &reqs[nreq++]);
  }
  if (left != MPI_PROC_NULL) {
    MPI_Irecv(&from_left, 1, MPI_INT, left, 2, cart_comm, &reqs[nreq++]);
  }
  if (right != MPI_PROC_NULL) {
    MPI_Irecv(&from_right, 1, MPI_INT, right, 3, cart_comm, &reqs[nreq++]);
  }

  // Now sends
  if (up != MPI_PROC_NULL) {
    MPI_Isend(&cart_rank, 1, MPI_INT, up,   1, cart_comm, &reqs[nreq++]);
  }
  if (down != MPI_PROC_NULL) {
    MPI_Isend(&cart_rank, 1, MPI_INT, down, 0, cart_comm, &reqs[nreq++]);
  }
  if (left != MPI_PROC_NULL) {
    MPI_Isend(&cart_rank, 1, MPI_INT, left, 3, cart_comm, &reqs[nreq++]);
  }
  if (right != MPI_PROC_NULL) {
    MPI_Isend(&cart_rank, 1, MPI_INT, right, 2, cart_comm, &reqs[nreq++]);
  }

  // Wait for all communication to complete
  MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

  // Synchronize to make output cleaner
  MPI_Barrier(cart_comm);

  // Each process reports its coordinates and neighbors
  printf("WORLD %2d -> CART %2d at (%d,%d): "
	 "up=%2d down=%2d left=%2d right=%2d | "
	 "from_up=%2d from_down=%2d from_left=%2d from_right=%2d\n",
	 world_rank, cart_rank, row, col,
	 up, down, left, right,
	 from_up, from_down, from_left, from_right);

  MPI_Comm_free(&cart_comm);
  MPI_Finalize();
  return 0;
}

