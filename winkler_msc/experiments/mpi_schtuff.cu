#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Comm cart_comm;
    int dims[3];  // Dimensions of the grid
    int periods[3] = {0, 0, 0};  // Non-periodic boundary conditions
    int coords[3];  // Coordinates of this process in the grid
    int neighbor_rank[6];  // Ranks of neighboring processes

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate dimensions for a cubic grid
    int cube_root = round(pow(size, 1.0 / 3.0));
    if (cube_root * cube_root * cube_root != size) {
        printf("Number of processes must be a perfect cube\n");
        MPI_Finalize();
        return 1;
    }
    dims[0] = dims[1] = dims[2] = cube_root;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create a 3D Cartesian grid communicator
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    // Get coordinates of this process in the grid
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // Get ranks of neighboring processes
    MPI_Cart_shift(cart_comm, 0, 1, &neighbor_rank[0], &neighbor_rank[1]);
    MPI_Cart_shift(cart_comm, 1, 1, &neighbor_rank[2], &neighbor_rank[3]);
    MPI_Cart_shift(cart_comm, 2, 1, &neighbor_rank[4], &neighbor_rank[5]);
    if(rank == 0){
        printf("Rank at pos %d, %d, %d has neighbores:\n", coords[0], coords[1], coords[2]);
        for (int i = 0; i < 6; i++){
            if(neighbor_rank[i] >= 0){
                MPI_Cart_coords(cart_comm, neighbor_rank[i], 3, coords);
                printf("    %d, %d, %d \n", coords[0], coords[1], coords[2]);
            }
            //printf("    %d \n", neighbor_rank[i]);
        }
    }
    // Exchange a message between adjacent ranks
    int send_buf = rank;
    int recv_buf;
    MPI_Request requests[6];
    for (int i = 0; i < 6; i++) {
        MPI_Isend(&send_buf, 1, MPI_INT, neighbor_rank[i], 0, cart_comm, &requests[i]);
    }
    for (int i = 0; i < 6; i++) {
        MPI_Recv(&recv_buf, 1, MPI_INT, neighbor_rank[i], 0, cart_comm, MPI_STATUS_IGNORE);
        //printf("Rank %d received message from Rank %d\n", rank, recv_buf);
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}