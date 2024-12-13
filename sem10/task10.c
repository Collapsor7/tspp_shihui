#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define IDX(i,j,k,Ny,Nz) ((i)*(Ny)*(Nz) + (j)*(Nz) + (k))

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 480;
    int n_iter = 100;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) n_iter = atoi(argv[2]);

    if ( rank == 0 ){
        printf("Iterations = %d \n",n_iter);
        printf("N = %d \n", N);
    }
   
    double start_time = MPI_Wtime();

    int dims[3] = {0,0,0};
    MPI_Dims_create(size, 3, dims);
    int Px = dims[0];
    int Py = dims[1];
    int Pz = dims[2];

    int periods[3] = {0,0,0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    int local_nx = N / Px;
    int local_ny = N / Py;
    int local_nz = N / Pz;

    if (rank == 0 ){
        printf("Sub-three-dimensional grid: %d x %d x %d \n", local_nx, local_ny, local_nz);
    }

    int NX = local_nx + 2;
    int NY = local_ny + 2;
    int NZ = local_nz + 2;

    double *u_old = (double*)malloc(sizeof(double)*NX*NY*NZ);
    double *u_new = (double*)malloc(sizeof(double)*NX*NY*NZ);
    double *f     = (double*)malloc(sizeof(double)*NX*NY*NZ);

    srand(rank+1);
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                int idx = IDX(i,j,k,NY,NZ);
                u_old[idx] = 0.0;
                u_new[idx] = 0.0;
                f[idx] = ((double)rand()/RAND_MAX)*1.0;
            }
        }
    }

    int nbr_x_minus, nbr_x_plus;
    int nbr_y_minus, nbr_y_plus;
    int nbr_z_minus, nbr_z_plus;

    {
        int x_minus_coord = coords[0]-1;
        if (x_minus_coord < 0) {
            nbr_x_minus = MPI_PROC_NULL;
        } else {
            int ncoords[3] = {x_minus_coord, coords[1], coords[2]};
            MPI_Cart_rank(cart_comm, ncoords, &nbr_x_minus);
        }

        int x_plus_coord = coords[0]+1;
        if (x_plus_coord >= Px) {
            nbr_x_plus = MPI_PROC_NULL;
        } else {
            int ncoords[3] = {x_plus_coord, coords[1], coords[2]};
            MPI_Cart_rank(cart_comm, ncoords, &nbr_x_plus);
        }
    }

    {
        int y_minus_coord = coords[1]-1;
        if (y_minus_coord < 0) {
            nbr_y_minus = MPI_PROC_NULL;
        } else {
            int ncoords[3] = {coords[0], y_minus_coord, coords[2]};
            MPI_Cart_rank(cart_comm, ncoords, &nbr_y_minus);
        }

        int y_plus_coord = coords[1]+1;
        if (y_plus_coord >= Py) {
            nbr_y_plus = MPI_PROC_NULL;
        } else {
            int ncoords[3] = {coords[0], y_plus_coord, coords[2]};
            MPI_Cart_rank(cart_comm, ncoords, &nbr_y_plus);
        }
    }

    {
        int z_minus_coord = coords[2]-1;
        if (z_minus_coord < 0) {
            nbr_z_minus = MPI_PROC_NULL;
        } else {
            int ncoords[3] = {coords[0], coords[1], z_minus_coord};
            MPI_Cart_rank(cart_comm, ncoords, &nbr_z_minus);
        }

        int z_plus_coord = coords[2]+1;
        if (z_plus_coord >= Pz) {
            nbr_z_plus = MPI_PROC_NULL;
        } else {
            int ncoords[3] = {coords[0], coords[1], z_plus_coord};
            MPI_Cart_rank(cart_comm, ncoords, &nbr_z_plus);
        }
    }

    MPI_Datatype yz_plane_type;
    MPI_Type_contiguous(local_ny*local_nz, MPI_DOUBLE, &yz_plane_type);
    MPI_Type_commit(&yz_plane_type);

    MPI_Datatype y_plane_type;
    MPI_Type_vector(local_nx, local_nz, NY*NZ, MPI_DOUBLE, &y_plane_type);
    MPI_Type_commit(&y_plane_type);

    MPI_Aint lb=0, extent;
    MPI_Type_get_extent(y_plane_type, &lb, &extent);
    MPI_Datatype y_plane_resized;
    MPI_Type_create_resized(y_plane_type, 0, extent, &y_plane_resized);
    MPI_Type_commit(&y_plane_resized);

    MPI_Datatype z_line_type;
    MPI_Type_vector(local_ny, 1, NZ, MPI_DOUBLE, &z_line_type);
    MPI_Type_commit(&z_line_type);

    MPI_Datatype z_plane_type;
    MPI_Type_create_hvector(local_nx, 1, (MPI_Aint)(NY*NZ*sizeof(double)), z_line_type, &z_plane_type);
    MPI_Type_commit(&z_plane_type);

    struct {
        double val;
        int flag;
    } meta_data;
    meta_data.val = 3.14;
    meta_data.flag = 42;

    MPI_Datatype meta_type;
    MPI_Aint base_addr, addr_val, addr_flag;
    MPI_Get_address(&meta_data, &base_addr);
    MPI_Get_address(&meta_data.val, &addr_val);
    MPI_Get_address(&meta_data.flag, &addr_flag);

    MPI_Aint disps[2];
    disps[0] = addr_val - base_addr;
    disps[1] = addr_flag - base_addr;

    int blocklen[2] = {1,1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
    MPI_Type_create_struct(2, blocklen, disps, types, &meta_type);
    MPI_Type_commit(&meta_type);

    double h = 1.0/(double)N;

    for (int iter = 0; iter < n_iter; iter++) {

        if (coords[0] % 2 == 0) {
            if (nbr_x_plus != MPI_PROC_NULL) 
                MPI_Send(&u_old[IDX(local_nx,1,1,NY,NZ)], 1, yz_plane_type, nbr_x_plus, 0, cart_comm);
            if (nbr_x_minus != MPI_PROC_NULL) 
                MPI_Recv(&u_old[IDX(0,1,1,NY,NZ)], 1, yz_plane_type, nbr_x_minus, 0, cart_comm, MPI_STATUS_IGNORE);

            if (nbr_x_minus != MPI_PROC_NULL) 
                MPI_Send(&u_old[IDX(1,1,1,NY,NZ)], 1, yz_plane_type, nbr_x_minus, 1, cart_comm);
            if (nbr_x_plus != MPI_PROC_NULL) 
                MPI_Recv(&u_old[IDX(local_nx+1,1,1,NY,NZ)], 1, yz_plane_type, nbr_x_plus, 1, cart_comm, MPI_STATUS_IGNORE);
        } else {

            if (nbr_x_minus != MPI_PROC_NULL) 
                MPI_Recv(&u_old[IDX(0,1,1,NY,NZ)], 1, yz_plane_type, nbr_x_minus, 0, cart_comm, MPI_STATUS_IGNORE);
            if (nbr_x_plus != MPI_PROC_NULL) 
                MPI_Send(&u_old[IDX(local_nx,1,1,NY,NZ)], 1, yz_plane_type, nbr_x_plus, 0, cart_comm);

            if (nbr_x_plus != MPI_PROC_NULL) 
                MPI_Recv(&u_old[IDX(local_nx+1,1,1,NY,NZ)], 1, yz_plane_type, nbr_x_plus, 1, cart_comm, MPI_STATUS_IGNORE);
            if (nbr_x_minus != MPI_PROC_NULL) 
                MPI_Send(&u_old[IDX(1,1,1,NY,NZ)], 1, yz_plane_type, nbr_x_minus, 1, cart_comm);
        }

        if (coords[1] % 2 == 0) {
            if (nbr_y_plus != MPI_PROC_NULL)
                MPI_Send(&u_old[IDX(1,local_ny,1,NY,NZ)], 1, y_plane_resized, nbr_y_plus, 2, cart_comm);
            if (nbr_y_minus != MPI_PROC_NULL)
                MPI_Recv(&u_old[IDX(1,0,1,NY,NZ)], 1, y_plane_resized, nbr_y_minus, 2, cart_comm, MPI_STATUS_IGNORE);

            if (nbr_y_minus != MPI_PROC_NULL)
                MPI_Send(&u_old[IDX(1,1,1,NY,NZ)], 1, y_plane_resized, nbr_y_minus, 3, cart_comm);
            if (nbr_y_plus != MPI_PROC_NULL)
                MPI_Recv(&u_old[IDX(1,local_ny+1,1,NY,NZ)], 1, y_plane_resized, nbr_y_plus, 3, cart_comm, MPI_STATUS_IGNORE);
        } else {
            if (nbr_y_minus != MPI_PROC_NULL)
                MPI_Recv(&u_old[IDX(1,0,1,NY,NZ)], 1, y_plane_resized, nbr_y_minus, 2, cart_comm, MPI_STATUS_IGNORE);
            if (nbr_y_plus != MPI_PROC_NULL)
                MPI_Send(&u_old[IDX(1,local_ny,1,NY,NZ)], 1, y_plane_resized, nbr_y_plus, 2, cart_comm);

            if (nbr_y_plus != MPI_PROC_NULL)
                MPI_Recv(&u_old[IDX(1,local_ny+1,1,NY,NZ)], 1, y_plane_resized, nbr_y_plus, 3, cart_comm, MPI_STATUS_IGNORE);
            if (nbr_y_minus != MPI_PROC_NULL)
                MPI_Send(&u_old[IDX(1,1,1,NY,NZ)], 1, y_plane_resized, nbr_y_minus, 3, cart_comm);
        }

        if (coords[2] % 2 == 0) {
            if (nbr_z_plus != MPI_PROC_NULL)
                MPI_Send(&u_old[IDX(1,1,local_nz,NY,NZ)], 1, z_plane_type, nbr_z_plus, 4, cart_comm);
            if (nbr_z_minus != MPI_PROC_NULL)
                MPI_Recv(&u_old[IDX(1,1,0,NY,NZ)], 1, z_plane_type, nbr_z_minus, 4, cart_comm, MPI_STATUS_IGNORE);

            if (nbr_z_minus != MPI_PROC_NULL)
                MPI_Send(&u_old[IDX(1,1,1,NY,NZ)], 1, z_plane_type, nbr_z_minus, 5, cart_comm);
            if (nbr_z_plus != MPI_PROC_NULL)
                MPI_Recv(&u_old[IDX(1,1,local_nz+1,NY,NZ)], 1, z_plane_type, nbr_z_plus, 5, cart_comm, MPI_STATUS_IGNORE);
        } else {
            if (nbr_z_minus != MPI_PROC_NULL)
                MPI_Recv(&u_old[IDX(1,1,0,NY,NZ)], 1, z_plane_type, nbr_z_minus, 4, cart_comm, MPI_STATUS_IGNORE);
            if (nbr_z_plus != MPI_PROC_NULL)
                MPI_Send(&u_old[IDX(1,1,local_nz,NY,NZ)], 1, z_plane_type, nbr_z_plus, 4, cart_comm);

            if (nbr_z_plus != MPI_PROC_NULL)
                MPI_Recv(&u_old[IDX(1,1,local_nz+1,NY,NZ)], 1, z_plane_type, nbr_z_plus, 5, cart_comm, MPI_STATUS_IGNORE);
            if (nbr_z_minus != MPI_PROC_NULL)
                MPI_Send(&u_old[IDX(1,1,1,NY,NZ)], 1, z_plane_type, nbr_z_minus, 5, cart_comm);
        }

        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j <= local_ny; j++) {
                for (int k = 1; k <= local_nz; k++) {
                    int idx = IDX(i,j,k,NY,NZ);
                    double u_xm = u_old[IDX(i-1,j,k,NY,NZ)];
                    double u_xp = u_old[IDX(i+1,j,k,NY,NZ)];
                    double u_ym = u_old[IDX(i,j-1,k,NY,NZ)];
                    double u_yp = u_old[IDX(i,j+1,k,NY,NZ)];
                    double u_zm = u_old[IDX(i,j,k-1,NY,NZ)];
                    double u_zp = u_old[IDX(i,j,k+1,NY,NZ)];
                    double rhs = f[idx];

                    u_new[idx] = (u_xm+u_xp+u_ym+u_yp+u_zm+u_zp - h*h*rhs)/6.0;
                }
            }
        }

        double *tmp = u_old; u_old = u_new; u_new = tmp;

        if (iter == n_iter - 1 || iter == 0) {
            double local_diff = 0.0;
            for (int i = 1; i <= local_nx; i++) {
                for (int j = 1; j <= local_ny; j++) {
                    for (int k = 1; k <= local_nz; k++) {
                        int idx = IDX(i,j,k,NY,NZ);
                        double d = (u_old[idx] - u_new[idx]);
                        local_diff += d*d;
                    }
                }
            }
            double global_diff;
            MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
            global_diff = sqrt(global_diff);

            if (rank == 0) {
                if (iter == 0 ){
                    printf("Initinal norm = %e\n", global_diff);
                } else {
                    printf("Final norm = %e \n", global_diff);
                }
            }
        }
    }
    double end_time = MPI_Wtime();
    if (rank == 0){
        double local_execution_time = end_time - start_time;
        printf("Total execution time (T): %f seconds\n", local_execution_time);  
    }

    MPI_Type_free(&yz_plane_type);
    MPI_Type_free(&y_plane_type);
    MPI_Type_free(&y_plane_resized);
    MPI_Type_free(&z_line_type);
    MPI_Type_free(&z_plane_type);
    MPI_Type_free(&meta_type);

    free(u_old);
    free(u_new);
    free(f);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}