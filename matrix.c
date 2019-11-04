#include <math.h>
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

void print_results(char* fmt, double total, double comm, double exec);

int** alloc_matrix(int n, int m);
void init_matrix(int n, int** a, int p);
int** prod_matrix(int n, int l, int m, int** a, int** b);
int** add_matrix(int n, int m, int** a, int** b);
int** pseudo_prod_matrix(int n, int l, int m, int** a, int** b);
void print_matrix(int n, int m, int** a);
void extract_matrix(int na, int ma, int ** a, int nb, int mb, int ** b, int row, int col);
void implant_matrix(int na, int ma, int ** a, int nb, int mb, int ** b, int row, int col);
int** trans_matrix(int n, int m, int** a);

void Scatter_elements_A_broadcast_elements_B(int n, int** mat_A, int** mat_B);
void Scatter_rows_A_broadcast_rows_B(int n, int** mat_A, int** mat_B);
void Scatter_rows_A_broadcast_rows_B_transposed(int n, int** mat_A, int** mat_B);
void Broadcast_cols_A_scatter_cols_b(int n, int** mat_A, int** mat_B);
void Cannon(int n, int** mat_A, int** mat_B);

int rank, size;
double total_time, comm_time, comp_time;

int main(int argc, char **argv)
{
    int n = 2000;

    int** mat_A;
    int** mat_B;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mat_B = alloc_matrix(n, n);
    mat_A = alloc_matrix(n, n);
    if(rank == 0) {
        init_matrix(n, mat_A, 1);
        
        /* printf("matrix A:\n");
        print_matrix(n, n, mat_A);
        printf("\n"); */
        
        init_matrix(n, mat_B, 2);

        /* printf("matrix B:\n");
        print_matrix(n, n, mat_B);
        printf("\n"); */
    }

    if(rank == 0) {
        printf("Matrix size: %d*%d\tProcessors: %d\n\n", n, n, size);
        printf("Method\t\t\t\t\t\tTotal Time\tComm Time\tExec Time\n");
        printf("==========================================================================================\n");
    }

    {
        total_time = MPI_Wtime();
        Scatter_elements_A_broadcast_elements_B(n, mat_A, mat_B);
        
        print_results("Scatter_elements_A_broadcast_elements_B:\t%lfs\t%lfs\t%lfs\n", MPI_Wtime() - total_time, comm_time, comp_time);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    {
        total_time = MPI_Wtime();
        Scatter_rows_A_broadcast_rows_B(n, mat_A, mat_B);
        
        print_results("Scatter_rows_A_broadcast_rows_B:\t\t%lfs\t%lfs\t%lfs\n", MPI_Wtime() - total_time, comm_time, comp_time);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    {
        total_time = MPI_Wtime();
        Scatter_rows_A_broadcast_rows_B_transposed(n, mat_A, mat_B);
        
        print_results("Scatter_rows_A_broadcast_rows_B_transposed:\t%lfs\t%lfs\t%lfs\n", MPI_Wtime() - total_time, comm_time, comp_time);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    {
        total_time = MPI_Wtime();
        Broadcast_cols_A_scatter_cols_b(n, mat_A, mat_B);
        
        print_results("Broadcast_cols_A_scatter_cols_b:\t\t%lfs\t%lfs\t%lfs\n", MPI_Wtime() - total_time, comm_time, comp_time);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    {
        total_time = MPI_Wtime();
        Cannon(n, mat_A, mat_B);
        print_results("Cannons Algorithm:\t\t\t\t%lfs\t%lfs\t%lfs\n", MPI_Wtime() - total_time, comm_time, comp_time);
    }
    
    MPI_Finalize();

    return 0;
}

void print_results(char* fmt, double total, double comm, double exec) {
    if(rank == 0) {
        printf(fmt, total, comm, exec);
    }
}

void Scatter_elements_A_broadcast_elements_B(int n, int** mat_A, int** mat_B) {
    int** local_a = alloc_matrix(n/size, n);
    
    comm_time = MPI_Wtime();
    MPI_Scatter(&mat_A[0][0], n*n/size, MPI_INT, &local_a[0][0], n*n/size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mat_B[0][0], n*n, MPI_INT, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;

    comp_time = MPI_Wtime();
    int** mult = prod_matrix(n/size, n, n, local_a, mat_B);
    comp_time = MPI_Wtime() - comp_time;

    int** gathered_mult = alloc_matrix(n, n);

    comm_time += MPI_Wtime() - comm_time;
    MPI_Gather(&mult[0][0], n*n/size, MPI_INT, &gathered_mult[0][0], n*n/size, MPI_INT, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;
    /* if(rank == 0) {
        print_matrix(n, n, gathered_mult);
        printf("\n");
    } */
}

void Scatter_rows_A_broadcast_rows_B(int n, int** mat_A, int** mat_B) {
    MPI_Datatype row_type;
    MPI_Type_contiguous(n, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    int** local_a = alloc_matrix(n/size, n);

    comm_time = MPI_Wtime();
    MPI_Scatter(&mat_A[0][0], n/size, row_type, &local_a[0][0], n/size, row_type, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mat_B[0][0], n, row_type, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;

    comp_time = MPI_Wtime();
    int** mult = prod_matrix(n/size, n, n, local_a, mat_B);
    comp_time = MPI_Wtime() - comp_time;

    int** gathered_mult = alloc_matrix(n, n);

    comm_time += MPI_Wtime() - comm_time;
    MPI_Gather(&mult[0][0], n/size, row_type, &gathered_mult[0][0], n/size, row_type, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;
    /* if(rank == 0) {
        print_matrix(n, n, gathered_mult);
        printf("\n");
    } */
}

void Scatter_rows_A_broadcast_rows_B_transposed(int n, int** mat_A, int** mat_B) {
    MPI_Datatype row_type;
    MPI_Type_contiguous(n, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    int** local_a = alloc_matrix(n/size, n);

    comm_time = MPI_Wtime();
    MPI_Scatter(&mat_A[0][0], n/size, row_type, &local_a[0][0], n/size, row_type, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;

    int** transed_mat;
    if(rank == 0) {
        transed_mat = trans_matrix(n, n, mat_B);
    } else {
        transed_mat = alloc_matrix(n, n);
    }

    comm_time += MPI_Wtime() - comm_time;
    MPI_Bcast(&transed_mat[0][0], n, row_type, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;

    comp_time = MPI_Wtime();
    int** mult = pseudo_prod_matrix(n/size, n, n, local_a, transed_mat);
    comp_time = MPI_Wtime() - comp_time;

    int** gathered_mult = alloc_matrix(n, n);
    comm_time += MPI_Wtime() - comm_time;
    MPI_Gather(&mult[0][0], n/size, row_type, &gathered_mult[0][0], n/size, row_type, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;

    /* if(rank == 0) {
        print_matrix(n, n, gathered_mult);
        printf("\n");
    } */
}

void Broadcast_cols_A_scatter_cols_b(int n, int** mat_A, int** mat_B) {
    MPI_Datatype send_col_type, recv_col_type;

    // send column type
    MPI_Type_vector(n, 1, n, MPI_INT, &send_col_type);
    MPI_Type_create_resized(send_col_type, 0, sizeof(int), &send_col_type);
    MPI_Type_commit(&send_col_type);

    // recv column type
    MPI_Type_vector(n, 1, n/size, MPI_INT, &recv_col_type);
    MPI_Type_create_resized(recv_col_type, 0, sizeof(int), &recv_col_type);
    MPI_Type_commit(&recv_col_type);
 
    comm_time = MPI_Wtime();
    MPI_Bcast(&mat_A[0][0], n, send_col_type, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;

    int** local_b = alloc_matrix(n, n/size);

    comm_time += MPI_Wtime() - comm_time;
    MPI_Scatter(&mat_B[0][0], n/size, send_col_type, &local_b[0][0], n/size, recv_col_type, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;

    comp_time = MPI_Wtime();
    int** mult = prod_matrix(n, n, n/size, mat_A, local_b);
    comp_time = MPI_Wtime() - comp_time;

    int** gathered_mult = alloc_matrix(n, n);
    comm_time += MPI_Wtime() - comm_time;
    MPI_Gather(&mult[0][0], n/size, recv_col_type, &gathered_mult[0][0], n/size, send_col_type, 0, MPI_COMM_WORLD);
    comm_time = MPI_Wtime() - comm_time;

    /* if(rank == 0) {
        print_matrix(n, n, gathered_mult);
        printf("\n");
    } */
}

void Cannon(int n, int** mat_A, int** mat_B) {
    int dims[2], periods[2], coords[2], leftCoords[2], rightCoords[2], upCoords[2], downCoords[2], sendCoords[2], recvCoords[2];
    int leftRank, rightRank, upRank, downRank, recvRank, sendRank;

    MPI_Comm grid;
    
    int p = (int)sqrt(size);
    if(p * p != size) {
        printf("square grid not posible with %d procs\n", size);
        MPI_Finalize();
        return;
    }
    dims[0] = dims[1] = p;
    periods[0] = periods[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);

    MPI_Comm_rank(grid, &rank);
    MPI_Cart_coords(grid, rank, 2, coords);

    leftCoords[0] = (coords[0] - 1)%p; leftCoords[1] = coords[1];
    rightCoords[0] = (coords[0] + 1)%p; rightCoords[1] = coords[1];
    upCoords[0] = coords[0]; upCoords[1] = (coords[1]-1)%p;
    downCoords[0] = coords[0]; downCoords[1] = (coords[1]+1)%p;

    MPI_Cart_rank(grid, leftCoords, &leftRank);
    MPI_Cart_rank(grid, rightCoords, &rightRank);
    MPI_Cart_rank(grid, upCoords, &upRank);
    MPI_Cart_rank(grid, downCoords, &downRank);

    int** local_a = alloc_matrix(n/p, n/p);
    int** local_b = alloc_matrix(n/p, n/p);

    // reset values because we have a loop
    comp_time = 0;
    double comp_time_temp = 0;
    comm_time = 0;
    double comm_time_accum = 0;

    // pre-comp section to init every proc with starting data
    {
        if(rank == 0) {
            for(int i = 0; i < p; i++) {
                for(int j = 0; j < p; j++) {
                    comp_time_temp = MPI_Wtime();
                    extract_matrix(n, n, mat_A, n/p, n/p, local_a, i*n/p,  (j-i+p)%p*n/p);
                    comp_time += MPI_Wtime() - comp_time_temp;

                    recvCoords[0] = j;
                    recvCoords[1] = i;
                    MPI_Cart_rank(grid, recvCoords, &recvRank);

                    comm_time_accum = MPI_Wtime();
                    MPI_Request _;
                    MPI_Isend(&local_a[0][0], n*n/(p*p), MPI_INT, recvRank, 1, MPI_COMM_WORLD, &_);
                    comm_time += MPI_Wtime() - comm_time_accum;

                    comp_time_temp = MPI_Wtime();
                    extract_matrix(n, n, mat_B, n/p, n/p, local_b, (i-j+p)%p*n/p, j*n/p);
                    comp_time += MPI_Wtime() - comp_time_temp;

                    MPI_Cart_rank(grid, recvCoords, &recvRank);
                    comm_time_accum = MPI_Wtime();
                    MPI_Isend(&local_b[0][0], n*n/(p*p), MPI_INT, recvRank, 2, MPI_COMM_WORLD, &_);
                    comm_time += MPI_Wtime() - comm_time_accum;
                }
            }
        }

        comm_time_accum = MPI_Wtime();
        MPI_Request _;
        MPI_Irecv(&local_a[0][0], n*n/(p*p), MPI_INT, 0, 1, MPI_COMM_WORLD, &_);
        MPI_Irecv(&local_b[0][0], n*n/(p*p), MPI_INT, 0, 2, MPI_COMM_WORLD, &_);
        comm_time += MPI_Wtime() - comm_time_accum;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // multiplying, accumulating and shifting
    int** local_c = alloc_matrix(n/p, n/p);
    {
        MPI_Cart_rank(grid, leftCoords, &leftRank);
        MPI_Cart_rank(grid, rightCoords, &rightRank);
        MPI_Cart_rank(grid, upCoords, &upRank);
        MPI_Cart_rank(grid, downCoords, &downRank);

        for(int i = 0; i < p; i++) {
            comp_time_temp = MPI_Wtime();
            local_c = add_matrix(n/p, n/p, local_c, prod_matrix(n/p, n/p, n/p, local_a, local_b));
            comp_time += MPI_Wtime() - comp_time_temp;

            comm_time_accum = MPI_Wtime();
            MPI_Sendrecv_replace(&local_a[0][0], n*n/(p*p), MPI_INT, leftRank, 1, rightRank, 1, grid, MPI_STATUS_IGNORE);
            MPI_Sendrecv_replace(&local_b[0][0], n*n/(p*p), MPI_INT, upRank, 2, downRank, 2, grid, MPI_STATUS_IGNORE);
            comm_time += MPI_Wtime() - comm_time_accum;
        }
    }

    int** result_matrix = alloc_matrix(n, n);

    // gather results from all partitions and implant into result_matrix
    {
        comm_time_accum = MPI_Wtime();
        MPI_Request _;
        MPI_Isend(&local_c[0][0], n*n/(p*p), MPI_INT, 0, rank, MPI_COMM_WORLD, &_);
        comm_time += MPI_Wtime() - comm_time_accum;

        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) {
            for(int i = 0; i < size; i++) {
                int** local_d = alloc_matrix(n/p, n/p);

                comm_time_accum = MPI_Wtime();
                MPI_Irecv(&local_d[0][0], n*n/(p*p), MPI_INT, i, i, MPI_COMM_WORLD, &_);
                comm_time += MPI_Wtime() - comm_time_accum;
                
                int coords[2];
                MPI_Cart_coords(grid, i, size, coords);
                comp_time_temp = MPI_Wtime();
                implant_matrix(n, n, result_matrix, n/p, n/p, local_d, coords[1]*(n/p), coords[0]*(n/p));
                comp_time += MPI_Wtime() - comp_time_temp;
            }
        }
    }

    /* if(rank == 0) {
        printf("\n");
        print_matrix(n, n, result_matrix);
    } */
}

int** alloc_matrix(int rows, int cols) {
    int* aa = (int*)calloc(rows*cols, sizeof(int));
    int** a = (int**)calloc(rows, sizeof(int*));

    for(int i = 0; i < rows; i++)
        a[i]=aa+i*cols;

    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            a[i][j]=0;

    return a;
}

void init_matrix(int n, int** a, int p) {
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            a[i][j] = rand() % 6/* i/p+j/p */;

    return;
}

int** trans_matrix(int n, int m, int** a) {
	int** b = alloc_matrix(m, n);

	for(int j = 0; j < m; j++)
        for(int i = 0; i < n; i++)
            b[j][i] = a[i][j];

	return b;
}

int** prod_matrix(int n, int l, int m, int** a, int** b) {
	int** c = alloc_matrix(n, m);

	for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++){
            c[i][j] = 0;
            for(int k = 0; k < l; k++){
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
	}

	return c;
}

int** pseudo_prod_matrix(int n, int l, int m, int** a, int** b) {
    int** c = alloc_matrix(n, m);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            c[i][j] = 0;
            for(int k = 0; k < l; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[j][k];
            }
        }
    }
    return c;
}

int** add_matrix(int n, int m, int** a, int** b) {
    int** sum = alloc_matrix(n, m);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            sum[i][j] = a[i][j] + b[i][j];
        }
    }
    return sum;
}

void print_matrix(int rows, int cols, int** a) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf(" %d ",a[i][j]);
        }
        printf("\n");
    }
}

void extract_matrix(int na, int ma, int** a, int nb, int mb, int** b, int row, int col) {
    if(na < row + nb || na < col + mb) {
        printf("Impossible to extract\n");
        return;
    }

    for(int i = 0; i < nb; i++)
        for(int j = 0; j < mb; j++)
            b[i][j] = a[row + i][col + j];
}

void implant_matrix(int na, int ma, int ** a, int nb, int mb, int ** b, int row, int col) {
    if(na < row + nb || na < col + mb) {
        printf("Impossible to implant\n");
        return;
    }

    for(int i = 0; i < nb; i++)
        for(int j = 0; j < mb; j++)
            a[row + i][col + j] = b[i][j];
}