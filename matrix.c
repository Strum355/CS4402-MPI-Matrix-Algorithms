#include <math.h>
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

int** alloc_matrix(int n, int m);
void init_matrix(int n, int** a, int p);
int** prod_matrix(int n, int l, int m, int** a, int** b);
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

int main(int argc, char **argv)
{
    int n = 4;

    int** mat_A;
    int** mat_B;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mat_B = alloc_matrix(n, n);
    mat_A = alloc_matrix(n, n);
    if(rank == 0) {
        init_matrix(n, mat_A, 1);
        
        printf("matrix A:\n");
        print_matrix(n, n, mat_A);
        printf("\n");
        
        init_matrix(n, mat_B, 2);

        printf("matrix B:\n");
        print_matrix(n, n, mat_B);
        printf("\n");
    }

    /* Scatter_elements_A_broadcast_elements_B(n, mat_A, mat_B);

    MPI_Barrier(MPI_COMM_WORLD);

    Scatter_rows_A_broadcast_rows_B(n, mat_A, mat_B);

    Scatter_rows_A_broadcast_rows_B_transposed(n, mat_A, mat_B);

    Broadcast_cols_A_scatter_cols_b(n, mat_A, mat_B);*/

    Cannon(n, mat_A, mat_B);

    MPI_Finalize();

    return 0;
}

void Scatter_elements_A_broadcast_elements_B(int n, int** mat_A, int** mat_B) {
    int** local_a = alloc_matrix(n/size, n);
    
    MPI_Scatter(&mat_A[0][0], n*n/size, MPI_INT, &local_a[0][0], n*n/size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&mat_B[0][0], n*n, MPI_INT, 0, MPI_COMM_WORLD);

    int** mult = prod_matrix(n/size, n, n, local_a, mat_B);

    int** gathered_mult = alloc_matrix(n, n);
    MPI_Gather(&mult[0][0], n*n/size, MPI_INT, &gathered_mult[0][0], n*n/size, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        print_matrix(n, n, gathered_mult);
        printf("\n");
    }
}

void Scatter_rows_A_broadcast_rows_B(int n, int** mat_A, int** mat_B) {
    MPI_Datatype row_type;
    MPI_Type_contiguous(n, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    int** local_a = alloc_matrix(n/size, n);

    MPI_Scatter(&mat_A[0][0], n/size, row_type, &local_a[0][0], n/size, row_type, 0, MPI_COMM_WORLD);

    MPI_Bcast(&mat_B[0][0], n, row_type, 0, MPI_COMM_WORLD);

    int** mult = prod_matrix(n/size, n, n, local_a, mat_B);

    int** gathered_mult = alloc_matrix(n, n);
    MPI_Gather(&mult[0][0], n/size, row_type, &gathered_mult[0][0], n/size, row_type, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        print_matrix(n, n, gathered_mult);
        printf("\n");
    }
}

void Scatter_rows_A_broadcast_rows_B_transposed(int n, int** mat_A, int** mat_B) {
    MPI_Datatype row_type;
    MPI_Type_contiguous(n, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    int** local_a = alloc_matrix(n/size, n);

    MPI_Scatter(&mat_A[0][0], n/size, row_type, &local_a[0][0], n/size, row_type, 0, MPI_COMM_WORLD);

    int** transed_mat;
    if(rank == 0) {
        transed_mat = trans_matrix(n, n, mat_B);
    } else {
        transed_mat = alloc_matrix(n, n);
    }

    MPI_Bcast(&transed_mat[0][0], n, row_type, 0, MPI_COMM_WORLD);

    int** mult = pseudo_prod_matrix(n/size, n, n, local_a, transed_mat);

    int** gathered_mult = alloc_matrix(n, n);
    MPI_Gather(&mult[0][0], n/size, row_type, &gathered_mult[0][0], n/size, row_type, 0, MPI_COMM_WORLD);

    if(rank == 0)print_matrix(n, n, gathered_mult);
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
 
    MPI_Bcast(&mat_A[0][0], n, send_col_type, 0, MPI_COMM_WORLD);

    int** local_b = alloc_matrix(n, n/size);

    MPI_Scatter(&mat_B[0][0], n/size, send_col_type, &local_b[0][0], n/size, recv_col_type, 0, MPI_COMM_WORLD);

    int** mult = prod_matrix(n, n, n/size, mat_A, local_b);

    int** gathered_mult = alloc_matrix(n, n);
    MPI_Gather(&mult[0][0], n/size, recv_col_type, &gathered_mult[0][0], n/size, send_col_type, 0, MPI_COMM_WORLD);
    if(rank == 0) print_matrix(n, n, gathered_mult);
}

void Cannon(int n, int** mat_A, int** mat_B) {
    int dims[2], periods[2], coords[2], leftCoords[2], rightCoords[2], upCoords[2], downCoords[2], sendCoords[2], recvCoords[2];
    int leftRank, rightRank, upRank, downRank, recvRank, sendRank;

    MPI_Comm grid;
    
    int p = (int)sqrt(size);
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

    if(rank == 0) {
        for(int i = 0; i < p; i++) {
            for(int j = 0; j < p; j++) {
                extract_matrix(n, n, mat_A, n/p, n/p, local_a, i*n/p, j*n/p);
                /* print_matrix(n/p, n/p, local_a);
                printf("\n"); */
                recvCoords[0] = i;
                recvCoords[1] = j;
                MPI_Cart_rank(grid, recvCoords, &recvRank);
                MPI_Request req;
                MPI_Status status;
                MPI_Isend(&mat_B[0][0], n*n/size, MPI_INT, recvRank, 1, MPI_COMM_WORLD, &req);
                MPI_Wait(&req, &status);
                printf("%d\n", status.MPI_SOURCE);
            }
        }
    }

    if(rank == 0) {
        print_matrix(n*n/size, n*n/size, mat_B);
    }
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
        printf("Impossible to extract");
        return;
    }

    for(int i = 0; i < nb; i++)
        for(int j = 0; j < mb; j++)
            b[i][j] = a[row + i][col + j];
}

void implant_matrix(int na, int ma, int ** a, int nb, int mb, int ** b, int row, int col) {
    if(na < row + nb || na < col + mb) {
        printf("Impossible to extract");
        return;
    }

    for(int i = 0; i < nb; i++)
        for(int j = 0; j < mb; j++)
            a[row + i][col + j] = b[i][j];
}