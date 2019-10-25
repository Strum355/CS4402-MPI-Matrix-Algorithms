#include <math.h>
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

int** alloc_matrix(int n, int m);
void init_matrix(int n, int** a, int p);
int** prod_matrix(int n, int l, int m, int** a, int** b);
void print_matrix(int n, int m, int** a);
void extract_matrix(int na, int ** a, int nb, int** b, int row, int col);
void implant_matrix(int na, int** a, int nb, int** b, int row, int col);

int** MPI_Row_split(int n, int** mat_A, int** mat_B);
int** MPI_Col_split(int n, int** mat_A, int** mat_B);

void Scatter_elements_A_broadcast_elements_B(int n, int** mat_A, int** mat_B);

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
    if(rank == 0) {
        mat_A = alloc_matrix(n, n);
        init_matrix(n, mat_A, 1);
        
        print_matrix(n, n, mat_A);
        printf("\n");
        
        init_matrix(n, mat_B, 2);

        print_matrix(n, n, mat_B);
        printf("\n");
    }

    Scatter_elements_A_broadcast_elements_B(n, mat_A, mat_B);

    MPI_Finalize();

    return 0;
}

void Scatter_elements_A_broadcast_elements_B(int n, int** mat_A, int** mat_B) {
    int** local_a = alloc_matrix(n/size, n);
    
    MPI_Scatter(&mat_A[0][0], n*n/size, MPI_INT, &local_a[0][0], n*n/size, MPI_INT, 0, MPI_COMM_WORLD);
    print_matrix(n/size, n, local_a);

    printf("\n");

    MPI_Bcast(&mat_B[0][0], n*n, MPI_INT, 0, MPI_COMM_WORLD);

    print_matrix(n/size, n,  prod_matrix(n/size, n, n, local_a, mat_B));
    printf("\n");
}


int** MPI_Col_split(int n, int** mat_A, int** mat_B) {
    MPI_Datatype send_col_type, recv_col_type;

    // sending column type
    MPI_Type_vector(n, 1, n, MPI_INT, &send_col_type);
    MPI_Type_create_resized(send_col_type, 0, sizeof(int), &send_col_type);
    MPI_Type_commit(&send_col_type);

    // recv column type
    MPI_Type_vector(n, 1, n/size, MPI_INT, &recv_col_type);
    MPI_Type_create_resized(recv_col_type, 0, sizeof(int), &recv_col_type);
    MPI_Type_commit(&recv_col_type);

    int** local_mat = alloc_matrix(n, n/size);

    MPI_Scatter(&mat_A[0][0], n/size, send_col_type, &local_mat[0][0], n/size, recv_col_type, 0, MPI_COMM_WORLD);
    
    /* print_matrix(n, n/size, local_mat);
    printf("\n"); */

    MPI_Type_free(&recv_col_type);
    MPI_Type_free(&send_col_type);
    
    return local_mat;
}


int** MPI_Row_split(int n, int** mat_A, int** mat_B) {
    MPI_Datatype row_type;
    MPI_Type_contiguous(n, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    int** local_mat = alloc_matrix(n/size, n);
    
    MPI_Scatter(&mat_A[0][0], n/size, row_type, &local_mat[0][0], n/size, row_type, 0, MPI_COMM_WORLD);

    /* print_matrix(n/size, n, local_mat);
    printf("\n"); */

    MPI_Type_free(&row_type);
    return local_mat;
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

/*
    n - number of rows of a
    l - number of columns of a // the matrix b must have l rows
    m - number of columns of b

    a - the first matrix
    b - the second matrix
*/
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

void print_matrix(int rows, int cols, int** a) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf(" %d ",a[i][j]);
        }
        printf("\n");
    }
}

void extract_matrix(int na, int ** a, int nb, int** b, int row, int col) {
    if(na<row+nb || na<col+nb) {
        printf("Impossible to extract");
        return;
    }

    for(int i = 0; i < nb; i++)
        for(int j = 0; j < nb; j++)
            b[i][j] = a[row+i][col+j];
}

void implant_matrix(int na, int** a, int nb, int** b, int row, int col) {
    if(na<row+nb || na<col+nb) {
        printf("Impossible to extract");
        return;
    }

    for(int i = 0; i < nb; i++)
        for(int j = 0; j < nb ; j++)
            a[row+i][col+j] = b[i][j];
}