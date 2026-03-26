// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "config.h"
#include "matrix.h"
#include "logging.h"
#include "math.h"

// Helper macro for indexing
#define MAT_IDX(mat, i, j) ((mat)->meta.transpose ? (j) * (mat)->meta.cols + (i) : (i) * (mat)->meta.cols + (j))

uint32_t matmul(matrix_2D_t* A, matrix_2D_t* B, matrix_2D_t* C)
{
    int A_rows = A->meta.transpose ? A->meta.cols : A->meta.rows;
    int A_cols = A->meta.transpose ? A->meta.rows : A->meta.cols;
    int B_rows = B->meta.transpose ? B->meta.cols : B->meta.rows;
    int B_cols = B->meta.transpose ? B->meta.rows : B->meta.cols;

    if (A_cols != B_rows) {
        LOG_ERROR("Matrix dimension mismatch for multiplication\n");
        return 1;
    }

    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
                float a = A->data[MAT_IDX(A, i, k)];
                float b = B->data[MAT_IDX(B, k, j)];
                sum += a * b;
            }
            C->data[i * C->meta.cols + j] = sum;
        }
    }
    return 0;
}

void matscale(matrix_2D_t* A, float scalar)
{
    for (int i = 0; i < A->meta.rows; ++i) {
        for (int j = 0; j < A->meta.cols; ++j) {
            A->data[i * A->meta.cols + j] *= scalar;
        }
    }
    return;
}

void vecprint(float* vec, uint16_t n, uint8_t log_level)
{
    if (LOG_LEVEL >= log_level)
    {
        printf("Vec %d [\n",n);
        for(int i = 0; i<n; i++)
        {
            printf("%f,\n", vec[i]);
        }
        printf("]\n");
    }
}

void matprint(matrix_2D_t* A, uint8_t log_level)
{
    if (LOG_LEVEL >= log_level){
        int A_rows = A->meta.transpose ? A->meta.cols : A->meta.rows;
        int A_cols = A->meta.transpose ? A->meta.rows : A->meta.cols;
        printf("Matrix %d x %d\n",A_rows,A_cols);
        for(int i = 0; i<A_rows; i++)
        {
            printf("[");
            for(int j = 0; j<A_cols; j++)
            {
                printf("%f,", A->data[MAT_IDX(A, i, j)]);
            }
            printf("]\n");
        }
    }
}

void mateye(matrix_2D_t* A)
{
    for (int i = 0; i < A->meta.rows; ++i) {
        for (int j = 0; j < A->meta.cols; ++j) {
            if(i==j){
                A->data[i * A->meta.cols + j] = 1.0;
            }
            else{
                A->data[i * A->meta.cols + j] = 0.0;
            }
        }
    }
    return;
}

void matfill(matrix_2D_t* A, float b)
{
    for (int i = 0; i < A->meta.rows; ++i) {
        for (int j = 0; j < A->meta.cols; ++j) {
            A->data[i * A->meta.cols + j] = b;
        }
    }
    return;
}

float frobenius_norm(matrix_2D_t* A)
{
    float norm = 0.0;
    for (int i = 0; i < A->meta.rows; ++i) {
        for (int j = 0; j < A->meta.cols; ++j) {
            norm += A->data[i * A->meta.cols + j]*A->data[i * A->meta.cols + j];
        }
    }
    return sqrtf(norm);
}

float determinant(matrix_2D_t* A)
{
    if(A->meta.rows == 2 && A->meta.cols == 2)
    {
        return A->data[0]*A->data[3] - A->data[1]*A->data[2];
    }
    if(A->meta.rows == 3 && A->meta.cols == 3)
    {
        float sum = 0;
        sum+= A->data[0]*(A->data[4]*A->data[8] - A->data[5]*A->data[7]);
        sum-= A->data[3]*(A->data[1]*A->data[8] - A->data[2]*A->data[7]);
        sum+= A->data[6]*(A->data[1]*A->data[5] - A->data[2]*A->data[4]);
        return sum;
    }
    LOG_ERROR("\nDETERMINANT NOT IMPLEMENTED FOR GIVEN MATRIX DIMENSIONS\n\n");
    return 0.0;
}

void matvec(matrix_2D_t* A, float* b, float* c, uint8_t n){
    matrix_2D_t B = {b,{n,1,0,0}};
    matrix_2D_t C = {c,{n,1,0,0}};
    matmul(A,&B,&C);
}

uint8_t matinv3x3(matrix_2D_t* A, matrix_2D_t* B)
{
    float det = determinant(A);
    if (fabsf(det) < NUMERICAL_ZERO_THRESHOLD) 
    {return 0;}
    float invdet = 1/det;
    // Compute the adjugate matrix and multiply by 1/det
    B->data[0*3+0] =  (A->data[1*3+1]*A->data[2*3+2] - A->data[1*3+2]*A->data[2*3+1]) * invdet;
    B->data[0*3+1] = -(A->data[0*3+1]*A->data[2*3+2] - A->data[0*3+2]*A->data[2*3+1]) * invdet;
    B->data[0*3+2] =  (A->data[0*3+1]*A->data[1*3+2] - A->data[0*3+2]*A->data[1*3+1]) * invdet;
    B->data[1*3+0] = -(A->data[1*3+0]*A->data[2*3+2] - A->data[1*3+2]*A->data[2*3+0]) * invdet;
    B->data[1*3+1] =  (A->data[0*3+0]*A->data[2*3+2] - A->data[0*3+2]*A->data[2*3+0]) * invdet;
    B->data[1*3+2] = -(A->data[0*3+0]*A->data[1*3+2] - A->data[0*3+2]*A->data[1*3+0]) * invdet;
    B->data[2*3+0] =  (A->data[1*3+0]*A->data[2*3+1] - A->data[1*3+1]*A->data[2*3+0]) * invdet;
    B->data[2*3+1] = -(A->data[0*3+0]*A->data[2*3+1] - A->data[0*3+1]*A->data[2*3+0]) * invdet;
    B->data[2*3+2] =  (A->data[0*3+0]*A->data[1*3+1] - A->data[0*3+1]*A->data[1*3+0]) * invdet;
    return 1;
}

void normalize(float* v, uint8_t n) {
    float norm = 0.0;
    for (int i = 0; i < n; i++)
        norm += v[i] * v[i];
    if (norm > VECTOR_ZERO_NORM_THRESHOLD){
        norm = sqrtf(norm);
        for (int i = 0; i < n; i++)
            v[i] /= norm;
    }
}

// Dot product
float dot(float* a, float* b, uint8_t n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

// Helper function to swap two float values
void swap_float(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

void solve_linear_gaussian(float* A, float* b, float* x, uint8_t n) {
    float tempA[n*n];
    float tempB[n];
    
    // Copy A and b into temporary buffers to avoid modifying the input.
    for (int i = 0; i < n; i++) {
        tempB[i] = b[i];
        for (int j = 0; j < n; j++)
            tempA[i*n+j] = A[i*n+j];
    }

    // Forward elimination with partial pivoting
    for (int k = 0; k < n; k++) {
        // Find the row with the largest pivot element in the current column k
        int max_row_idx = k;
        for (int i = k + 1; i < n; i++) {
            if (fabsf(tempA[i*n+k]) > fabsf(tempA[max_row_idx*n+k])) {
                max_row_idx = i;
            }
        }

        // If the pivot row is not the current row, swap them
        if (max_row_idx != k) {
            // Swap the entire rows in the matrix tempA
            for (int j = k; j < n; j++) {
                swap_float(&tempA[k*n+j], &tempA[max_row_idx*n+j]);
            }
            // Swap the corresponding elements in the vector tempB
            swap_float(&tempB[k], &tempB[max_row_idx]);
        }

        if (fabsf(tempA[k*n+k]) < NUMERICAL_ZERO_THRESHOLD) {
            LOG_DEBUG("Matrix is singular or nearly singular. Cannot solve.\n");
            return;
        }

        // Perform elimination for rows below the pivot
        for (int i = k + 1; i < n; i++) {
            float factor = tempA[i*n+k] / tempA[k*n+k];
            for (int j = k; j < n; j++) {
                tempA[i*n+j] -= factor * tempA[k*n+j];
            }
            tempB[i] -= factor * tempB[k];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        x[i] = tempB[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= tempA[i*n+j] * x[j];
        }
        x[i] /= tempA[i*n+i];
    }
}

// Orthonormalize U_prime into U via modified Gram-Schmidt
void orthonormalize(matrix_2D_t* U_prime, matrix_2D_t* U) {
    int n = U_prime->meta.cols;
    int m = U_prime->meta.rows;
    float col[m];
    for (int j = 0; j < n; ++j) {
        // Copy U'_j to U_j
        for (int i = 0; i < m; ++i)
            U->data[i*n+j] = U_prime->data[i*n+j];

        // Subtract projections onto previous vectors
        for (int k = 0; k < j; ++k) {
            float dot = 0.0;
            for (int i = 0; i < m; ++i)
                dot += U->data[i*n+j] * U->data[i*n+k];

            for (int i = 0; i < m; ++i)
                U->data[i*n+j] -= dot * U->data[i*n+k];
        }

        // Normalize column j
        for (int i = 0; i < m; ++i)
            col[i] = U->data[i*n+j];
        normalize(col, m);
        for (int i = 0; i < m; ++i)
            U->data[i*n+j] = col[i];
    }
}

// Inverse power method
float inverse_power_method(float* A, float* eigvec, uint8_t n) {
    float x[n], y[n], lambda_old = 0.0, lambda = 0.0;

    // Initial vector
    for (int i = 0; i < n; i++)
        x[i] = 1.0;
    normalize(x,n);

    for (int iter = 0; iter < MAX_ITER_INV_POWER; iter++) {
        // Solve Ay = x
        solve_linear_gaussian(A, x, y, n);
        normalize(y, n);

        // Rayleigh quotient estimate of eigenvalue
        lambda = dot(y, x, n);
        float diff = fabsf(lambda - lambda_old);
        for (int i = 0; i < n; i++)
            x[i] = y[i];
        if (isnan(diff) || diff < TOL_INV_POWER)
            break;
        lambda_old = lambda;
    }

    for (int i = 0; i < n; i++)
        eigvec[i] = x[i];

    return lambda;
}

// Identity matrix
void identity_matrix(float* V, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            V[i * n + j] = (i == j) ? 1.0 : 0.0;
}

// Jacobi eigenvalue algorithm
void jacobi_eigen(float* A, float* V, float* d, int n, int max_iterations, float tolerance) {
    identity_matrix(V, n);

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        float max = fabsf(A[p * n + q]);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (fabsf(A[i * n + j]) > max) {
                    max = fabsf(A[i * n + j]);
                    p = i;
                    q = j;
                }
            }
        }

        // Converged
        if (max < tolerance) break;

        float app = A[p * n + p];
        float aqq = A[q * n + q];
        float apq = A[p * n + q];
        float phi = 0.5 * atan2f(2.0 * apq, aqq - app);
        float c = cosf(phi);
        float s = sinf(phi);

        // Apply rotation
        for (int i = 0; i < n; ++i) {
            float aip = A[i * n + p];
            float aiq = A[i * n + q];
            A[i * n + p] = c * aip - s * aiq;
            A[i * n + q] = s * aip + c * aiq;
        }
        for (int i = 0; i < n; ++i) {
            float api = A[p * n + i];
            float aqi = A[q * n + i];
            A[p * n + i] = c * api - s * aqi;
            A[q * n + i] = s * api + c * aqi;
        }

        A[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p * n + q] = A[q * n + p] = 0.0;

        // Rotate eigenvector matrix
        for (int i = 0; i < n; ++i) {
            float vip = V[i * n + p];
            float viq = V[i * n + q];
            V[i * n + p] = c * vip - s * viq;
            V[i * n + q] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues
    for (int i = 0; i < n; ++i)
        d[i] = A[i * n + i];
}

void svd(matrix_2D_t* A, matrix_2D_t* U, float* S, matrix_2D_t* V, work_memory_t work_memory)
{
    uint8_t cols = A->meta.cols;
    uint8_t rows = A->meta.rows;
    float* ATA_data = allocate_work_memory(&work_memory, cols*cols*sizeof(float));
    matrix_2D_t At = {A->data, {rows, cols, 1, 0}};
    matrix_2D_t ATA = {ATA_data, {cols, cols, 0, 0}};
    matmul(&At, A, &ATA);
    float* eigenvalues = allocate_work_memory(&work_memory, cols*sizeof(float));
    jacobi_eigen(ATA.data, V->data, eigenvalues, cols, SVD_MAX_ITER_JACOBI, SVD_TOL_JACOBI);

    // Sort eigenvalues and corresponding eigenvectors in descending order
    uint8_t indices[cols];
    for (int i = 0; i < cols; ++i){
        indices[i] = i;
    }
    for (int i = 0; i < cols-1; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            if (eigenvalues[indices[i]] < eigenvalues[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    float* sortedV = allocate_work_memory(&work_memory, cols*cols*sizeof(float));
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < cols; ++j){
            sortedV[j * cols + i] = V->data[j * cols + indices[i]];
        }
        S[i] = sqrtf(eigenvalues[indices[i]]);
    }
    for (int i = 0; i < cols*cols; ++i) {
        V->data[i] = sortedV[i];
    }

    // Compute U = A * V * S^-1
    float* S_inv_data = allocate_work_memory(&work_memory, cols*cols*sizeof(float));
    float* temp_data = allocate_work_memory(&work_memory, cols*cols*sizeof(float));
    memset(S_inv_data,0,cols*cols*sizeof(float));
    for (int i = 0; i < cols; ++i) {
        if (S[i] > NUMERICAL_ZERO_THRESHOLD && (!isnan(S[i]))){
            S_inv_data[i*cols+i] = 1 / S[i];
        }
    }
    matrix_2D_t S_inv = {S_inv_data, {cols, cols, 0, 0}};
    matrix_2D_t temp = {temp_data, {cols, cols, 0, 0}};
    float* u_prime = allocate_work_memory(&work_memory, rows*rows*sizeof(float));
    matrix_2D_t U_prime = {u_prime, {rows, rows, 0, 0}};
    matmul(V, &S_inv, &temp);
    matmul(A, &temp, &U_prime);
    for (int i = 0; i < rows; ++i) {
        if (S_inv_data[i*cols+i] < NUMERICAL_ZERO_THRESHOLD)
        {
            U_prime.data[i*cols+i] = 1.0;
        }
    }
    orthonormalize(&U_prime, U);
}

void apply_householder(float* v, float* x, int n) {
    float beta = 2.0f * dot(v, x, n);
    for (int i = 0; i < n; ++i) x[i] -= beta * v[i];
}

// Solve upper-triangular Rx = c, R is n x n
void back_substitution(float* R, float* c, float* x, int n) {
    for (int i = n - 1; i >= 0; --i) {
        x[i] = c[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= R[i * n + j] * x[j];
        x[i] /= R[i * n + i];
    }
}

void solve_linear_qr(matrix_2D_t* A, matrix_2D_t* b, matrix_2D_t* x, work_memory_t work_memory) {
    uint8_t m = A->meta.rows;
    uint8_t n = A->meta.cols;

    // Allocate temporary matrices. R_temp will be an in-place modification of A.
    // c_temp will be an in-place modification of b.
    float* R_temp = allocate_work_memory(&work_memory, m * n * sizeof(float));
    float* c_temp = allocate_work_memory(&work_memory, m * sizeof(float));

    // Copy A and b to temporary storage, as they will be modified.
    for (int i = 0; i < m * n; ++i) R_temp[i] = A->data[i];
    for (int i = 0; i < m; ++i) c_temp[i] = b->data[i];
    
    // Allocate space for one Householder vector
    float* v = allocate_work_memory(&work_memory, m * sizeof(float));

    // For each column, find and apply the Householder reflection
    for (int k = 0; k < n; ++k) {
        // --- 1. Compute the Householder vector v for the current column k ---
        float col_norm = 0.0f;
        for (int i = k; i < m; ++i) {
            col_norm += R_temp[i * n + k] * R_temp[i * n + k];
        }
        col_norm = sqrtf(col_norm);

        // If the column segment is all zeros, no reflection is needed
        if (col_norm < NUMERICAL_ZERO_THRESHOLD) continue;
        
        // v = x + sign(x_0) * ||x|| * e_0
        for (int i = 0; i < m; ++i) {
            v[i] = (i < k) ? 0.0f : R_temp[i * n + k];
        }
        float R_kk = R_temp[k * n + k];
        v[k] += (R_kk >= 0) ? col_norm : -col_norm;
        
        // Normalize the Householder vector v
        float v_norm = 0.0f;
        for (int i = k; i < m; ++i) {
            v_norm += v[i] * v[i];
        }
        v_norm = sqrtf(v_norm);
        
        if (v_norm < NUMERICAL_ZERO_THRESHOLD) continue;
        
        for (int i = k; i < m; ++i) {
            v[i] /= v_norm;
        }

        // --- 2. Apply the reflection (I - 2vv^T) to the remaining submatrix of R ---
        for (int j = k; j < n; ++j) {
            float dot_prod = 0.0f;
            for (int i = k; i < m; ++i) {
                dot_prod += v[i] * R_temp[i * n + j];
            }
            float beta = 2.0f * dot_prod;
            for (int i = k; i < m; ++i) {
                R_temp[i * n + j] -= beta * v[i];
            }
        }

        // --- 3. Apply the same reflection to the vector c_temp (to compute Q^T*b) ---
        float dot_prod_c = 0.0f;
        for (int i = k; i < m; ++i) {
            dot_prod_c += v[i] * c_temp[i];
        }
        float beta_c = 2.0f * dot_prod_c;
        for (int i = k; i < m; ++i) {
            c_temp[i] -= beta_c * v[i];
        }
    }

    back_substitution(R_temp, c_temp, x->data, n);
}