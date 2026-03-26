// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "pmsis.h"
#include "work_memory.h"

typedef struct
{
	uint8_t rows;
	uint8_t cols;
    uint8_t transpose;
    uint8_t misc;
} matrix_meta_2D_t;

typedef struct
{
    float* data;
    matrix_meta_2D_t meta;
} matrix_2D_t;

/**
 * @brief Performs matrix multiplication: C = A * B.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @param C Pointer to the output matrix.
 * @return Status code: Success (0), mismatched matrix dimensions (1).
 */
uint32_t matmul(matrix_2D_t* A, matrix_2D_t* B, matrix_2D_t* C);
 
/**
 * @brief Scales all elements of matrix A by a given scalar.
 * @param A Pointer to the matrix to scale.
 * @param scalar The scaling factor.
 */
void matscale(matrix_2D_t* A, float scalar);
 
/**
 * @brief Prints the contents of a vector with fprintf().
 * @param vec Pointer to the vector.
 * @param n Number of elements in the vector.
 * @param log_level Logging verbosity level.
 */
void vecprint(float* vec, uint16_t n, uint8_t log_level);
 
/**
 * @brief Prints the contents of a matrix with fprintf().
 * @param A Pointer to the matrix.
 * @param log_level Logging verbosity level.
 */
void matprint(matrix_2D_t* A, uint8_t log_level);
 
/**
 * @brief Sets matrix A to the identity matrix.
 * @param A Pointer to the matrix to modify.
 */
void mateye(matrix_2D_t* A);
 
/**
 * @brief Fills all elements of matrix A with the value b.
 * @param A Pointer to the matrix to fill.
 * @param b Value to fill the matrix with.
 */
void matfill(matrix_2D_t* A, float b);
 
/**
 * @brief Computes the Frobenius norm of matrix A.
 * @param A Pointer to the matrix.
 * @return The Frobenius norm.
 */
float frobenius_norm(matrix_2D_t* A);
 
/**
 * @brief Computes the determinant of matrix A (supports only 2x2 and 3x3 matrices).
 * @param A Pointer to the matrix.
 * @return The determinant value.
 */
float determinant(matrix_2D_t* A);
 
/**
 * @brief Multiplies matrix A by vector b, stores result in vector c.
 * @param A Pointer to the matrix.
 * @param b Pointer to the input vector.
 * @param c Pointer to the output vector.
 * @param n Number of elements in the vector.
 */
void matvec(matrix_2D_t* A, float* b, float* c, uint8_t n);
 
/**
 * @brief Computes the inverse of a 3x3 matrix A and stores it in B.
 * @param A Pointer to the input 3x3 matrix.
 * @param B Pointer to the output matrix (inverse).
 * @return Status code (1 for success, zero for failure).
 */
uint8_t matinv3x3(matrix_2D_t* A, matrix_2D_t* B);
 
/**
 * @brief Normalizes a vector to unit length.
 * @param v Pointer to the vector.
 * @param n Number of elements in the vector.
 */
void normalize(float* v, uint8_t n);
 
/**
 * @brief Computes the dot product of two vectors.
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param n Number of elements in the vectors.
 * @return The dot product value.
 */
float dot(float* a, float* b, uint8_t n);

/**
 * @brief Solves a linear system A*x = b using Gaussian elimination with partial pivoting.
 * @param A Pointer to the matrix data (n x n).
 * @param b Pointer to the vector data (n x 1).
 * @param x Pointer to the output solution vector (n x 1).
 * @param n The dimension of the system.
 */
void solve_linear_gaussian(float* A, float* b, float* x, uint8_t n);

/**
 * @brief Orthonormalizes the columns of matrix U and stores the result in U_prime.
 *
 * @param U_prime Pointer to the output matrix to store the orthonormalized columns.
 * @param U Pointer to the input matrix whose columns are to be orthonormalized.
 */
void orthonormalize(matrix_2D_t* U_prime, matrix_2D_t* U);

/**
 * @brief Computes the smallest eigenvalue and corresponding eigenvector of matrix A using the inverse power method.
 *
 * @param A Pointer to the input square matrix (flattened array).
 * @param eigvec Pointer to the output array to store the computed eigenvector.
 * @param n Size of the matrix (number of rows/columns).
 * @return The smallest eigenvalue found.
 */
float inverse_power_method(float* A, float* eigvec, uint8_t n);

/**
 * @brief Computes all eigenvalues and eigenvectors of a real symmetric matrix using the Jacobi method.
 *
 * @param A Pointer to the input square matrix (flattened array).
 * @param V Pointer to the output matrix to store eigenvectors (flattened array).
 * @param d Pointer to the output array to store eigenvalues.
 * @param n Size of the matrix (number of rows/columns).
 * @param max_iterations Maximum number of iterations to perform.
 * @param tolerance Convergence tolerance for off-diagonal elements.
 */
void jacobi_eigen(float* A, float* V, float* d, int n, int max_iterations, float tolerance);

/**
 * @brief Computes the Singular Value Decomposition (SVD) of a square matrix.
 *
 *        Only supports square matrices, but can be used for solving overdetermined linear systems.
 *
 * @param A Pointer to the input matrix to decompose.
 * @param U Pointer to the output matrix to store left singular vectors.
 * @param S Pointer to the output array to store singular values.
 * @param V Pointer to the output matrix to store right singular vectors.
 * @param work_memory Workspace memory required for computation.
 */
void svd(matrix_2D_t* A, matrix_2D_t* U, float* S, matrix_2D_t* V, work_memory_t work_memory);

/**
 * @brief Solves the linear system Ax=b using an efficient QR decomposition method.
 *
 * This function solves the system without explicitly forming the Q matrix. Instead,
 * Householder reflections are applied directly to a copy of the vector b.
 * This function is more efficient and numerically stable than forming Q.
 * It can handle overdetermined systems (where rows > cols) and finds the
 * least-squares solution.
 *
 * @param A The matrix A (m x n).
 * @param b The vector b (m x 1).
 * @param x The solution vector x (n x 1).
 * @param work_memory A pointer to the work memory structure for temporary allocations.
 */
void solve_linear_qr(matrix_2D_t* A, matrix_2D_t* b, matrix_2D_t* x, work_memory_t work_memory);

#endif /* __MATRIX_H__ */