// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __MAT3_H__
#define __MAT3_H__

#include "pmsis.h"
#include "matrix.h"
#include "vio_definitions.h"
 
/* ---------------------------------------------------------------------- */
/* Types                                                                    */
/* ---------------------------------------------------------------------- */
 
/** Concrete 3-vector type.  Identical in memory to point3D_float_t. */
typedef point3D_float_t vec3_t;
 
/** Flat, row-major 3x3 matrix.  m[row*3 + col]. */
typedef struct {
    float m[9];
} mat3x3_t;
 
 
/* ---------------------------------------------------------------------- */
/* Cast helpers for raw float arrays (zero-cost, no data movement)         */
/* ---------------------------------------------------------------------- */
 
/**
 * @brief View a raw float[9] array as a mat3x3_t pointer.
 *
 * Requires the array to be row-major and contiguous, which is the layout
 * used by all float[9] rotation/Jacobian matrices in imu_optimization.c.
 * No data is copied; the caller retains ownership of the original array.
 */
static inline mat3x3_t *mat3x3_from_raw(float raw[9])
{
    return (mat3x3_t *)(void *)raw;
}
 
/**
 * @brief View a raw float[3] array as a vec3_t pointer.
 *
 * Requires the array to be contiguous, which matches the float[3] fields
 * in CameraState9Dof, CameraState15Dof, and temporary stack arrays.
 * No data is copied; the caller retains ownership of the original array.
 */
static inline vec3_t *vec3_from_raw(float raw[3])
{
    return (vec3_t *)(void *)raw;
}
 
 
/* ---------------------------------------------------------------------- */
/* vec3 operations                                                          */
/* ---------------------------------------------------------------------- */
 
/**
 * @brief out = a + b
 */
void vec3_add(const vec3_t *a, const vec3_t *b, vec3_t *out);
 
/**
 * @brief out = a - b
 */
void vec3_sub(const vec3_t *a, const vec3_t *b, vec3_t *out);
 
/**
 * @brief out = v * s
 */
void vec3_scale(const vec3_t *v, float s, vec3_t *out);
 
/**
 * @brief Returns the Euclidean norm (length) of v.
 */
float vec3_norm(const vec3_t *v);
 
/**
 * @brief Returns the dot product of a and b.
 */
float vec3_dot(const vec3_t *a, const vec3_t *b);
 
/**
 * @brief out = a × b  (cross product).
 */
void vec3_cross(const vec3_t *a, const vec3_t *b, vec3_t *out);
 
 
/* ---------------------------------------------------------------------- */
/* mat3x3 operations                                                        */
/* ---------------------------------------------------------------------- */
 
/**
 * @brief Set A to the 3x3 identity matrix.
 */
void mat3x3_identity(mat3x3_t *A);
 
/**
 * @brief C = A + B  (element-wise).
 * C may alias A or B.
 */
void mat3x3_add(const mat3x3_t *A, const mat3x3_t *B, mat3x3_t *C);
 
/**
 * @brief C = A * B  (matrix product).
 * C must NOT alias A or B.
 */
void mat3x3_mul(const mat3x3_t *A, const mat3x3_t *B, mat3x3_t *C);
 
/**
 * @brief B = A * s  (scalar multiplication).
 * B may alias A.
 */
void mat3x3_scale(const mat3x3_t *A, float s, mat3x3_t *B);
 
/**
 * @brief B = A^T  (transpose).
 * B must NOT alias A.
 */
void mat3x3_transpose(const mat3x3_t *A, mat3x3_t *B);
 
/**
 * @brief out = A * v  (matrix–vector product).
 */
void mat3x3_vec3_mul(const mat3x3_t *A, const vec3_t *v, vec3_t *out);
 
/**
 * @brief Build the skew-symmetric (cross-product) matrix of v.
 *
 *         [  0   -v.z  v.y ]
 * out  =  [ v.z   0   -v.x ]
 *         [-v.y  v.x   0   ]
 */
void mat3x3_skew_symmetric(const vec3_t *v, mat3x3_t *out);
 
/**
 * @brief Rodrigues exponential map: rotation vector → rotation matrix.
 *
 * R = Exp(phi) = I + (sin θ / θ)·[phi]_x + ((1 - cos θ) / θ²)·[phi]_x²
 * where θ = |phi|.
 *
 * For |phi| < 1e-6 the function returns the identity matrix.
 *
 * @param phi  3-vector rotation axis scaled by angle (rad).
 * @param R    Output rotation matrix.
 */
void mat3x3_rodrigues_exp(const vec3_t *phi, mat3x3_t *R);
 
 
/* ---------------------------------------------------------------------- */
/* Bridge helpers between mat3x3_t and the generic matrix_2D_t             */
/* ---------------------------------------------------------------------- */
 
/**
 * @brief Copy the 9 floats from a 3×3 matrix_2D_t into a mat3x3_t.
 *
 * The source matrix_2D_t must have rows==3 and cols==3 and its data
 * pointer must be non-NULL.  The two objects remain independent after
 * this call (deep copy).
 */
void mat3x3_from_matrix2d(const matrix_2D_t *src, mat3x3_t *dst);
 
/**
 * @brief Create a matrix_2D_t that is a zero-copy view over a mat3x3_t.
 *
 * The returned matrix_2D_t shares storage with *src.  The caller must
 * ensure that src outlives any use of the returned view.
 */
matrix_2D_t matrix2d_from_mat3x3(mat3x3_t *src);

#endif /* __MAT3_H__ */