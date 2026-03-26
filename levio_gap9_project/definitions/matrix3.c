// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "matrix3.h"
#include <math.h>   /* sqrtf */

/* ---------------------------------------------------------------------- */
/* vec3 operations                                                          */
/* ---------------------------------------------------------------------- */

void vec3_add(const vec3_t *a, const vec3_t *b, vec3_t *out)
{
    out->x = a->x + b->x;
    out->y = a->y + b->y;
    out->z = a->z + b->z;
}

void vec3_sub(const vec3_t *a, const vec3_t *b, vec3_t *out)
{
    out->x = a->x - b->x;
    out->y = a->y - b->y;
    out->z = a->z - b->z;
}

void vec3_scale(const vec3_t *v, float s, vec3_t *out)
{
    out->x = v->x * s;
    out->y = v->y * s;
    out->z = v->z * s;
}

float vec3_norm(const vec3_t *v)
{
    return sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
}

float vec3_dot(const vec3_t *a, const vec3_t *b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

void vec3_cross(const vec3_t *a, const vec3_t *b, vec3_t *out)
{
    /* Use temporaries so that out may safely alias a or b. */
    float x = a->y * b->z - a->z * b->y;
    float y = a->z * b->x - a->x * b->z;
    float z = a->x * b->y - a->y * b->x;
    out->x = x;
    out->y = y;
    out->z = z;
}


/* ---------------------------------------------------------------------- */
/* mat3x3 operations                                                        */
/* ---------------------------------------------------------------------- */

void mat3x3_identity(mat3x3_t *A)
{
    memset(A->m, 0, sizeof(A->m));
    A->m[0] = A->m[4] = A->m[8] = 1.0f;
}

void mat3x3_add(const mat3x3_t *A, const mat3x3_t *B, mat3x3_t *C)
{
    /* Safe when C aliases A or B because every element is written once. */
    for (int i = 0; i < 9; ++i)
        C->m[i] = A->m[i] + B->m[i];
}

void mat3x3_mul(const mat3x3_t *A, const mat3x3_t *B, mat3x3_t *C)
{
    /* C must not alias A or B — we accumulate into a local copy first. */
    mat3x3_t tmp;
    memset(tmp.m, 0, sizeof(tmp.m));
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                tmp.m[i * 3 + j] += A->m[i * 3 + k] * B->m[k * 3 + j];
    *C = tmp;
}

void mat3x3_scale(const mat3x3_t *A, float s, mat3x3_t *B)
{
    for (int i = 0; i < 9; ++i)
        B->m[i] = A->m[i] * s;
}

void mat3x3_transpose(const mat3x3_t *A, mat3x3_t *B)
{
    /* B must not alias A — write into a local copy first. */
    mat3x3_t tmp;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            tmp.m[r * 3 + c] = A->m[c * 3 + r];
    *B = tmp;
}

void mat3x3_vec3_mul(const mat3x3_t *A, const vec3_t *v, vec3_t *out)
{
    /* Use temporaries so that out may safely alias v (unusual but safe). */
    float x = A->m[0] * v->x + A->m[1] * v->y + A->m[2] * v->z;
    float y = A->m[3] * v->x + A->m[4] * v->y + A->m[5] * v->z;
    float z = A->m[6] * v->x + A->m[7] * v->y + A->m[8] * v->z;
    out->x = x;
    out->y = y;
    out->z = z;
}

void mat3x3_skew_symmetric(const vec3_t *v, mat3x3_t *out)
{
    /*
     *         [  0   -v.z  v.y ]
     * [v]_×  =  [ v.z   0   -v.x ]
     *         [-v.y  v.x   0   ]
     *
     * Used to convert a cross-product  v × w  into a matrix-vector
     * product  [v]_× · w.
     */
    out->m[0] =  0.0f;   out->m[1] = -v->z;  out->m[2] =  v->y;
    out->m[3] =  v->z;   out->m[4] =  0.0f;  out->m[5] = -v->x;
    out->m[6] = -v->y;   out->m[7] =  v->x;  out->m[8] =  0.0f;
}

void mat3x3_rodrigues_exp(const vec3_t *phi, mat3x3_t *R)
{
    /*
     * Rodrigues' formula:
     *   R = I + (sin θ / θ)·S + ((1 - cos θ) / θ²)·S²
     * where  θ = |phi|  and  S = [phi]_×.
     *
     * For very small angles (θ < 1e-6) the rotation is negligible and
     * the function returns the identity to avoid division by zero.
     */
    float theta = vec3_norm(phi);

    mat3x3_t I;
    mat3x3_identity(&I);

    if (theta < 1e-6f) {
        *R = I;
        return;
    }

    mat3x3_t S;
    mat3x3_skew_symmetric(phi, &S);

    mat3x3_t S2;
    mat3x3_mul(&S, &S, &S2);

    float a = sinf(theta) / theta;
    float b = (1.0f - cosf(theta)) / (theta * theta);

    /*
     * R = I + a*S + b*S²
     * Written as two successive in-place additions to avoid an extra
     * temporary matrix.
     */
    mat3x3_t aS, bS2;
    mat3x3_scale(&S,  a, &aS);
    mat3x3_scale(&S2, b, &bS2);

    mat3x3_add(&I,  &aS,  R);   /* R  = I + a*S       */
    mat3x3_add(R,   &bS2, R);   /* R += b*S²           */
}


/* ---------------------------------------------------------------------- */
/* Bridge helpers                                                           */
/* ---------------------------------------------------------------------- */

void mat3x3_from_matrix2d(const matrix_2D_t *src, mat3x3_t *dst)
{
    /*
     * Deep copy: after this call the two objects are independent.
     * The caller is responsible for ensuring src->meta.rows == 3 and
     * src->meta.cols == 3.
     */
    memcpy(dst->m, src->data, 9 * sizeof(float));
}

matrix_2D_t matrix2d_from_mat3x3(mat3x3_t *src)
{
    /*
     * Zero-copy view: the returned matrix_2D_t shares the storage of *src.
     * The caller must ensure that src outlives any use of the returned view.
     */
    matrix_2D_t view;
    view.data       = src->m;
    view.meta.rows  = 3;
    view.meta.cols  = 3;
    view.meta.transpose = 0;
    view.meta.misc      = 0;
    return view;
}