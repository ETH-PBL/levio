// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "triangulation.h"
#include "math.h"

#define Min(a, b)       (((a)<(b))?(a):(b))

/**
 * @brief Triangulates a single 3D point using the Direct Linear Transform (DLT) method.
 *
 * Forms a 4x4 linear system A from the two projection equations per camera:
 *   x * (T[2,:] * X) = T[0,:] * X
 *   y * (T[2,:] * X) = T[1,:] * X
 *
 * The null-space solution (homogeneous 3D point) is found via the inverse
 * power method on A^T*A and then dehomogenized.
 */
point3D_float_t triangulatePoint(point2D_u16_t* pt0,
                                 point2D_u16_t* pt1,
                                 matrix_2D_t* K,
                                 matrix_2D_t* T0,
                                 matrix_2D_t* T1,
                                 work_memory_t work_memory)
{
    float x0[3] = { (pt0->x-K->data[2])/K->data[0], (pt0->y-K->data[5])/K->data[4], 1.0};
    float x1[3] = { (pt1->x-K->data[2])/K->data[0], (pt1->y-K->data[5])/K->data[4], 1.0};
    float* A_data = allocate_work_memory(&work_memory,16*sizeof(float));
    for (uint8_t j = 0; j < 4; ++j){
        A_data[0*4+j] = x0[0]*T0->data[2*4+j]-T0->data[0*4+j];
        A_data[1*4+j] = x0[1]*T0->data[2*4+j]-T0->data[1*4+j];
        A_data[2*4+j] = x1[0]*T1->data[2*4+j]-T1->data[0*4+j];
        A_data[3*4+j] = x1[1]*T1->data[2*4+j]-T1->data[1*4+j];
    }
    matrix_2D_t A = {A_data,{4,4,0,0}};
    matrix_2D_t At = {A_data,{4,4,1,0}};
    float* ATA_data = allocate_work_memory(&work_memory,16*sizeof(float));
    matrix_2D_t ATA = {ATA_data,{4,4,0,0}};
    matmul(&At,&A,&ATA);
    float X[4] = {0};
    inverse_power_method(ATA_data,X,4);
    
    // Check for degenerate case: if X[3] is near zero, triangulation failed
    if (fabsf(X[3]) < 1e-6f) {
        return (point3D_float_t){0.0f, 0.0f, 0.0f};  // Return invalid point marker
    }
    return (point3D_float_t){X[0]/X[3], X[1]/X[3], X[2]/X[3]};
}

void triangulatePoints(point2D_u16_t* keypoints0,
                       point2D_u16_t* keypoints1,
                       feature_match_t* matches,
                       matrix_2D_t* K,
                       matrix_2D_t* T0,
                       matrix_2D_t* T1,
                       point3D_float_t* triangulatedPoints,
                       work_memory_t work_memory,
                       uint16_t match_counter)
{
    // DLT Method
    for (int i = 0; i < match_counter; i++) {
        feature_match_t current_match = matches[i];
        point2D_u16_t pt0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t pt1 = keypoints1[current_match.feat_idx1];
        triangulatedPoints[i] = triangulatePoint(&pt0, &pt1, K, T0, T1, work_memory);
    }
}

typedef struct triangulation_args
{
    point2D_u16_t* keypoints0;
    point2D_u16_t* keypoints1;
    feature_match_t* matches;
    matrix_2D_t* K;
    matrix_2D_t* T0;
    matrix_2D_t* T1;
    point3D_float_t* triangulatedPoints;
    work_memory_t work_memory;
    uint16_t match_counter;
    uint8_t nb_cores;
}triangulation_args_t;

/**
 * @brief Cluster kernel: each core triangulates its assigned subset of matches.
 *
 * Work is divided into equal intervals across nb_cores. Each core triangulates
 * the range [start, stop) using a per-core slice of work_memory.
 */
void triangulatePointsSubset(void* args)
{
    triangulation_args_t* tri_args = (triangulation_args_t*) args;
    uint16_t core_id = pi_core_id();
	uint16_t nb_cores = tri_args->nb_cores;
    work_memory_t work_memory = split_work_memory(&tri_args->work_memory, nb_cores, core_id);

    uint16_t intervals = (tri_args->match_counter + nb_cores - 1)/nb_cores;
    uint16_t start = core_id * intervals;
    uint16_t stop = Min((core_id + 1) * intervals, tri_args->match_counter); 

    // DLT Method
    for (int i = start; i < stop; i++) {
        feature_match_t current_match = tri_args->matches[i];
        point2D_u16_t pt0 = tri_args->keypoints0[current_match.feat_idx0];
        point2D_u16_t pt1 = tri_args->keypoints1[current_match.feat_idx1];
        tri_args->triangulatedPoints[i] = triangulatePoint(&pt0, &pt1, tri_args->K, tri_args->T0, tri_args->T1, work_memory);
    }
}

void triangulatePointsMulticore(point2D_u16_t* keypoints0,
                                point2D_u16_t* keypoints1,
                                feature_match_t* matches,
                                matrix_2D_t* K,
                                matrix_2D_t* T0,
                                matrix_2D_t* T1,
                                point3D_float_t* triangulatedPoints,
                                work_memory_t work_memory,
                                uint16_t match_counter,
                                uint8_t nb_cores)
{
    triangulation_args_t args = {keypoints0,
                                 keypoints1,
                                 matches,
                                 K,
                                 T0,
                                 T1,
                                 triangulatedPoints,
                                 work_memory,
                                 match_counter,
                                 nb_cores};
    pi_cl_team_fork(nb_cores, triangulatePointsSubset, &args);
}