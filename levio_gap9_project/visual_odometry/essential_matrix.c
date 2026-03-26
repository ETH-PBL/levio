// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "essential_matrix.h"
#include "math.h"
#include "definitions/type_definitions.h"
#include "triangulation.h"

#define Abs(a)          (((int)(a)<0)?(-(a)):(a))

#define SAMPLE_SIZE 8

typedef struct ransac_iteration_essential_args
{
    point2D_float_t* norm_keypoints0;
    point2D_float_t* norm_keypoints1;
    uint16_t* sample_idxs;
    uint8_t* inlierMasks;
    int* inlierCounts;
    float* avgInlierErrors;
    float* E_data_all_cores;
    work_memory_t* work_memory;
    float inlierThreshold;
    uint16_t match_counter;
}ransac_iteration_essential_args_t;


/**
 * @brief Estimates the essential matrix from normalized point correspondences
 *        using the 8-point algorithm.
 *
 * Builds the 9x9 outer-product sum H^T*H from the Kronecker products of
 * matched normalized image coordinates, then extracts the minimum-eigenvalue
 * vector as the fundamental matrix F. F is then projected onto the essential
 * matrix manifold by enforcing rank-2 via SVD: E = U * diag(σ,σ,0) * V^T
 * where σ = (S[0]+S[1])/2.
 */
void computeEssentialMatrix(point2D_float_t* norm_keypoints0,
                            point2D_float_t* norm_keypoints1,
                            uint16_t* sample_idxs,
                            matrix_2D_t* E,
                            matrix_2D_t* U,
                            matrix_2D_t* V,
                            work_memory_t work_memory,
                            uint16_t sample_counter)
{
    /* Compose H^T*H matrix */
    float H_row[9];
    float* HTH_data = allocate_work_memory(&work_memory,9*9*sizeof(float));
    memset(HTH_data,0,9*9*sizeof(float));
    for(uint16_t index = 0; index < sample_counter; index++)
    {
        uint16_t sample_idx = sample_idxs[index];
        float x0 = norm_keypoints0[sample_idx].x;
        float y0 = norm_keypoints0[sample_idx].y;
        float x1 = norm_keypoints1[sample_idx].x;
        float y1 = norm_keypoints1[sample_idx].y;
        H_row[0] = x1 * x0;
        H_row[1] = x1 * y0;
        H_row[2] = x1;
        H_row[3] = y1 * x0;
        H_row[4] = y1 * y0;
        H_row[5] = y1;
        H_row[6] = x0;
        H_row[7] = y0;
        H_row[8] = 1.0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                HTH_data[i * 9 + j] += H_row[i]*H_row[j];
            }
        }
    }

    // Extract Fundamental Matrix
    float F_data[9] = {0};
    inverse_power_method(HTH_data,F_data,9);

    // Ensure rank 2
    float S[3] = {0};
    matrix_2D_t F = {F_data,{3,3,0,0}};
    svd(&F,U,S,V,work_memory);
    
    float sigma = (S[0] + S[1])*0.5f;
    // E = USV^T
    float S_data[9] = {0};
    S_data[0] = sigma;
    S_data[4] = sigma;
    float temp_data[9] = {0};
    matrix_2D_t temp = {temp_data, {3,3,0,0}};
    matrix_2D_t S_dash = {S_data, {3,3,0,0}};
    matrix_2D_t Vt = {V->data, {3,3,1,0}};
    matmul(&S_dash,&Vt,&temp);
    matmul(U,&temp,E);
    return;
}

/**
 * @brief Computes the Sampson approximation of the epipolar error for a point pair.
 *
 * The Sampson error is a first-order geometric approximation of the reprojection
 * error:  (x2^T * E * x1)^2 / (||E*x1||_xy^2 + ||E^T*x2||_xy^2)
 *
 * It is cheaper to compute than the true geometric error and serves as the
 * RANSAC inlier criterion for essential matrix estimation.
 */
float computeSampsonError(point2D_float_t pt1, point2D_float_t pt2, matrix_2D_t* E) {
    float Ex1[3];
    float Etx2[3];
    for (int i = 0; i < 3; i++) {
        Ex1[i] =  E->data[i*3+0]*pt1.x + E->data[i*3+1]*pt1.y + E->data[i*3+2]*1.0;
        Etx2[i] = E->data[0*3+i]*pt2.x + E->data[1*3+i]*pt2.y + E->data[2*3+i]*1.0;
    }

    float x2tEx1 = pt2.x*Ex1[0] + pt2.y*Ex1[1] + 1.0*Ex1[2];
    float denom = Ex1[0]*Ex1[0] + Ex1[1]*Ex1[1] + Etx2[0]*Etx2[0] + Etx2[1]*Etx2[1];

    if (denom < 1e-12f) return 1e6f;
    return (x2tEx1 * x2tEx1) / denom;
}

void ransacEssentialMatrix(point2D_u16_t* keypoints0,
                           point2D_u16_t* keypoints1,
                           feature_match_t* matches,
                           matrix_2D_t* K,
                           matrix_2D_t* bestE,
                           work_memory_t work_memory,
                           uint16_t match_counter,
                           uint16_t ransac_iterations)
{
    LOG_INFO("Start 8-Point RANSAC\n");
    if (match_counter < 8)
    {
        matfill(bestE, 0.0);
        return;
    }
    uint16_t bestInlierCount = 0;
    uint8_t* bestInlierMask = allocate_work_memory(&work_memory, match_counter*sizeof(uint8_t));
    uint8_t* inlierMask = allocate_work_memory(&work_memory, match_counter*sizeof(uint8_t));
    float inlierThreshold = (SAMPSON_PIXEL_THRESHOLD/K->data[0])*(SAMPSON_PIXEL_THRESHOLD/K->data[0]);
    float bestAvgInlierError = inlierThreshold;
    uint16_t* sample_idxs = allocate_work_memory(&work_memory, SAMPLE_SIZE*sizeof(uint16_t));

    uint16_t* all_indices = allocate_work_memory(&work_memory, match_counter * sizeof(uint16_t));
    for (uint16_t i = 0; i < match_counter; i++) {
        all_indices[i] = i;
    }

    point2D_float_t* norm_keypoints0 = allocate_work_memory(&work_memory, match_counter*sizeof(point2D_float_t));
    point2D_float_t* norm_keypoints1 = allocate_work_memory(&work_memory, match_counter*sizeof(point2D_float_t));
    for (int i = 0; i < match_counter; i++) {
        feature_match_t current_match = matches[i];
        point2D_u16_t pt0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t pt1 = keypoints1[current_match.feat_idx1];
        norm_keypoints0[i] = (point2D_float_t){(pt0.x-K->data[2])/K->data[0], (pt0.y-K->data[5])/K->data[4]};
        norm_keypoints1[i] = (point2D_float_t){(pt1.x-K->data[2])/K->data[0], (pt1.y-K->data[5])/K->data[4]};
    }

    for (int iter = 0; iter < ransac_iterations; iter++) {
        // Randomly select 8 point
        // Fisher-Yates shuffle for first sample_size elements
        for (uint8_t i = 0; i < SAMPLE_SIZE; i++) {
            // Generate unbiased random index in range [i, match_counter-1]
            uint16_t j = i + rand_range(match_counter - i);
            // Swap elements
            uint16_t temp = all_indices[i];
            all_indices[i] = all_indices[j];
            all_indices[j] = temp;
            
            sample_idxs[i] = all_indices[i];
        }

        float U_data[9] = {0};
        float V_data[9] = {0};
        matrix_2D_t V = {V_data, {3,3,0,0}};
        matrix_2D_t U = {U_data, {3,3,0,0}};
        float E_data[9];
        matrix_2D_t E = {E_data,{3,3,0,0}};
        computeEssentialMatrix(norm_keypoints0, norm_keypoints1, sample_idxs, &E, &U, &V, work_memory, SAMPLE_SIZE);

        // Count inliers
        int inlierCount = 0;
        float avgInlierError = 0.0;
        for (int i = 0; i < match_counter; i++) {
            float error = computeSampsonError(norm_keypoints0[i], norm_keypoints1[i], &E);
            if (error < inlierThreshold) {
                inlierMask[i] = 1;
                inlierCount++;
                avgInlierError += error;
            }
            else {
                inlierMask[i] = 0;
            }
        }
        if (inlierCount > 0){
            avgInlierError = avgInlierError/inlierCount;
        } else {
            avgInlierError = inlierThreshold;
        }

        if ((inlierCount > bestInlierCount) || (inlierCount == bestInlierCount && avgInlierError < bestAvgInlierError)) {
            bestInlierCount = inlierCount;
            bestAvgInlierError = avgInlierError;
            for (int i = 0; i < 9; i++){
                    bestE->data[i] = E.data[i];
            }
            for (int i = 0; i < match_counter; i++) {
                bestInlierMask[i] = inlierMask[i];
            }
        }
    }
    LOG_INFO("Best inlier count: %d\n", bestInlierCount);
    return;
}

/**
 * @brief Cluster kernel: each core estimates one essential matrix from its pre-drawn sample.
 *
 * Each core reads its 8 sample indices from sample_idxs[core_id*SAMPLE_SIZE] and writes
 * its result to E_data_all_cores[core_id*9]. Work memory is split per core.
 */
void ransacIterationEssentialMatrix(void* args)
{
    ransac_iteration_essential_args_t* iter_args = (ransac_iteration_essential_args_t*) args;
    uint16_t core_id = pi_core_id();
    work_memory_t work_memory = split_work_memory(iter_args->work_memory, NB_CORES, core_id);

    float U_data[9] = {0};
    float V_data[9] = {0};
    matrix_2D_t V = {V_data, {3,3,0,0}};
    matrix_2D_t U = {U_data, {3,3,0,0}};
    matrix_2D_t E = {&iter_args->E_data_all_cores[core_id*9],{3,3,0,0}};
    computeEssentialMatrix(iter_args->norm_keypoints0,
                           iter_args->norm_keypoints1,
                           &iter_args->sample_idxs[SAMPLE_SIZE*core_id],
                           &E, &U, &V,
                           work_memory,
                           SAMPLE_SIZE);
}

/**
 * @brief Cluster kernel: each core evaluates the Sampson inlier count for its candidate E.
 *
 * Each core reads its candidate E from E_data_all_cores[core_id*9] and writes
 * per-match inlier flags to inlierMasks[core_id*match_counter + i], along with
 * the inlier count and average inlier error.
 */
void ransacIterationSampsonError(void* args)
{
    ransac_iteration_essential_args_t* iter_args = (ransac_iteration_essential_args_t*) args;
    uint16_t core_id = pi_core_id();
    work_memory_t work_memory = split_work_memory(iter_args->work_memory, NB_CORES, core_id);

    matrix_2D_t E = {&iter_args->E_data_all_cores[core_id*9],{3,3,0,0}};
    iter_args->avgInlierErrors[core_id] = 0;
    iter_args->inlierCounts[core_id] = 0;
    for (int i = 0; i < iter_args->match_counter; ++i) {
        float error = computeSampsonError(iter_args->norm_keypoints0[i],
                                          iter_args->norm_keypoints1[i],
                                          &E);
        if (error < iter_args->inlierThreshold) {
            iter_args->inlierMasks[i+iter_args->match_counter*core_id] = 1;
            iter_args->inlierCounts[core_id] += 1;
            iter_args->avgInlierErrors[core_id] += error;
        }
        else {
            iter_args->inlierMasks[i+iter_args->match_counter*core_id] = 0;
        }
    }
    if (iter_args->inlierCounts[core_id] > 0){
        iter_args->avgInlierErrors[core_id] = iter_args->avgInlierErrors[core_id]/iter_args->inlierCounts[core_id];
    } else {
        iter_args->avgInlierErrors[core_id] = iter_args->inlierThreshold;
    }
}

void ransacEssentialMatrixMulticore(point2D_u16_t* keypoints0,
                                    point2D_u16_t* keypoints1,
                                    feature_match_t* matches,
                                    matrix_2D_t* K,
                                    matrix_2D_t* bestE,
                                    work_memory_t work_memory,
                                    uint16_t match_counter,
                                    uint16_t ransac_iterations,
                                    uint8_t nb_cores)
{
    LOG_INFO("Start 8-Point RANSAC\n");
    if (match_counter < 8)
    {
        matfill(bestE, 0.0);
        return;
    }
    uint16_t bestInlierCount = 0;
    uint8_t* bestInlierMask = allocate_work_memory(&work_memory, match_counter*sizeof(uint8_t));
    float inlierThreshold = (SAMPSON_PIXEL_THRESHOLD/K->data[0])*(SAMPSON_PIXEL_THRESHOLD/K->data[0]);
    float bestAvgInlierError = inlierThreshold;

    point2D_float_t* norm_keypoints0 = allocate_work_memory(&work_memory, match_counter*sizeof(point2D_float_t));
    point2D_float_t* norm_keypoints1 = allocate_work_memory(&work_memory, match_counter*sizeof(point2D_float_t));
    for (int i = 0; i < match_counter; i++) {
        feature_match_t current_match = matches[i];
        point2D_u16_t pt0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t pt1 = keypoints1[current_match.feat_idx1];
        norm_keypoints0[i] = (point2D_float_t){(pt0.x-K->data[2])/K->data[0], (pt0.y-K->data[5])/K->data[4]};
        norm_keypoints1[i] = (point2D_float_t){(pt1.x-K->data[2])/K->data[0], (pt1.y-K->data[5])/K->data[4]};
    }

    uint8_t* inlierMasks = allocate_work_memory(&work_memory, NB_CORES*match_counter*sizeof(uint8_t));
    uint16_t* sample_idxs = allocate_work_memory(&work_memory, NB_CORES*SAMPLE_SIZE*sizeof(uint16_t));
    int* inlierCounts = allocate_work_memory(&work_memory, NB_CORES*sizeof(int));
    float* avgInlierErrors = allocate_work_memory(&work_memory, NB_CORES*sizeof(float));
    float* E_data_all_cores = allocate_work_memory(&work_memory, NB_CORES*9*sizeof(float));

    uint16_t* all_indices = allocate_work_memory(&work_memory, match_counter * sizeof(uint16_t));
    for (uint16_t i = 0; i < match_counter; i++) {
        all_indices[i] = i;
    }

    ransac_iteration_essential_args_t args = {norm_keypoints0,
                                              norm_keypoints1,
                                              sample_idxs,
                                              inlierMasks,
                                              inlierCounts,
                                              avgInlierErrors,
                                              E_data_all_cores,
                                              &work_memory,
                                              inlierThreshold,
                                              match_counter};

    for (int iter = 0; iter < ransac_iterations/nb_cores; iter++) {
        // Randomly select 8 point
        for (uint8_t core_id = 0; core_id < nb_cores; core_id++)
        {
        // Fisher-Yates shuffle for first sample_size elements
            for (uint8_t i = 0; i < SAMPLE_SIZE; i++) {
                // Generate unbiased random index in range [i, match_counter-1]
                uint16_t j = i + rand_range(match_counter - i);
                // Swap elements
                uint16_t temp = all_indices[i];
                all_indices[i] = all_indices[j];
                all_indices[j] = temp;
                
                sample_idxs[core_id*SAMPLE_SIZE+i] = all_indices[i];
            }
        }

        pi_cl_team_fork(nb_cores, ransacIterationEssentialMatrix, &args);
        pi_cl_team_fork(nb_cores, ransacIterationSampsonError, &args);

        /* For each core */
        for (int i = 0; i < nb_cores; ++i) {
            if ((inlierCounts[i] > bestInlierCount) || (inlierCounts[i] == bestInlierCount && avgInlierErrors[i] < bestAvgInlierError)) {
                bestInlierCount = inlierCounts[i];
                bestAvgInlierError = avgInlierErrors[i];
                for (int j = 0; j < 9; j++){
                        bestE->data[j] = E_data_all_cores[i*9+j];
                }
                for (int j = 0; j < match_counter; j++) {
                    bestInlierMask[j] = inlierMasks[i*match_counter+j];
                }
            }
        }
    }
    LOG_INFO("Best inlier count: %d\n", bestInlierCount);
    return;
}

/**
 * @brief Counts triangulated points that pass the cheirality check for both cameras.
 *
 * A point is valid if it has positive depth (z > 0) in the reference camera frame
 * AND positive depth after projecting through T1 (i.e., it lies in front of both cameras).
 * Used to select the correct rotation/translation hypothesis in recoverPose().
 */
uint16_t check_valid_points(matrix_2D_t* T0,
                            matrix_2D_t* T1,
                            point3D_float_t* triangulatedPoints,
                            uint16_t match_counter)
{
    uint16_t positive_z = 0;
    for (int i = 0; i < match_counter; i++) {
        uint8_t valid = (triangulatedPoints[i].z > 0);
        float proj[4] = {0};
        matvec(T1,(float[4]){triangulatedPoints[i].x,triangulatedPoints[i].y,triangulatedPoints[i].z,1.0}, proj, 4);
        valid = valid && (proj[2] > 0);
        if (valid)
        {
            positive_z += 1;
        }
    }
    return positive_z;
}

void recoverPose(point2D_u16_t* keypoints0,
                 point2D_u16_t* keypoints1,
                 feature_match_t* matches,
                 matrix_2D_t* K,
                 matrix_2D_t* bestE,
                 matrix_2D_t* T,
                 work_memory_t work_memory,
                 uint16_t match_counter,
                 uint8_t singlecore_execution,
                 uint8_t nb_cores)
{
    LOG_INFO("Recovering Pose\n");
    float U_data[9] = {0};
    float V_data[9] = {0};
    float S[3] = {0};
    matrix_2D_t U = {U_data,{3,3,0,0}};  
    matrix_2D_t V = {V_data,{3,3,0,0}};
    svd(bestE,&U,S,&V, work_memory);

    float detu = determinant(&U);
    if(detu < 0)
    {
        matscale(&U, -1.0);
    }
    float detv = determinant(&V);
    if(detv < 0)
    {
        matscale(&V, -1.0);
    }
    float t[3] = {U.data[2], U.data[5], U.data[8]};
    float t_neg[3] = {-U.data[2], -U.data[5], -U.data[8]};

    float W_data[9] = {0};
    W_data[1] = -1.0;
    W_data[3] = 1.0;
    W_data[8] = 1.0;
    float R1_data[9] = {0};
    float temp_data [9] = {0};
    matrix_2D_t temp = {temp_data,{3,3,0,0}};
    matrix_2D_t R1 = {R1_data,{3,3,0,0}};
    matrix_2D_t Vt = {V.data,{3,3,1,0}};
    matrix_2D_t W = {W_data,{3,3,0,0}};
    matrix_2D_t Wt = {W.data,{3,3,1,0}};
    matmul(&Wt,&Vt,&temp);
    matmul(&U,&temp,&R1);

    float R2_data[9] = {0};
    matrix_2D_t R2 = {R2_data,{3,3,0,0}};
    matmul(&W,&Vt,&temp);
    matmul(&U,&temp,&R2);

    float* T0_data = allocate_work_memory(&work_memory,sizeof(float)*16);
    float* T1_data = allocate_work_memory(&work_memory,sizeof(float)*16);
    matrix_2D_t T0 = {T0_data,{4,4,0,0}};
    matrix_2D_t T1 = {T1_data,{4,4,0,0}};
    mateye(&T0);
    // Initializing T to identity in case no valid solution is found.
    mateye(T);

    point3D_float_t* wold_points = allocate_work_memory(&work_memory,sizeof(point3D_float_t)*match_counter);
    matrix_2D_t* rotations[2] = {&R1, &R2};
    float* translations[2] = {t, t_neg};

    uint16_t best_valid_points = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            composeTransformation(rotations[i], translations[j], &T1);
            if(singlecore_execution)
            {
                triangulatePoints(keypoints0, keypoints1, matches, K, &T0, &T1, wold_points, work_memory, match_counter);
            }
            else
            {
                triangulatePointsMulticore(keypoints0, keypoints1, matches, K, &T0, &T1, wold_points, work_memory, match_counter, nb_cores);
            }
            uint16_t valid_points = check_valid_points(&T0, &T1, wold_points, match_counter);
            /* If no configuration has > 0 valid points, return identitiy matrix */
            if (valid_points > best_valid_points)
            {
                best_valid_points = valid_points;
                composeTransformation(rotations[i],translations[j],T);
            }
        }
    }
}