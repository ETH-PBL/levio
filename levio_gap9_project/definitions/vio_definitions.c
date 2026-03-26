// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "vio_definitions.h"

#include "math.h"

float averageParallax(point2D_u16_t* keypoints0,
                      point2D_u16_t* keypoints1,
                      feature_match_t* matches,
                      matrix_2D_t* K,
                      uint16_t match_counter)
{
    if (match_counter == 0) return 0.0f;
    float parallax_sum = 0.0;
    for (int i = 0; i < match_counter; i++) {
        feature_match_t current_match = matches[i];
        point2D_u16_t pt0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t pt1 = keypoints1[current_match.feat_idx1];
        int32_t x_diff = (int32_t)pt0.x - (int32_t)pt1.x;
        int32_t y_diff = (int32_t)pt0.y - (int32_t)pt1.y;
        int32_t diff_sq = x_diff*x_diff + y_diff*y_diff;
        parallax_sum += sqrtf((float)diff_sq);
    }
    float avg_parallax_normalized = parallax_sum/match_counter/K->data[0];
    return avg_parallax_normalized;
}

void reprojection_error_vector(point3D_float_t* world_point,
                                point2D_u16_t* image_point,
                                matrix_2D_t* T,
                                matrix_2D_t* K,
                                float* e)
{
    float fu = K->data[0];
    float fv = K->data[4];
    float uc = K->data[2];
    float vc = K->data[5];

    float pwh[4];
    float pch[4];
    pwh[0] = world_point->x;
    pwh[1] = world_point->y;
    pwh[2] = world_point->z;
    pwh[3] = 1.0;
    matvec(T,pwh,pch,4);
    
    float ue = uc + fu * pch[0] / pch[2];
    float ve = vc + fv * pch[1] / pch[2];
    float u = image_point->x;
    float v = image_point->y;

    e[0] = u - ue;
    e[1] = v - ve;

    /* Point in image plane or behind camera */
    float z_penalty = (pch[2] < 0) * (-pch[2]) * BEHIND_CAMERA_PENALTY;
    e[0] += copysignf(z_penalty,e[0]);
    e[1] += copysignf(z_penalty,e[1]);
}

float reprojection_error_squared(point3D_float_t* world_point,
                                 point2D_u16_t* image_point,
                                 matrix_2D_t* T,
                                 matrix_2D_t* K)
{
    float e[2] = {0};
    reprojection_error_vector(world_point, image_point, T, K, e);
    return e[0] * e[0] + e[1] * e[1];
}

float reprojection_error(point3D_float_t* world_point,
                         point2D_u16_t* image_point,
                         matrix_2D_t* T,
                         matrix_2D_t* K)
{
    float e[2] = {0};
    reprojection_error_vector(world_point, image_point, T, K, e);
    return sqrtf(e[0] * e[0] + e[1] * e[1]);
}