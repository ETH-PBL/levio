// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __IMU_BIAS_OPTIMIZATION__H__
#define __IMU_BIAS_OPTIMIZATION__H__

#include "imu_optimization.h"

#define STATE_SIZE_BIAS 15
#define BIAS_SIZE 6

// 15-DoF state including IMU biases
typedef struct {
    float r[3], p[3], v[3], b_g[3], b_a[3];
} CameraState15Dof;

/**
 * @brief Solves the dense linear system Ax=b using Cholesky decomposition.
 *
 * Extracts A (n×n) and b (n×1) from the augmented matrix Ab (n×(n+1)),
 * performs Cholesky decomposition, and solves for x via forward/backward substitution.
 *
 * @param Ab          Augmented matrix [A|b] of size n×(n+1), row-major.
 * @param x           Output solution vector of size n.
 * @param n           Dimension of the linear system.
 * @param work_memory Workspace for temporary allocations during decomposition.
 * @return 1 on success, 0 if A is not positive-definite or memory allocation fails.
 */
int solve_cholesky_system(float* Ab, float* x, int n, work_memory_t work_memory);

/**
 * @brief Optimizes IMU biases (gyroscope and accelerometer) sequentially across a window of keyframes.
 *
 * For each interval between consecutive optimized poses, builds a 15-DoF IMU residual and
 * Jacobian w.r.t. biases and solves the resulting normal equations to update the bias estimates.
 *
 * @param biases                 Output array of bias vectors [b_g (3), b_a (3)] per keyframe.
 * @param optimized_poses        Array of optimized camera poses (world-to-camera transforms).
 * @param optimized_velocities   Array of optimized world-frame velocities per keyframe.
 * @param imu_factors_bias       Array of pre-integration factors with bias Jacobians.
 * @param num_poses              Number of poses in the current optimization window.
 * @param total_poses            Total number of poses allocated (capacity).
 * @param g                      World-frame gravity vector [m/s^2] (3-element array).
 * @param work_memory            Workspace for temporary allocations.
 * @return 1 on success, 0 if any Cholesky solve fails.
 */
int optimize_biases_sequentially(float biases[MAX_KEYFRAMES][BIAS_SIZE],
                                 const pose_t* optimized_poses,
                                 const point3D_float_t* optimized_velocities,
                                 const imu_factor_with_bias_t* imu_factors_bias,
                                 int num_poses,
                                 int total_poses,
                                 const float* g,
                                 work_memory_t work_memory);

#endif /* __IMU_BIAS_OPTIMIZATION__H__ */