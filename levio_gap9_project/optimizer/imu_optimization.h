// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __IMU_OPTIMIZATION__H__
#define __IMU_OPTIMIZATION__H__

#include "pmsis.h"
#include "definitions/config.h"
#include "definitions/type_definitions.h"

#define STATE_SIZE 9
#define ERROR_SIZE 9

typedef struct {
    float r[3]; // Rotation (Rodrigues vector)
    float p[3]; // Position (world frame)
    float v[3]; // Velocity (world frame)
} CameraState9Dof;

/**
 * @brief Multiplies a row-major 3x3 matrix by a 3-vector.
 *
 * @param M      Row-major 3x3 matrix (9 elements).
 * @param v      Input 3-vector.
 * @param result Output 3-vector: result = M * v.
 */
void mat33_vec3_mult(const float M[9], const float v[3], float result[3]);

/**
 * @brief Computes the inverse of the right Jacobian of SO(3).
 *
 * Jr(phi)^-1 = I + 1/2*[phi]_x + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta)))*[phi]_x^2
 *
 * @param phi    3-element Rodrigues (rotation) vector.
 * @param Jr_inv Output 9-element row-major 3x3 inverse Jacobian matrix.
 */
void compute_right_jacobian_so3_inverse(const float phi[3], float Jr_inv[9]);

/**
 * @brief Computes the IMU residual error vector and its analytical Jacobian.
 *
 * States use the physical body->world convention:
 *   state.r = Rodrigues vector of R_bw (body->world rotation)
 *   state.p = position in world frame
 *   state.v = velocity in world frame
 *
 * @param e_imu         Output 9x1 error vector [e_r(3), e_p(3), e_v(3)].
 * @param J_imu         Output 9x18 Jacobian [∂e/∂x_i | ∂e/∂x_j].
 * @param state_i       State at time i (rotation, position, velocity).
 * @param state_j       State at time j (rotation, position, velocity).
 * @param imu_measurement Pre-integrated IMU factor between states i and j.
 * @param g_world       World-frame gravity vector [m/s^2] (3-element array).
 * @param work_memory   Workspace for temporary allocations.
 */
void compute_imu_error_and_jacobian(float* e_imu, float* J_imu,
                                    const CameraState9Dof* state_i,
                                    const CameraState9Dof* state_j,
                                    const imu_factor_t* imu_measurement,
                                    const float g_world[3],
                                    work_memory_t work_memory);

/**
 * @brief Accumulates IMU factor residuals and Jacobians into the system matrix Sb.
 *
 * For each consecutive pair of poses in the window, computes the IMU error and
 * Jacobian, remaps them to the T_cw storage parameterization, and subtracts the
 * weighted contributions into the augmented system matrix Sb (using the -H, -b convention).
 *
 * @param poses        Array of camera poses in T_cw convention.
 * @param num_poses    Number of poses in the current optimization window.
 * @param total_poses  Total number of poses seen so far (for ring-buffer index wrapping).
 * @param velocities   Array of world-frame velocities per keyframe.
 * @param imu_factors  Array of pre-integration factors with bias Jacobians.
 * @param Sb           Augmented system matrix of size (N*9) x (N*9+1), modified in place.
 * @param g            World-frame gravity vector [m/s^2] (3-element array).
 * @param work_memory  Workspace for temporary allocations.
 * @return             Total IMU residual cost (sum of squared weighted errors).
 */
float add_imu_factors(pose_t* poses, uint16_t num_poses,
                      uint16_t total_poses,
                      point3D_float_t* velocities,
                      imu_factor_with_bias_t* imu_factors,
                      float* Sb,
                      float* g,
                      work_memory_t work_memory);

/**
 * @brief Adds a kinematic velocity prior to the system matrix Sb.
 *
 * Constrains each keyframe velocity to be consistent with the finite-difference
 * position displacement: v_j ≈ (p_j - p_i) / dt. This resolves gauge freedom
 * in absolute velocity when IMU data is available.
 *
 * @param Sb          Augmented system matrix of size (N*9) x (N*9+1), modified in place.
 * @param poses       Array of camera poses in T_cw convention.
 * @param velocities  Array of world-frame velocities per keyframe.
 * @param imu_factors Array of IMU factors (used for dt between consecutive keyframes).
 * @param num_poses   Number of poses in the current optimization window.
 * @param total_poses Total number of poses seen so far (for ring-buffer index wrapping).
 * @param lambda      Regularization weight (units: 1/(m/s)^2).
 * @param work_memory Workspace for temporary allocations.
 */
void add_kinematic_velocity_prior(float* Sb,
                                  const pose_t* poses,
                                  const point3D_float_t* velocities,
                                  const imu_factor_with_bias_t* imu_factors,
                                  uint16_t num_poses,
                                  uint16_t total_poses,
                                  float lambda,
                                  work_memory_t work_memory);

#endif /* __IMU_OPTIMIZATION__H__ */