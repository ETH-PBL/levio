// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __EPNP__H__
#define __EPNP__H__

#include "pmsis.h"
#include "definitions/type_definitions.h"

/**
 * @brief Estimates the camera pose using the RANSAC-based EPnP algorithm.
 *
 * This function computes the camera pose (extrinsic parameters) from a set of 3D-2D point correspondences
 * using the Efficient Perspective-n-Point (EPnP) algorithm within a RANSAC framework for robustness against outliers.
 *
 * @param world_points Pointer to an array of 3D world points (point3D_float_t).
 * @param image_points Pointer to an array of corresponding 2D image points (point2D_u16_t).
 * @param K Pointer to the camera intrinsic matrix (matrix_2D_t).
 * @param T Pointer to the output transformation matrix (matrix_2D_t) representing the estimated pose.
 * @param work_memory Workspace memory required for intermediate computations.
 * @param number_of_correspondences Number of 3D-2D point correspondences.
 * @param ransac_iterations Number of RANSAC iterations to perform.
 * @return The reprojection error of the best pose found.
 */

float ransac_epnp_compute_pose(point3D_float_t* world_points,
                               point2D_u16_t* image_points,
                               matrix_2D_t* K,
                               matrix_2D_t* T,
                               work_memory_t work_memory,
                               uint16_t number_of_correspondences,
                               uint16_t ransac_iterations);

/**
 * @brief Estimates the camera pose using the RANSAC-based EPnP algorithm with multicore support.
 *
 * This function is similar to ransac_epnp_compute_pose, but leverages multiple cores to accelerate the RANSAC process.
 *
 * @param world_points Pointer to an array of 3D world points (point3D_float_t).
 * @param image_points Pointer to an array of corresponding 2D image points (point2D_u16_t).
 * @param K Pointer to the camera intrinsic matrix (matrix_2D_t).
 * @param T Pointer to the output transformation matrix (matrix_2D_t) representing the estimated pose.
 * @param work_memory Workspace memory required for intermediate computations.
 * @param number_of_correspondences Number of 3D-2D point correspondences.
 * @param ransac_iterations Number of RANSAC iterations to perform.
 * @param nb_cores Number of processor cores to use for parallelization.
 * @return The reprojection error of the best pose found.
 */
float ransac_epnp_compute_pose_multicore(point3D_float_t* world_points,
                                         point2D_u16_t* image_points,
                                         matrix_2D_t* K,
                                         matrix_2D_t* T,
                                         work_memory_t work_memory,
                                         uint16_t number_of_correspondences,
                                         uint16_t ransac_iterations,
                                         uint8_t nb_cores);

#endif /* __EPNP__H__ */