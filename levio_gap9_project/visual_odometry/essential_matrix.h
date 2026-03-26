// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __ESSENTIAL_MATRIX__H__
#define __ESSENTIAL_MATRIX__H__

#include "pmsis.h"
#include "definitions/type_definitions.h"

/**
 * @brief Estimates the essential matrix using the RANSAC algorithm.
 *
 * @param keypoints0        Pointer to the array of 2D keypoints from the first image.
 * @param keypoints1        Pointer to the array of 2D keypoints from the second image.
 * @param matches           Pointer to the array of feature matches between keypoints0 and keypoints1.
 * @param K                 Pointer to the camera intrinsic matrix.
 * @param bestE             Pointer to the output essential matrix (best estimate).
 * @param work_memory       Workspace memory required for computation.
 * @param match_counter     Number of matches to process.
 * @param ransac_iterations Number of RANSAC iterations to perform.
 */
void ransacEssentialMatrix(point2D_u16_t* keypoints0,
                           point2D_u16_t* keypoints1,
                           feature_match_t* matches,
                           matrix_2D_t* K,
                           matrix_2D_t* bestE,
                           work_memory_t work_memory,
                           uint16_t match_counter,
                           uint16_t ransac_iterations);

/**
 * @brief Estimates the essential matrix using the RANSAC algorithm with multicore support.
 *
 * @param keypoints0        Pointer to the array of 2D keypoints from the first image.
 * @param keypoints1        Pointer to the array of 2D keypoints from the second image.
 * @param matches           Pointer to the array of feature matches between keypoints0 and keypoints1.
 * @param K                 Pointer to the camera intrinsic matrix.
 * @param bestE             Pointer to the output essential matrix (best estimate).
 * @param work_memory       Workspace memory required for computation.
 * @param match_counter     Number of matches to process.
 * @param ransac_iterations Number of RANSAC iterations to perform.
 * @param nb_cores          Number of processor cores to use for parallel execution.
 */
void ransacEssentialMatrixMulticore(point2D_u16_t* keypoints0,
                                    point2D_u16_t* keypoints1,
                                    feature_match_t* matches,
                                    matrix_2D_t* K,
                                    matrix_2D_t* bestE,
                                    work_memory_t work_memory,
                                    uint16_t match_counter,
                                    uint16_t ransac_iterations,
                                    uint8_t nb_cores);

/**
 * @brief Recovers the relative camera pose (rotation and translation) from the essential matrix.
 *
 * @param keypoints0           Pointer to the array of 2D keypoints from the first image.
 * @param keypoints1           Pointer to the array of 2D keypoints from the second image.
 * @param matches              Pointer to the array of feature matches between keypoints0 and keypoints1.
 * @param K                    Pointer to the camera intrinsic matrix.
 * @param bestE                Pointer to the estimated essential matrix.
 * @param T                    Pointer to the output transformation matrix (rotation and translation).
 * @param work_memory          Workspace memory required for computation.
 * @param match_counter        Number of matches to process.
 * @param singlecore_execution Flag indicating whether to use single-core execution (1) or multicore (0).
 * @param nb_cores             Number of processor cores to use for parallel execution.
 */
void recoverPose(point2D_u16_t* keypoints0,
                 point2D_u16_t* keypoints1,
                 feature_match_t* matches,
                 matrix_2D_t* K,
                 matrix_2D_t* bestE,
                 matrix_2D_t* T,
                 work_memory_t work_memory,
                 uint16_t match_counter,
                 uint8_t singlecore_execution,
                 uint8_t nb_cores);

#endif /* __ESSENTIAL_MATRIX__H__ */