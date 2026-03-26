// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __TRIANGULATION__H__
#define __TRIANGULATION__H__

#include "pmsis.h"
#include "definitions/type_definitions.h"

/**
 * @brief Triangulates a single 3D point from two 2D observations.
 *
 * @param pt0 Pointer to the 2D point in the first image.
 * @param pt1 Pointer to the 2D point in the second image.
 * @param K   Pointer to the camera intrinsic matrix.
 * @param T0  Pointer to the pose (extrinsic matrix) of the first camera.
 * @param T1  Pointer to the pose (extrinsic matrix) of the second camera.
 * @param work_memory Workspace memory required for computation.
 * @return The triangulated 3D point in float representation.
 */
point3D_float_t triangulatePoint(point2D_u16_t* pt0,
                                 point2D_u16_t* pt1,
                                 matrix_2D_t* K,
                                 matrix_2D_t* T0,
                                 matrix_2D_t* T1,
                                 work_memory_t work_memory);

/**
 * @brief Triangulates multiple 3D points from matched 2D keypoints.
 *
 * @param keypoints0         Pointer to array of 2D keypoints in the first image.
 * @param keypoints1         Pointer to array of 2D keypoints in the second image.
 * @param matches            Pointer to array of feature matches between the two images.
 * @param K                  Pointer to the camera intrinsic matrix.
 * @param T0                 Pointer to the pose (extrinsic matrix) of the first camera.
 * @param T1                 Pointer to the pose (extrinsic matrix) of the second camera.
 * @param triangulatedPoints Pointer to output array for triangulated 3D points.
 * @param work_memory        Workspace memory required for computation.
 * @param match_counter      Number of matches to process.
 */
void triangulatePoints(point2D_u16_t* keypoints0,
                       point2D_u16_t* keypoints1,
                       feature_match_t* matches,
                       matrix_2D_t* K,
                       matrix_2D_t* T0,
                       matrix_2D_t* T1,
                       point3D_float_t* triangulatedPoints,
                       work_memory_t work_memory,
                       uint16_t match_counter);

/**
 * @brief Triangulates multiple 3D points using multiple cores for parallel processing.
 *
 * @param keypoints0         Pointer to array of 2D keypoints in the first image.
 * @param keypoints1         Pointer to array of 2D keypoints in the second image.
 * @param matches            Pointer to array of feature matches between the two images.
 * @param K                  Pointer to the camera intrinsic matrix.
 * @param T0                 Pointer to the pose (extrinsic matrix) of the first camera.
 * @param T1                 Pointer to the pose (extrinsic matrix) of the second camera.
 * @param triangulatedPoints Pointer to output array for triangulated 3D points.
 * @param work_memory        Workspace memory required for computation.
 * @param match_counter      Number of matches to process.
 * @param nb_cores           Number of cores to use for parallel processing.
 */
void triangulatePointsMulticore(point2D_u16_t* keypoints0,
                                point2D_u16_t* keypoints1,
                                feature_match_t* matches,
                                matrix_2D_t* K,
                                matrix_2D_t* T0,
                                matrix_2D_t* T1,
                                point3D_float_t* triangulatedPoints,
                                work_memory_t work_memory,
                                uint16_t match_counter,
                                uint8_t nb_cores);

#endif /* __TRIANGULATION__H__ */