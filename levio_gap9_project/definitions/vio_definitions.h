// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __VIO_DEFINITIONS_H__
#define __VIO_DEFINITIONS_H__

#include "pmsis.h"
#include "config.h"
#include "matrix.h"

typedef struct point2D_u16
{
    uint16_t x;
    uint16_t y;
} point2D_u16_t;

typedef struct point2D_float
{
    float x;
    float y;
} point2D_float_t;

typedef struct point3D_float
{
    float x;
    float y;
    float z;
} point3D_float_t;

typedef struct imu_measurement
{
	float t_stamp;
	point3D_float_t acc;
	point3D_float_t gyro;
} imu_measurement_t;

typedef struct
{
    uint16_t feat_idx0;
    uint16_t feat_idx1;
    uint8_t match_score;
} feature_match_t;

typedef struct
{
    uint16_t width;
    uint16_t height;
    uint8_t* pixels;
} image_data_t;

typedef uint32_t orb_descriptor_t[8];

typedef struct orb_features
{
    point2D_u16_t* kpts;
    orb_descriptor_t* descs;
    uint16_t kpt_counter;
    uint16_t kpt_capacity;
} orb_features_t;


/**
 * @brief Computes the average parallax between matched keypoints from two images.
 *
 * This function calculates the average parallax (displacement) between corresponding keypoints
 * in two images, given their coordinates and a set of feature matches. The camera intrinsic
 * matrix is used for normalization.
 *
 * @param keypoints0        Pointer to the array of keypoints from the first image.
 * @param keypoints1        Pointer to the array of keypoints from the second image.
 * @param matches           Pointer to the array of feature matches between the two sets of keypoints.
 * @param K                 Pointer to the camera intrinsic matrix.
 * @param match_counter     Number of valid matches in the matches array.
 * @return                  The average parallax value as a float.
 */
float averageParallax(point2D_u16_t* keypoints0,
                      point2D_u16_t* keypoints1,
                      feature_match_t* matches,
                      matrix_2D_t* K,
                      uint16_t match_counter);

/**
 * @brief Computes the reprojection error vector between a 3D world point and its corresponding 2D image point.
 *
 * @param world_point Pointer to the 3D point in world coordinates.
 * @param image_point Pointer to the 2D point in image coordinates (pixel coordinates).
 * @param T Pointer to the transformation matrix (e.g., camera pose).
 * @param K Pointer to the camera intrinsic matrix.
 * @param e Pointer to the output error vector (size 2).
 */
void reprojection_error_vector(point3D_float_t* world_point,
                               point2D_u16_t* image_point,
                               matrix_2D_t* T,
                               matrix_2D_t* K,
                               float* e);

/**
 * @brief Computes the squared reprojection error between a 3D world point and its corresponding 2D image point.
 *
 * @param world_point Pointer to the 3D point in world coordinates.
 * @param image_point Pointer to the 2D point in image coordinates (pixel coordinates).
 * @param T Pointer to the transformation matrix (e.g., camera pose).
 * @param K Pointer to the camera intrinsic matrix.
 * @return The squared reprojection error as a float.
 */
float reprojection_error_squared(point3D_float_t* world_point,
                                 point2D_u16_t* image_point,
                                 matrix_2D_t* T,
                                 matrix_2D_t* K);

/**
 * @brief Computes the reprojection error (Euclidean distance) between a 3D world point and its corresponding 2D image point.
 *
 * @param world_point Pointer to the 3D point in world coordinates.
 * @param image_point Pointer to the 2D point in image coordinates (pixel coordinates).
 * @param T Pointer to the transformation matrix (e.g., camera pose).
 * @param K Pointer to the camera intrinsic matrix.
 * @return The reprojection error as a float.
 */
float reprojection_error(point3D_float_t* world_point,
                         point2D_u16_t* image_point,
                         matrix_2D_t* T,
                         matrix_2D_t* K);


#endif /* __VIO_DEFINITIONS_H__ */