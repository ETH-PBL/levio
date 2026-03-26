// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __CONFIG__H__
#define __CONFIG__H__

/* ===================================================================
 * LEVIO VIO Configuration
 *
 * Central configuration file for all tunable parameters.
 * Modify these values to adapt the system to different cameras,
 * IMU sensors, or operating conditions.
 * =================================================================== */

/* -------------------------------------------------------------------
 * Camera Intrinsics
 * ------------------------------------------------------------------- */
#define CAMERA_FX  114.66f
#define CAMERA_FY  114.32f
#define CAMERA_CX   77.8f
#define CAMERA_CY   62.09f

/* -------------------------------------------------------------------
 * Image Dimensions
 * ------------------------------------------------------------------- */
#define IMG_WIDTH  160
#define IMG_HEIGHT 120
#define IMG_SIZE   (IMG_WIDTH * IMG_HEIGHT)

/* -------------------------------------------------------------------
 * Camera / Frame Rate
 * ------------------------------------------------------------------- */
#define CAMERA_FPS 20.0f

/* -------------------------------------------------------------------
 * IMU Configuration
 * ------------------------------------------------------------------- */
#define IMU_SAMPLING_PERIOD  0.005f  /* 200 Hz IMU rate (seconds) */

/* IMU-to-camera extrinsic rotation (row-major 3x3) */
#define IMU_TO_CAM_R00  0.01486554f
#define IMU_TO_CAM_R01  0.99955725f
#define IMU_TO_CAM_R02 -0.02577444f
#define IMU_TO_CAM_R10 -0.99988093f
#define IMU_TO_CAM_R11  0.01496721f
#define IMU_TO_CAM_R12  0.00375619f
#define IMU_TO_CAM_R20  0.0041403f
#define IMU_TO_CAM_R21  0.02571553f
#define IMU_TO_CAM_R22  0.99966073f

/* -------------------------------------------------------------------
 * Processing Configuration
 * ------------------------------------------------------------------- */
#define NB_CORES    8
#define SINGLE_CORE 1

/* L1 scratch-pad memory budget (bytes) */
#define L1_WORK_MEMORY_SIZE 110000

/* Cluster and fabric controller frequency (Hz) */
#define CLUSTER_FREQUENCY_HZ (370 * 1000 * 1000)

/* -------------------------------------------------------------------
 * ORB Feature Detection
 * ------------------------------------------------------------------- */
#define FAST_THRESHOLD     20
#define HARRIS_THRESHOLD 5000
#define PATCH_SIZE          8
#define FEATURES_PER_PATCH  4

/* -------------------------------------------------------------------
 * Feature Matching
 * ------------------------------------------------------------------- */
#define HAMMING_THRESHOLD    35
#define MAX_FLOW            200
#define DISABLE_MAX_FLOW      0

/* -------------------------------------------------------------------
 * Pose Graph Dimensions
 * ------------------------------------------------------------------- */
#define MAX_KEYPOINTS       704
#define MAX_WORLD_FEATURES 1000
#define MAX_KEYFRAMES         8
#define MAX_OBSERVATIONS   (200 * 12)

/* -------------------------------------------------------------------
 * RANSAC Iterations
 * ------------------------------------------------------------------- */
#define EPNP_ITERS       64
#define ESSENTIAL_ITERS 640

/* -------------------------------------------------------------------
 * Visual Odometry Thresholds
 * ------------------------------------------------------------------- */
#define MIN_MATCHES_EPNP         25
#define MIN_MATCHES_ESSENTIAL    10    /* 8 for algorithm + 2 margin */
#define MAX_WORLD_POINT_AGE      15
#define REPROJECTION_ERROR_SQ   2.0f   /* squared pixel threshold */
#define KEYFRAME_PARALLAX_THRESHOLD 0.05f
#define SAMPSON_PIXEL_THRESHOLD 0.7f   /* pixel error for essential matrix inlier test */
#define EPNP_INLIER_THRESHOLD_SQ 2.0f /* sqrtf(2.0) pixels, squared = 2.0 */
#define MIN_TRANSLATION_SCALE   0.01f  /* minimum scale for relative translation */
#define BEHIND_CAMERA_PENALTY   1e3f   /* penalty weight for points with negative depth */

/* -------------------------------------------------------------------
 * Optimizer Parameters
 * ------------------------------------------------------------------- */
#define UPDATE_WEIGHT  1.0f
#define HUBER_DELTA    1.0f
#define LM_DAMPING     1e-4f  /* Levenberg-Marquardt damping factor */

/* -------------------------------------------------------------------
 * IMU Noise Parameters
 * ------------------------------------------------------------------- */
#define GYRO_NOISE_DENSITY          0.02f    /* rad/s/sqrt(Hz)   */
#define ACCEL_NOISE_DENSITY         0.1f     /* m/s^2/sqrt(Hz)   */
#define GRAVITY_UNCERTAINTY         1.0f     /* m/s^2            */
#define IMU_WEIGHT_SCALE            1.0f
#define GYRO_BIAS_RANDOM_WALK_NOISE  2.0e-5f /* rad/s            */
#define ACCEL_BIAS_RANDOM_WALK_NOISE 3.0e-4f /* m/s^2            */

/* -------------------------------------------------------------------
 * IMU Data Dimensions
 * ------------------------------------------------------------------- */
#define IMU_WIDTH  28
#define IMU_HEIGHT 10

/* -------------------------------------------------------------------
 * Gravity
 * ------------------------------------------------------------------- */
#define GRAVITY_MAGNITUDE 9.81f

/* Initial gravity estimate (body frame, m/s^2).
 * Updated at runtime by the IMU calibration. */
#define GRAVITY_INIT_X  1.6978f
#define GRAVITY_INIT_Y  9.0580f
#define GRAVITY_INIT_Z  3.3624f

/* -------------------------------------------------------------------
 * Numerical Tolerances (matrix / SVD internals)
 * ------------------------------------------------------------------- */
#define SVD_MAX_ITER_JACOBI        10
#define SVD_TOL_JACOBI          1e-12f
#define MAX_ITER_INV_POWER         15
#define TOL_INV_POWER           1e-10f
#define NUMERICAL_ZERO_THRESHOLD 1e-12f
#define VECTOR_ZERO_NORM_THRESHOLD 1e-26f

/* -------------------------------------------------------------------
 * Logging
 * ------------------------------------------------------------------- */
#define LOG_LEVEL          0x02  /* 0x00=ERROR, 0x01=WARNING, 0x02=INFO, 0x03=DEBUG */
#define LOG_TIMING_ENABLED 0x01

#endif /* __CONFIG__H__ */