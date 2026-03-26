// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __IMU_PREINTEGRATION__H__
#define __IMU_PREINTEGRATION__H__

#include "pmsis.h"
#include "definitions/type_definitions.h"

// IMU bias estimates
typedef struct {
    point3D_float_t acc;  // Accelerometer bias
    point3D_float_t gyro; // Gyroscope bias
} ImuBias;

/**
 * @brief Retrieves the current IMU bias estimates of the preintegrator.
 * @return ImuBias structure containing bias values for accelerometer and gyroscope.
 */
ImuBias get_bias();

/**
 * @brief Sets new IMU bias values of the preintegrator.
 * @param new_bias ImuBias structure containing the bias values to set.
 */
void set_bias(ImuBias new_bias);

/**
 * @brief Initializes the IMU preintegrator with the specified cluster device.
 * @param cluster_device Pointer to the cluster device to be used for preintegration.
 */
void initialize_imu_preintegrator(pi_device_t* cluster_device);

/**
 * @brief Processes IMU measurements and performs preintegration calculations.
 * @param imu_measurements_l2 Pointer to array of IMU measurements to process.
 * @param work_memory Work memory structure containing temporary buffers and state.
 */
void process_imu_preintegration(imu_measurement_t* imu_measurements_l2,
                                work_memory_t work_memory);

/**
 * @brief Extracts preintegration results and restarts the preintegrator for next keyframe.
 * @param graph_stats Pointer to pose graph statistics structure to store results.
 * @param work_memory Work memory structure containing preintegration state and buffers.
 */
void extract_and_restart_imu_preintegration(pose_graph_stats_t* graph_stats,
                                            work_memory_t work_memory);


/** @brief Retrieves the current position and time delta from the preintegrator.
 * @param dp Pointer to store the current position delta.
 * @param dt Pointer to store the current time delta.
 */
void get_current_dp_and_dt(point3D_float_t* dp, float* dt);

#endif /* __IMU_PREINTEGRATION__H__ */