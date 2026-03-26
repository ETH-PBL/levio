// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __IMU_CALIB__H__
#define __IMU_CALIB__H__

#include "definitions/type_definitions.h"

/* ------------------------------------------------------------------ */
/* Calibration result                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    point3D_float_t gyro_bias;    /* b_g  [rad/s]  */
    point3D_float_t gravity_b;    /* g_b  [m/s^2] in sensor/body frame */
                                      /* NOTE: actually (g_b - J_v_ba*b_a/T)
                                         unless b_a_prior is supplied        */
} imu_calib_result_t;

/**
 * @brief Estimates gyro bias and gravity direction from a stationary IMU pre-integration window,
 *        using pre-computed bias Jacobians (Forster et al. TRO 2017).
 *
 * @param factor    Pre-integration factor with bias Jacobians.
 * @param b_a_prior Prior accelerometer bias estimate [m/s^2]; pass zero-vector if unknown.
 * @param w_vel     Weight for the velocity-derived gravity estimate (0..1, w_vel+w_pos=1).
 * @param w_pos     Weight for the position-derived gravity estimate (0..1, w_vel+w_pos=1).
 * @param out       Output calibration result (gyro_bias, gravity_b).
 * @return 0 on success, -1 if J_r_bg is singular (cannot solve for gyro bias).
 */
int imu_calib_stationary_with_jacobians(const imu_factor_with_bias_t *factor,
                                        point3D_float_t               b_a_prior,
                                        float                         w_vel,
                                        float                         w_pos,
                                        imu_calib_result_t            *out);

/**
 * @brief Estimates gyro bias and gravity direction from a stationary IMU pre-integration window,
 *        without pre-computed Jacobians (solves using rotation integral matrices A_v and A_p).
 *
 * @param factor Pre-integration factor (dt, dp, dv, dR).
 * @param w_vel  Weight for the velocity-derived gravity estimate (0..1, w_vel+w_pos=1).
 * @param w_pos  Weight for the position-derived gravity estimate (0..1, w_vel+w_pos=1).
 * @param out    Output calibration result (gyro_bias, gravity_b).
 * @return 0 on success, -1 if A_v is singular, -2 if A_p is singular.
 */
int imu_calib_stationary(const imu_factor_t *factor,
                         float                   w_vel,
                         float                   w_pos,
                         imu_calib_result_t     *out);

/**
 * @brief Constructs a synthetic zero-motion IMU factor with bias Jacobians
 *        for a given gyro bias and body-frame gravity vector.
 *
 * Useful for testing calibration routines or initializing the optimizer
 * when the sensor is known to be stationary.
 *
 * @param T   Integration time [s].
 * @param bg  Gyroscope bias [rad/s] used to compute Jacobians.
 * @param g_b Gravity vector in the body/sensor frame [m/s^2].
 * @param out Output zero-motion factor with filled base measurements and Jacobians.
 */
void imu_make_zero_motion_factor(float                  T,
                                 point3D_float_t        bg,
                                 point3D_float_t        g_b,
                                 imu_factor_with_bias_t *out);

#endif /* __IMU_CALIB__H__ */