// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "imu_preintegration.h"

#include "math.h"

/**
 * @brief Holds the results of the preintegration.
 *
 * This struct contains the delta position, velocity, and rotation accumulated
 * over an integration period, along with their Jacobians with respect to the
 * IMU biases and the propagated covariance matrix.
 */
typedef struct {
    float delta_t; // Total integration time

    // Preintegrated measurements (Deltas)
    point3D_float_t delta_p;   // Delta position
    point3D_float_t delta_v;   // Delta velocity
    mat3x3_t delta_R; // Delta rotation

    // Jacobians of deltas w.r.t. biases
    mat3x3_t J_p_ba; // Jacobian of position w.r.t. accelerometer bias
    mat3x3_t J_p_bg; // Jacobian of position w.r.t. gyroscope bias
    mat3x3_t J_v_ba; // Jacobian of velocity w.r.t. accelerometer bias
    mat3x3_t J_v_bg; // Jacobian of velocity w.r.t. gyroscope bias
    mat3x3_t J_R_bg; // Jacobian of rotation w.r.t. gyroscope bias (Note: dR/d_ba is zero)
} Preintegration;

Preintegration* preintegrator;
ImuBias* bias;

mat3x3_t imu_to_cam_tf = {{IMU_TO_CAM_R00, IMU_TO_CAM_R01, IMU_TO_CAM_R02,
                           IMU_TO_CAM_R10, IMU_TO_CAM_R11, IMU_TO_CAM_R12,
                           IMU_TO_CAM_R20, IMU_TO_CAM_R21, IMU_TO_CAM_R22}};

void fetch_imu_measurements(imu_measurement_t* imu_measurements_l2,
                            imu_measurement_t* imu_measurements)
{
    pi_cl_dma_cmd_t cmd;
    /* DMA Copy Frame */
    pi_cl_dma_cmd((uint32_t) imu_measurements_l2, (uint32_t) imu_measurements, IMU_HEIGHT*sizeof(imu_measurement_t), PI_CL_DMA_DIR_EXT2LOC, &cmd);
    /* Wait for DMA transfer to finish. */
    pi_cl_dma_wait(&cmd);
}

// --- Utility Functions ---
void print_vec3(const char* name, const point3D_float_t* v) {
    printf("%s: [%.4f, %.4f, %.4f]\n", name, v->x, v->y, v->z);
}

void print_mat3x3(const char* name, const mat3x3_t* A) {
    printf("%s:\n", name);
    for (int i = 0; i < 3; i++) {
        printf("  [%.4f, %.4f, %.4f]\n", A->m[i*3 + 0], A->m[i*3 + 1], A->m[i*3 + 2]);
    }
}

/**
 * @brief Resets the Preintegration struct to an initial state.
 * @param p Pointer to the Preintegration struct to be reset.
 */
void preintegration_reset(Preintegration* p) {
    p->delta_t = 0.0f;

    // Deltas are initialized to zero/identity
    p->delta_p = (point3D_float_t){0, 0, 0};
    p->delta_v = (point3D_float_t){0, 0, 0};
    mat3x3_identity(&p->delta_R);

    // Jacobians are initialized to zero
    memset(&p->J_p_ba, 0, sizeof(mat3x3_t));
    memset(&p->J_p_bg, 0, sizeof(mat3x3_t));
    memset(&p->J_v_ba, 0, sizeof(mat3x3_t));
    memset(&p->J_v_bg, 0, sizeof(mat3x3_t));
    memset(&p->J_R_bg, 0, sizeof(mat3x3_t));
}

/**
 * @brief Integrates a single IMU measurement using a trapezoidal (mid-point) rule.
 *
 * @param p         Pointer to the Preintegration struct to update.
 * @param m         The new IMU measurement (input).
 * @param dt        The time delta for this integration step (input).
 * @param bias      The current estimate of IMU biases (input).
 */
void preintegration_integrate(
    Preintegration* p,
    const imu_measurement_t* m,
    float dt,
    const ImuBias* bias,
    work_memory_t work_memory)
{
    // 1. Correct measurements with current bias estimates
    point3D_float_t* acc_cam_frame = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    point3D_float_t* gyro_cam_frame = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    point3D_float_t* acc_unbiased = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    point3D_float_t* gyro_unbiased = allocate_work_memory(&work_memory, sizeof(point3D_float_t));

    mat3x3_vec3_mul(&imu_to_cam_tf, &m->acc, acc_cam_frame);
    mat3x3_vec3_mul(&imu_to_cam_tf, &m->gyro, gyro_cam_frame);
    vec3_sub(acc_cam_frame, &bias->acc, acc_unbiased);
    vec3_sub(gyro_cam_frame, &bias->gyro, gyro_unbiased);

    // 2. Calculate rotation increment and update total rotation
    point3D_float_t* rot_vec = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    mat3x3_t* dR = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    mat3x3_t* dR_T = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    
    vec3_scale(gyro_unbiased, dt, rot_vec);
    mat3x3_rodrigues_exp(rot_vec, dR);

    // Keep a copy of the rotation at the beginning of the step for updates
    mat3x3_t* R_old = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    *R_old = p->delta_R;
    
    // Update total rotation first
    mat3x3_mul(R_old, dR, &p->delta_R);

    // 3. Propagate state (Deltas) using Trapezoidal Integration
    mat3x3_t* acc_skew = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    mat3x3_skew_symmetric(acc_unbiased, acc_skew);

    point3D_float_t* acc_world_old = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    point3D_float_t* acc_world_new = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    mat3x3_vec3_mul(R_old, acc_unbiased, acc_world_old);
    mat3x3_vec3_mul(&p->delta_R, acc_unbiased, acc_world_new);

    point3D_float_t* acc_avg = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    vec3_add(acc_world_old, acc_world_new, acc_avg);
    vec3_scale(acc_avg, 0.5f, acc_avg);

    // First, calculate the position increment from the initial velocity
    point3D_float_t* pos_vel_term = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    vec3_scale(&p->delta_v, dt, pos_vel_term); // pos_vel_term = v_k * dt

    // Next, calculate the position increment from the average acceleration
    point3D_float_t* vel_increment = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    vec3_scale(acc_avg, dt, vel_increment); // This is also the velocity update term
    point3D_float_t* pos_acc_term = allocate_work_memory(&work_memory, sizeof(point3D_float_t));
    vec3_scale(vel_increment, 0.5f * dt, pos_acc_term); // pos_acc_term = 0.5 * a_avg * dt^2

    // Update position by adding both terms
    vec3_add(&p->delta_p, pos_vel_term, &p->delta_p);
    vec3_add(&p->delta_p, pos_acc_term, &p->delta_p);
    
    // Update velocity
    vec3_add(&p->delta_v, vel_increment, &p->delta_v);

    p->delta_t += dt;

    // 4. Propagate Jacobians
    mat3x3_t* I = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    mat3x3_identity(I);
    mat3x3_t* temp_mat1 = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    mat3x3_t* temp_mat2 = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    
    // Update Jacobians w.r.t bias_g
    mat3x3_transpose(dR, dR_T);
    mat3x3_scale(I, -1.0f * dt, temp_mat1);
    mat3x3_mul(dR_T, &p->J_R_bg, temp_mat2);
    mat3x3_add(temp_mat2, temp_mat1, &p->J_R_bg);
    
    mat3x3_t* R_acc_skew = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    mat3x3_t* term_v_bg = allocate_work_memory(&work_memory, sizeof(mat3x3_t));
    mat3x3_mul(R_old, acc_skew, R_acc_skew);
    mat3x3_mul(R_acc_skew, &p->J_R_bg, term_v_bg);
    mat3x3_scale(term_v_bg, -1.0f * dt, term_v_bg);

    mat3x3_scale(&p->J_v_bg, dt, temp_mat1);
    mat3x3_add(&p->J_p_bg, temp_mat1, &p->J_p_bg);
    mat3x3_scale(term_v_bg, 0.5f * dt, temp_mat1);
    mat3x3_add(&p->J_p_bg, temp_mat1, &p->J_p_bg);
    mat3x3_add(&p->J_v_bg, term_v_bg, &p->J_v_bg);

    // Update Jacobians w.r.t bias_a
    mat3x3_scale(&p->J_v_ba, dt, temp_mat1);
    mat3x3_add(&p->J_p_ba, temp_mat1, &p->J_p_ba);
    mat3x3_scale(R_old, -0.5f * dt * dt, temp_mat1);
    mat3x3_add(&p->J_p_ba, temp_mat1, &p->J_p_ba);

    mat3x3_scale(R_old, -1.0f * dt, temp_mat1);
    mat3x3_add(&p->J_v_ba, temp_mat1, &p->J_v_ba);
}

ImuBias get_bias()
{
    return *bias;
}

void set_bias(ImuBias new_bias)
{
    *bias = new_bias;
}

void initialize_imu_preintegrator(pi_device_t* cluster_device)
{
    preintegrator = pi_cl_l1_malloc(cluster_device, sizeof(Preintegration));
    bias = pi_cl_l1_malloc(cluster_device, sizeof(ImuBias));
    ImuBias init_bias = {
        .acc = {0.0f, 0.0f, 0.0f}, // Assume some small initial bias
        .gyro = {0.0f, 0.0f, 0.0f}
    };
    *bias = init_bias;
    preintegration_reset(preintegrator);
}

void process_imu_preintegration(imu_measurement_t* imu_measurements_l2,
                                work_memory_t work_memory)
{
    imu_measurement_t* imu_measurements = allocate_work_memory(&work_memory, IMU_HEIGHT*sizeof(imu_measurement_t));
    fetch_imu_measurements(imu_measurements_l2, imu_measurements);
    LOG_INFO("Performing IMU preintegration\n");

    float dt = IMU_SAMPLING_PERIOD;

    for (int i = 0; i < IMU_HEIGHT; ++i) {
        preintegration_integrate(preintegrator, &imu_measurements[i], dt, bias, work_memory);
    }
}

void extract_and_restart_imu_preintegration(pose_graph_stats_t* graph_stats,
                                            work_memory_t work_memory)
{
     // --- Results ---
    LOG_WARNING("\n*** Preintegration Results ***\n");
    LOG_WARNING("Total integration time: %.2f s\n", preintegrator->delta_t);

    print_vec3("Delta Position", &preintegrator->delta_p);
    print_vec3("Delta Velocity", &preintegrator->delta_v);
    print_mat3x3("Delta Rotation", &preintegrator->delta_R);
    print_mat3x3("Jacobi Pos acc", &preintegrator->J_p_ba);
    print_mat3x3("Jacobi Pos gyr", &preintegrator->J_p_bg);
    print_mat3x3("Jacobi Rot gyr", &preintegrator->J_R_bg);
    print_mat3x3("Jacobi Vel acc", &preintegrator->J_v_ba);
    print_mat3x3("Jacobi Vel gyr", &preintegrator->J_v_bg);
    LOG_WARNING("***\n\n");
    imu_factor_with_bias_t* imu_factors = allocate_work_memory(&work_memory,MAX_KEYFRAMES*sizeof(imu_factor_with_bias_t));
    uint8_t current_keyframe_index = (graph_stats->total_keyframes-1) % MAX_KEYFRAMES;
    storage_move_in(imu_factors, IMU_FACTORS);
    imu_factors[current_keyframe_index].base.dt = preintegrator->delta_t;
    imu_factors[current_keyframe_index].base.dp = preintegrator->delta_p;
    imu_factors[current_keyframe_index].base.dv = preintegrator->delta_v;
    for (int i = 0; i < 9; ++i) {
        imu_factors[current_keyframe_index].base.dR[i] = preintegrator->delta_R.m[i];
    }
    for (int i = 0; i < 9; ++i) {
        imu_factors[current_keyframe_index].J_p_ba[i] = preintegrator->J_p_ba.m[i];
        imu_factors[current_keyframe_index].J_p_bg[i] = preintegrator->J_p_bg.m[i];
        imu_factors[current_keyframe_index].J_r_bg[i] = preintegrator->J_R_bg.m[i];
        imu_factors[current_keyframe_index].J_v_ba[i] = preintegrator->J_v_ba.m[i];
        imu_factors[current_keyframe_index].J_v_bg[i] = preintegrator->J_v_bg.m[i];
    }
    storage_move_out(imu_factors, IMU_FACTORS);
    preintegration_reset(preintegrator);
}

void get_current_dp_and_dt(point3D_float_t* dp, float* dt)
{
    memcpy(dp, &preintegrator->delta_p, sizeof(point3D_float_t));
    memcpy(dt, &preintegrator->delta_t, sizeof(float));
}