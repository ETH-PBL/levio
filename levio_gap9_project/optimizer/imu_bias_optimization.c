// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "imu_bias_optimization.h"
#include "math.h"
#include "pmsis.h"

#define ERROR_SIZE_IMU  9
#define ERROR_SIZE_BIAS 6

#define Max(a, b)       (((a)>(b))?(a):(b))


/**
 * @brief Performs Cholesky decomposition on a symmetric positive-definite matrix.
 *
 * Decomposes the matrix A into L, where A = L * L^T. The input matrix A is
 * expected to be symmetric and positive-definite. The decomposition is done in-place.
 * Only the lower triangle of A is read and written.
 *
 * @param A Pointer to the start of the (n x n) matrix A.
 * @param n The dimension of the matrix.
 * @return 1 on success, 0 if the matrix is not positive-definite.
 */
int cholesky_decompose(float* A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += A[i * n + k] * A[j * n + k];
            }

            if (i == j) {
                float d = A[i * n + i] - sum;
                if (d <= 1e-12f) {
                    // Matrix is not positive-definite or is ill-conditioned.
                    return 0;
                }
                A[i * n + i] = sqrtf(d);
            } else {
                A[i * n + j] = (1.0f / A[j * n + j] * (A[i * n + j] - sum));
            }
        }
    }
    // The lower triangular part of A now holds the Cholesky factor L.
    return 1;
}

/**
 * @brief Solves the system Ax = b using a pre-computed Cholesky decomposition.
 *
 * @param L The Cholesky factor L (the lower triangular matrix from decomposition).
 * @param b The vector b in the equation Ax = b.
 * @param x The solution vector x (output).
 * @param n The dimension of the system.
 */
void cholesky_solve(const float* L, const float* b, float* x, int n) {
    // Validate input parameters
    if (!L || !b || !x || n <= 0) return;
    
    // Step 1: Solve L*y = b for y using forward substitution.
    // The result y is stored in the output vector x temporarily.
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < i; j++) {
            sum += L[i * n + j] * x[j];
        }
        x[i] = (b[i] - sum) / L[i * n + i];
    }

    // Step 2: Solve L^T*x = y for x using backward substitution.
    // The vector y is already in x.
    for (int i = n - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int j = i + 1; j < n; j++) {
            sum += L[j * n + i] * x[j]; // Using L[j*n+i] for L^T
        }
        x[i] = (x[i] - sum) / L[i * n + i];
    }
}

/**
 * @brief Solves the dense system Ax=b using Cholesky decomposition.
 *
 * This function extracts the matrix A and vector b from an augmented matrix Ab,
 * performs Cholesky decomposition on A, and then solves for x.
 *
 * @param Ab Pointer to the augmented matrix [A|b] of size n x (n+1).
 * @param x  Pointer to the output solution vector x.
 * @param n  The dimension of the system.
 * @param work_memory Workspace for temporary storage.
 * @return 1 on success, 0 on failure (e.g., matrix not positive-definite).
 */
int solve_cholesky_system(float* Ab, float* x, int n, work_memory_t work_memory) {
    // Allocate temporary memory for the matrix A and vector b.
    float* A = allocate_work_memory(&work_memory, n * n * sizeof(float));
    float* b = allocate_work_memory(&work_memory, n * sizeof(float));
    if (!A || !b) {
        // Not enough memory
        return 0;
    }

    // Extract A and b from the augmented matrix Ab
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = Ab[i * (n + 1) + j];
        }
        b[i] = Ab[i * (n + 1) + n];
    }

    // Perform Cholesky decomposition in-place on A.
    // A will become the lower-triangular matrix L.
    if (!cholesky_decompose(A, n)) {
        LOG_DEBUG("Cholesky decomposition failed. Matrix may not be positive-definite.\n");
        return 0;
    }

    // Solve the system using the decomposed matrix.
    // The matrix A now contains L.
    cholesky_solve(A, b, x, n);

    return 1;
}

/**
 * @brief Corrects pre-integrated measurements using a first-order approximation for bias change.
 */
static void correct_preintegration_for_bias(
    float dR_corr[9], float dp_corr[3], float dv_corr[3],
    const imu_factor_with_bias_t* factor,
    const float delta_b_g[3], const float delta_b_a[3],
    work_memory_t work_memory)
{
    // Correct Rotation: ΔR' = ΔR * Exp(-J_r_bg * δb_g)
    float J_r_bg_delta_b_g[3];
    mat33_vec3_mult(factor->J_r_bg, delta_b_g, J_r_bg_delta_b_g);
    for(int i=0; i<3; ++i) J_r_bg_delta_b_g[i] *= -1.0f; // Exp(-phi)
    
    float* R_corr = allocate_work_memory(&work_memory, 9 * sizeof(float));
    rodriguesToMatrix(J_r_bg_delta_b_g, R_corr);

    const float* dR_orig = factor->base.dR;
    for(int r=0; r<3; ++r) {
        for(int c=0; c<3; ++c) {
            dR_corr[r*3+c] = dR_orig[r*3+0]*R_corr[0*3+c] + dR_orig[r*3+1]*R_corr[1*3+c] + dR_orig[r*3+2]*R_corr[2*3+c];
        }
    }

    // Correct Position: Δp' = Δp + J_p_bg*δb_g + J_p_ba*δb_a
    float Jp_bg_dbg[3], Jp_ba_dba[3];
    mat33_vec3_mult(factor->J_p_bg, delta_b_g, Jp_bg_dbg);
    mat33_vec3_mult(factor->J_p_ba, delta_b_a, Jp_ba_dba);
    dp_corr[0] = factor->base.dp.x + Jp_bg_dbg[0] + Jp_ba_dba[0];
    dp_corr[1] = factor->base.dp.y + Jp_bg_dbg[1] + Jp_ba_dba[1];
    dp_corr[2] = factor->base.dp.z + Jp_bg_dbg[2] + Jp_ba_dba[2];

    // Correct Velocity: Δv' = Δv + J_v_bg*δb_g + J_v_ba*δb_a
    float Jv_bg_dbg[3], Jv_ba_dba[3];
    mat33_vec3_mult(factor->J_v_bg, delta_b_g, Jv_bg_dbg);
    mat33_vec3_mult(factor->J_v_ba, delta_b_a, Jv_ba_dba);
    dv_corr[0] = factor->base.dv.x + Jv_bg_dbg[0] + Jv_ba_dba[0];
    dv_corr[1] = factor->base.dv.y + Jv_bg_dbg[1] + Jv_ba_dba[1];
    dv_corr[2] = factor->base.dv.z + Jv_bg_dbg[2] + Jv_ba_dba[2];
}


/**
 * @brief Computes the IMU error and its full Jacobian w.r.t two 15-DoF states.
 * (This is the complete helper function required by the sequential optimizer)
 */
void compute_imu_bias_error_and_jacobian(
    float e_imu[9], float J_imu[9 * STATE_SIZE_BIAS * 2],
    const CameraState15Dof* state_i, const CameraState15Dof* state_j,
    const imu_factor_with_bias_t* imu_measurement,
    const float g_world[3],
    work_memory_t work_memory)
{
    const float delta_b_g[3] = {state_i->b_g[0], state_i->b_g[1], state_i->b_g[2]};
    const float delta_b_a[3] = {state_i->b_a[0], state_i->b_a[1], state_i->b_a[2]};

    // Correct pre-integrated measurements with current best estimate of bias_i
    float* dR_corr = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float dp_corr[3], dv_corr[3];
    correct_preintegration_for_bias(dR_corr, dp_corr, dv_corr, imu_measurement, delta_b_g, delta_b_a, work_memory);

    // --- Compute Error Vector (9x1) ---
    float* J_imu_base = allocate_work_memory(&work_memory, ERROR_SIZE_IMU * 9 * 2 * sizeof(float)); // 9x18 Jacobian from original state
    CameraState9Dof state_i_9dof = {{state_i->r[0], state_i->r[1], state_i->r[2]}, {state_i->p[0], state_i->p[1], state_i->p[2]}, {state_i->v[0], state_i->v[1], state_i->v[2]}};
    CameraState9Dof state_j_9dof = {{state_j->r[0], state_j->r[1], state_j->r[2]}, {state_j->p[0], state_j->p[1], state_j->p[2]}, {state_j->v[0], state_j->v[1], state_j->v[2]}};
    imu_factor_t imu_base_corr;
    memcpy(&imu_base_corr, &imu_measurement->base, sizeof(imu_factor_t));
    memcpy(imu_base_corr.dR, dR_corr, 9 * sizeof(float));
    imu_base_corr.dp = (point3D_float_t){dp_corr[0], dp_corr[1], dp_corr[2]};
    imu_base_corr.dv = (point3D_float_t){dv_corr[0], dv_corr[1], dv_corr[2]};
    
    compute_imu_error_and_jacobian(e_imu, J_imu_base, &state_i_9dof, &state_j_9dof, &imu_base_corr, g_world, work_memory);

    // --- Compute Full 15x30 Jacobian ---
    memset(J_imu, 0, ERROR_SIZE_IMU * STATE_SIZE_BIAS * 2 * sizeof(float));
    
    // Copy the original 9x9 blocks for [r, p, v] into the new 15x15 blocks
    for(int r_block=0; r_block<3; ++r_block) { // e_r, e_p, e_v blocks
        for (int r=0; r<3; ++r) {
            // State i
            memcpy(&J_imu[(r_block*3+r)*(STATE_SIZE_BIAS*2) + 0], &J_imu_base[(r_block*3+r)*(9*2) + 0], 9 * sizeof(float));
            // State j
            memcpy(&J_imu[(r_block*3+r)*(STATE_SIZE_BIAS*2) + STATE_SIZE_BIAS], &J_imu_base[(r_block*3+r)*(9*2) + 9], 9 * sizeof(float));
        }
    }
    
    // --- New Jacobian blocks w.r.t biases ---
    float* R_i_T = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* R_i = allocate_work_memory(&work_memory, 9 * sizeof(float));
    rodriguesToMatrix(state_i->r, R_i);
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) R_i_T[r*3+c] = R_i[c*3+r];

    float* Jr_inv = allocate_work_memory(&work_memory, 9 * sizeof(float));
    compute_right_jacobian_so3_inverse(&e_imu[0], Jr_inv);

    // Partial derivatives of the error w.r.t. bias_i
    // The pre-integration Jacobians (J_r_bg, J_p_bg, etc.) are defined as:
    //   Δp = Δp_nom + J_p_bg·δb_g + J_p_ba·δb_a + ...
    // So ∂(Δmeas)/∂b_i = J_meas_bi (positive)
    const int b_i_col_offset = 9; // Start of bias columns for state i

    // ∂e_r / ∂b_gi = Jr_inv * J_r_bg
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
        float sum = 0;
        for(int k=0; k<3; ++k) sum += Jr_inv[r*3+k] * imu_measurement->J_r_bg[k*3+c];
        J_imu[r*(STATE_SIZE_BIAS*2) + b_i_col_offset + c] = sum;
    }
    
    // ∂e_p / ∂b_gi = R_i_T * J_p_bg
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
        float sum = 0;
        for(int k=0; k<3; ++k) sum += R_i_T[r*3+k] * imu_measurement->J_p_bg[k*3+c];
        J_imu[(3+r)*(STATE_SIZE_BIAS*2) + b_i_col_offset + c] = sum;
    }

    // ∂e_p / ∂b_ai = R_i_T * J_p_ba
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
        float sum = 0;
        for(int k=0; k<3; ++k) sum += R_i_T[r*3+k] * imu_measurement->J_p_ba[k*3+c];
        J_imu[(3+r)*(STATE_SIZE_BIAS*2) + b_i_col_offset + 3 + c] = sum;
    }
    
    // ∂e_v / ∂b_gi = R_i_T * J_v_bg
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
        float sum = 0;
        for(int k=0; k<3; ++k) sum += R_i_T[r*3+k] * imu_measurement->J_v_bg[k*3+c];
        J_imu[(6+r)*(STATE_SIZE_BIAS*2) + b_i_col_offset + c] = sum;
    }

    // ∂e_v / ∂b_ai = R_i_T * J_v_ba
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
        float sum = 0;
        for(int k=0; k<3; ++k) sum += R_i_T[r*3+k] * imu_measurement->J_v_ba[k*3+c];
        J_imu[(6+r)*(STATE_SIZE_BIAS*2) + b_i_col_offset + 3 + c] = sum;
    }
    
    // The error does not depend on bias_j, so those Jacobians are zero.
}


/**
 * @brief Sequentially optimizes for IMU biases after poses/velocities are fixed.
 *
 * This function forms and solves a small linear system (H_b * dx = b_b) where
 * the only variables are the updates to the accelerometer and gyroscope biases
 * for each keyframe in the window.
 *
 * @param biases A 2D array of current bias estimates [num_poses][6], which will be updated in-place.
 * @param optimized_poses The array of poses from the main optimizer.
 * @param optimized_velocities The array of velocities from the main optimizer.
 * @param imu_factors_bias The pre-integrated measurements with bias Jacobians.
 * @param num_poses The number of keyframes in the optimization window.
 * @param g The gravity vector.
 * @param work_memory Workspace for temporary allocations.
 * @return 1 on success, 0 on failure.
 */
int optimize_biases_sequentially(
    float biases[MAX_KEYFRAMES][BIAS_SIZE],
    const pose_t* optimized_poses,
    const point3D_float_t* optimized_velocities,
    const imu_factor_with_bias_t* imu_factors_bias,
    int num_poses,
    int total_poses,
    const float* g,
    work_memory_t work_memory
) {
    if (num_poses < 2) return 1; // Nothing to optimize

    LOG_DEBUG("Calling new bias estimation function, for num poses %d\n", num_poses);

    const int bias_system_dim = num_poses * BIAS_SIZE;

    // Allocate the augmented system matrix [H_b | b_b] for the bias-only problem
    float* Hb = allocate_work_memory(&work_memory, bias_system_dim * (bias_system_dim + 1) * sizeof(float));
    if (!Hb) return 0;
    memset(Hb, 0, bias_system_dim * (bias_system_dim + 1) * sizeof(float));

    float* e_imu = allocate_work_memory(&work_memory, 9*sizeof(float));
    float* J_full = allocate_work_memory(&work_memory, 9 * STATE_SIZE_BIAS * 2 * sizeof(float));
    float* J_b = allocate_work_memory(&work_memory, 9 * BIAS_SIZE * 2 * sizeof(float));
    CameraState15Dof* state_i = allocate_work_memory(&work_memory, sizeof(CameraState15Dof));
    CameraState15Dof* state_j = allocate_work_memory(&work_memory, sizeof(CameraState15Dof));

    // --- A. Add IMU Factor Constraints ---
    uint8_t ignore_imu_factor_idx = (Max(total_poses-MAX_KEYFRAMES, 0)) % MAX_KEYFRAMES;
    for(int curr_idx = 0; curr_idx < num_poses; ++curr_idx) {
        if (curr_idx == ignore_imu_factor_idx) {
            continue;
        }
        int prev_idx = (MAX_KEYFRAMES + curr_idx - 1) % MAX_KEYFRAMES;
        const imu_factor_with_bias_t* factor = &imu_factors_bias[curr_idx];

        // 1. Reconstruct the full 15-DoF states from the fixed inputs
        float R_data[9], t_vec[3];
        matrix_2D_t T_mat = { (float*)&optimized_poses[prev_idx], {4,4,0,0} };
        matrix_2D_t R_mat = { R_data, {3,3,0,0} };

        // State i
        memset(state_i, 0 , sizeof(CameraState15Dof));
        rotationMatrixOfTransformation(&T_mat, &R_mat);
        translationVectorOfTransformation(&T_mat, t_vec);
        matrixToRodrigues(R_data, state_i->r);
        // t_vec = -R * p_world  =>  p_world = -R^T * t_vec
        for (int ii = 0; ii < 3; ++ii)
            state_i->p[ii] = -(R_data[0*3+ii]*t_vec[0] + R_data[1*3+ii]*t_vec[1] + R_data[2*3+ii]*t_vec[2]);
        memcpy(state_i->v, &optimized_velocities[prev_idx], 3 * sizeof(float));
        memcpy(state_i->b_g, &biases[prev_idx][0], 3 * sizeof(float));  // Gyro bias only
        memcpy(state_i->b_a, &biases[prev_idx][3], 3 * sizeof(float));  // Accel bias only

        // State j
        memset(state_j, 0 , sizeof(CameraState15Dof));
        T_mat.data = (float*)&optimized_poses[curr_idx];
        rotationMatrixOfTransformation(&T_mat, &R_mat);
        translationVectorOfTransformation(&T_mat, t_vec);
        matrixToRodrigues(R_data, state_j->r);
        // t_vec = -R * p_world  =>  p_world = -R^T * t_vec
        for (int ii = 0; ii < 3; ++ii)
            state_j->p[ii] = -(R_data[0*3+ii]*t_vec[0] + R_data[1*3+ii]*t_vec[1] + R_data[2*3+ii]*t_vec[2]);
        memcpy(state_j->v, &optimized_velocities[curr_idx], 3 * sizeof(float));
        memcpy(state_j->b_g, &biases[curr_idx][0], 3 * sizeof(float));  // Gyro bias only
        memcpy(state_j->b_a, &biases[curr_idx][3], 3 * sizeof(float));  // Accel bias only

        // 2. Calculate the error and the full 9x30 Jacobian
        memset(e_imu, 0, 9*sizeof(float));
        memset(J_full, 0, 9 * STATE_SIZE_BIAS * 2 * sizeof(float));
        compute_imu_bias_error_and_jacobian(e_imu, J_full, state_i, state_j, factor, g, work_memory);

        // 3. Extract ONLY the 9x12 Jacobian for biases: J_b = [∂e/∂b_i, ∂e/∂b_j]
        memset(J_b, 0, 9 * BIAS_SIZE * 2 * sizeof(float));
        for (int r = 0; r < 9; ++r) {
            memcpy(&J_b[r * 12 + 0], &J_full[r * 30 + 9],  6 * sizeof(float)); // Columns for b_i
            memcpy(&J_b[r * 12 + 6], &J_full[r * 30 + 24], 6 * sizeof(float)); // Columns for b_j
        }

        // 4. Get the information matrix for the IMU factor
        float omega_diag[9];
        float dt = factor->base.dt;
        float inv_var_rot = 1.0f / (dt * GYRO_NOISE_DENSITY * GYRO_NOISE_DENSITY);
        // Velocity variance: sensor noise (σ_a²·dt) + gravity uncertainty (σ_g²·dt²)
        float var_vel = dt * ACCEL_NOISE_DENSITY * ACCEL_NOISE_DENSITY
                      + dt * dt * GRAVITY_UNCERTAINTY * GRAVITY_UNCERTAINTY;
        float inv_var_vel = 1.0f / var_vel;
        // Position variance: double-integrated sensor noise (σ_a²·dt³/3) + gravity (σ_g²·dt⁴/4)
        float var_pos = (dt * dt * dt / 3.0f) * ACCEL_NOISE_DENSITY * ACCEL_NOISE_DENSITY
                      + (dt * dt * dt * dt / 4.0f) * GRAVITY_UNCERTAINTY * GRAVITY_UNCERTAINTY;
        float inv_var_pos = 1.0f / var_pos;
        for(int i=0; i<3; ++i) omega_diag[i] = inv_var_rot;
        for(int i=0; i<3; ++i) omega_diag[i+3] = inv_var_pos;
        for(int i=0; i<3; ++i) omega_diag[i+6] = inv_var_vel;

        // 5. Add this factor's contribution to the H_b system: H += J_b^T * Omega * J_b; b += -J_b^T * Omega * e
        int offsets[] = {prev_idx * BIAS_SIZE, curr_idx * BIAS_SIZE};
        for (int br = 0; br < 2; ++br) { // block row
            for (int bc = 0; bc < 2; ++bc) { // block col
                for (int r = 0; r < BIAS_SIZE; ++r) { // row within block
                    for (int c = 0; c < BIAS_SIZE; ++c) { // col within block
                        float h_val = 0;
                        for (int k = 0; k < 9; ++k) { // sum over error dimension
                            h_val += J_b[k*12 + br*6+r] * omega_diag[k] * J_b[k*12 + bc*6+c];
                        }
                        Hb[(offsets[br]+r) * (bias_system_dim+1) + (offsets[bc]+c)] += h_val;
                    }
                }
            }
            // Update b vector part: b_br += -J_br^T * Omega * e
            for (int r = 0; r < BIAS_SIZE; ++r) { // row within block
                float b_val = 0;
                for (int k = 0; k < 9; ++k) { // sum over error dimension
                    b_val += J_b[k*12 + br*6+r] * omega_diag[k] * e_imu[k];
                }
                Hb[(offsets[br]+r) * (bias_system_dim+1) + bias_system_dim] -= b_val;
            }
        }
    }

    // --- B. Add Bias Random Walk Constraints ---
    for (int curr_idx = 1; curr_idx < num_poses; ++curr_idx) {
        int prev_idx = curr_idx - 1;
        float dt = imu_factors_bias[curr_idx].base.dt;

        float e_rw[BIAS_SIZE];
        for(int i=0; i<BIAS_SIZE; ++i) e_rw[i] = biases[curr_idx][i] - biases[prev_idx][i];

        float omega_rw_diag[BIAS_SIZE];
        float inv_var_bg = 1.0f / (dt * GYRO_BIAS_RANDOM_WALK_NOISE * GYRO_BIAS_RANDOM_WALK_NOISE);
        float inv_var_ba = 1.0f / (dt * ACCEL_BIAS_RANDOM_WALK_NOISE * ACCEL_BIAS_RANDOM_WALK_NOISE);
        for(int i=0; i<3; ++i) omega_rw_diag[i] = inv_var_bg;
        for(int i=3; i<6; ++i) omega_rw_diag[i] = inv_var_ba;

        // Jacobian is J_rw = [-I, I]. H_contrib = J_rw^T * Omega * J_rw
        int off_prev = prev_idx * BIAS_SIZE;
        int off_curr = curr_idx * BIAS_SIZE;
        for (int i = 0; i < BIAS_SIZE; ++i) {
            // Add Omega to diagonal blocks
            Hb[(off_prev+i)*(bias_system_dim+1) + (off_prev+i)] += omega_rw_diag[i];
            Hb[(off_curr+i)*(bias_system_dim+1) + (off_curr+i)] += omega_rw_diag[i];
            // Subtract Omega from off-diagonal blocks
            Hb[(off_prev+i)*(bias_system_dim+1) + (off_curr+i)] -= omega_rw_diag[i];
            Hb[(off_curr+i)*(bias_system_dim+1) + (off_prev+i)] -= omega_rw_diag[i];

            // Update b vector part: b += -J_rw^T * Omega * e_rw
            float b_update = omega_rw_diag[i] * e_rw[i];
            Hb[(off_prev+i)*(bias_system_dim+1) + bias_system_dim] += b_update;
            Hb[(off_curr+i)*(bias_system_dim+1) + bias_system_dim] -= b_update;
        }
    }

    // --- C. Solve the System ---
    float lambda = LM_DAMPING;
    for (int i = 0; i < bias_system_dim; ++i) {
        Hb[i * (bias_system_dim + 1) + i] += lambda;
    }

    float* delta_b = allocate_work_memory(&work_memory, bias_system_dim * sizeof(float));
    if (!delta_b) return 0;

    if (solve_cholesky_system(Hb, delta_b, bias_system_dim, work_memory)) {
        // --- D. Update the biases ---
        for (int i = 0; i < num_poses; ++i) {
            for (int j = 0; j < BIAS_SIZE; ++j) {
                biases[i][j] -= delta_b[i * BIAS_SIZE + j];
            }
        }
        return 1;
    }

    // Solver failed
    return 0;
}