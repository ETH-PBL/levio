// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "imu_optimization.h"

#include "math.h"

#define Min(a, b)       (((a)<(b))?(a):(b))
#define Max(a, b)       (((a)>(b))?(a):(b))

// Create a skew-symmetric matrix from a 3-vector
static void skew_symmetric(float v[3], float A[9]) {
    A[0]=0;
    A[1]=-v[2];
    A[2]=v[1];
    A[3]=v[2];
    A[4]=0;
    A[5]=-v[0];
    A[6]=-v[1];
    A[7]=v[0];
    A[8]=0;
}

// Multiply a 3x3 matrix by a 3-vector
void mat33_vec3_mult(const float M[9], const float v[3], float result[3]) {
    result[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2];
    result[1] = M[3]*v[0] + M[4]*v[1] + M[5]*v[2];
    result[2] = M[6]*v[0] + M[7]*v[1] + M[8]*v[2];
}

/**
 * @brief Recover world-frame position from a T_cw pose matrix.
 *
 * T_cw stores:  t_cw = -C * p_world  where C = R_bw (world->body).
 * Therefore:    p_world = -C^T * t_cw
 *
 * @param C    Row-major 3x3 rotation (world->body), from rotationMatrixOfTransformation.
 * @param t    3-vector translation (= -C * p_world), from translationVectorOfTransformation.
 * @param p    Output world-frame position.
 */
static void recover_world_position(const float C[9], const float t[3], float p[3]) {
    p[0] = -(C[0]*t[0] + C[3]*t[1] + C[6]*t[2]);
    p[1] = -(C[1]*t[0] + C[4]*t[1] + C[7]*t[2]);
    p[2] = -(C[2]*t[0] + C[5]*t[1] + C[8]*t[2]);
}

// --- Core IMU Jacobian and System Composition ---

/**
 * @brief Computes the inverse of the right Jacobian of SO(3).
 *
 * Jr(phi)^-1 = I + 1/2*[phi]_x + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta)))*[phi]_x^2
 *
 * @param phi     3-element Rodrigues vector.
 * @param Jr_inv  Output 3x3 inverse Jacobian matrix.
 */
void compute_right_jacobian_so3_inverse(const float phi[3], float Jr_inv[9]) {
    const float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
    const float theta = sqrtf(theta_sq);

    float phi_x[9];
    skew_symmetric(phi, phi_x);

    float phi_x_sq[9];
    for(int r=0; r<3; ++r) {
        for(int c=0; c<3; ++c) {
            phi_x_sq[r*3+c] = phi_x[r*3+0]*phi_x[0*3+c] + phi_x[r*3+1]*phi_x[1*3+c] + phi_x[r*3+2]*phi_x[2*3+c];
        }
    }

    float A, B;
    if (theta < 1e-4f) {
        A = 0.5f;
        B = (1.0f / 12.0f) - (theta_sq / 720.0f);
    } else {
        A = 0.5f;
        B = (1.0f / theta_sq) - (1.0f + cosf(theta)) / (2.0f * theta * sinf(theta));
    }

    for (int i = 0; i < 9; ++i) {
        Jr_inv[i] = (i%4 == 0) ? 1.0f : 0.0f;
        Jr_inv[i] += A * phi_x[i];
        Jr_inv[i] += B * phi_x_sq[i];
    }
}

/**
 * @brief Computes the IMU error vector and its FULL analytical Jacobian.
 *
 * States are expressed in physical (body->world / world-frame) convention:
 *   state.r  = Rodrigues vector of R_i (body->world)
 *   state.p  = position in world frame
 *   state.v  = velocity in world frame
 *
 * The caller (add_imu_factors) is responsible for converting from the T_cw
 * pose storage convention before calling this function.
 *
 * @param e_imu  Output 9x1 error vector [e_r(3), e_p(3), e_v(3)].
 * @param J_imu  Output 9x18 Jacobian [∂e/∂x_i | ∂e/∂x_j], w.r.t. [r_bw, p_world, v_world].
 */
void compute_imu_error_and_jacobian(
    float* e_imu, float* J_imu,
    const CameraState9Dof* state_i,
    const CameraState9Dof* state_j,
    const imu_factor_t* imu_measurement,
    const float g_world[3],
    work_memory_t work_memory)
{
    float* R_i   = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* R_j   = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* R_i_T = allocate_work_memory(&work_memory, 9 * sizeof(float));
    rodriguesToMatrix(state_i->r, R_i);
    rodriguesToMatrix(state_j->r, R_j);
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) R_i_T[r*3+c] = R_i[c*3+r];

    const float* dR = imu_measurement->dR;
    const float dp[3] = {imu_measurement->dp.x, imu_measurement->dp.y, imu_measurement->dp.z};
    const float dv[3] = {imu_measurement->dv.x, imu_measurement->dv.y, imu_measurement->dv.z};
    const float dt = imu_measurement->dt;

    // --- 1. COMPUTE ERROR VECTOR (9x1) ---

    // Rotation error: e_R = log(ΔR^T * R_i^T * R_j)
    float* R_err_mat = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* temp_mat  = allocate_work_memory(&work_memory, 9 * sizeof(float));
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) temp_mat[r*3+c] = R_i_T[r*3+0]*R_j[0*3+c] + R_i_T[r*3+1]*R_j[1*3+c] + R_i_T[r*3+2]*R_j[2*3+c];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) R_err_mat[r*3+c] = dR[0*3+r]*temp_mat[0*3+c] + dR[1*3+r]*temp_mat[1*3+c] + dR[2*3+r]*temp_mat[2*3+c];
    matrixToRodrigues(R_err_mat, &e_imu[0]);

    // Position error: e_p = R_i^T * (p_j - p_i - v_i*dt - 0.5*g*dt^2) - Δp
    float pos_world_term[3];
    pos_world_term[0] = state_j->p[0] - state_i->p[0] - state_i->v[0]*dt - 0.5f*g_world[0]*dt*dt;
    pos_world_term[1] = state_j->p[1] - state_i->p[1] - state_i->v[1]*dt - 0.5f*g_world[1]*dt*dt;
    pos_world_term[2] = state_j->p[2] - state_i->p[2] - state_i->v[2]*dt - 0.5f*g_world[2]*dt*dt;
    float pos_body_term[3];
    mat33_vec3_mult(R_i_T, pos_world_term, pos_body_term);
    e_imu[3] = pos_body_term[0] - dp[0];
    e_imu[4] = pos_body_term[1] - dp[1];
    e_imu[5] = pos_body_term[2] - dp[2];

    // Velocity error: e_v = R_i^T * (v_j - v_i - g*dt) - Δv
    float vel_world_term[3];
    vel_world_term[0] = state_j->v[0] - state_i->v[0] - g_world[0]*dt;
    vel_world_term[1] = state_j->v[1] - state_i->v[1] - g_world[1]*dt;
    vel_world_term[2] = state_j->v[2] - state_i->v[2] - g_world[2]*dt;
    float vel_body_term[3];
    mat33_vec3_mult(R_i_T, vel_world_term, vel_body_term);
    e_imu[6] = vel_body_term[0] - dv[0];
    e_imu[7] = vel_body_term[1] - dv[1];
    e_imu[8] = vel_body_term[2] - dv[2];

    // --- 2. COMPUTE JACOBIAN (9x18), w.r.t. [r_bw, p_world, v_world] for i then j ---
    memset(J_imu, 0, ERROR_SIZE * STATE_SIZE * 2 * sizeof(float));
    float* Jr_inv = allocate_work_memory(&work_memory, 9 * sizeof(float));
    compute_right_jacobian_so3_inverse(&e_imu[0], Jr_inv);

    // Block J_i
    float* R_j_T_R_i = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* dER_dRi   = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float R_j_T[9];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) R_j_T[r*3+c] = R_j[c*3+r];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) R_j_T_R_i[r*3+c] = R_j_T[r*3+0]*R_i[0*3+c] + R_j_T[r*3+1]*R_i[1*3+c] + R_j_T[r*3+2]*R_i[2*3+c];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) dER_dRi[r*3+c] = -(Jr_inv[r*3+0]*R_j_T_R_i[0*3+c] + Jr_inv[r*3+1]*R_j_T_R_i[1*3+c] + Jr_inv[r*3+2]*R_j_T_R_i[2*3+c]);
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[r*(STATE_SIZE*2) + c] = dER_dRi[r*3+c];

    float* dEP_dRi = allocate_work_memory(&work_memory, 9 * sizeof(float));
    skew_symmetric(pos_body_term, dEP_dRi);
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[(3+r)*(STATE_SIZE*2) + c] = dEP_dRi[r*3+c];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[(3+r)*(STATE_SIZE*2) + (3+c)] = -R_i_T[r*3+c];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[(3+r)*(STATE_SIZE*2) + (6+c)] = -R_i_T[r*3+c] * dt;

    float* dEV_dRi = allocate_work_memory(&work_memory, 9 * sizeof(float));
    skew_symmetric(vel_body_term, dEV_dRi);
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[(6+r)*(STATE_SIZE*2) + c] = dEV_dRi[r*3+c];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[(6+r)*(STATE_SIZE*2) + (6+c)] = -R_i_T[r*3+c];

    // Block J_j
    const int j_col_offset = STATE_SIZE;
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[r*(STATE_SIZE*2) + (j_col_offset+c)] = Jr_inv[r*3+c];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[(3+r)*(STATE_SIZE*2) + j_col_offset+3+c] = R_i_T[r*3+c];
    for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) J_imu[(6+r)*(STATE_SIZE*2) + j_col_offset+6+c] = R_i_T[r*3+c];
}

/**
 * @brief Composes the S and b system matrices from a single Jacobian and error.
 *
 * Sb uses the (-H, -b) convention to match the VO pipeline accumulation.
 * Caller must negate e before calling if the sign convention requires it.
 */
void compose_system_from_jacobian(
    float* Sb, int num_poses,
    int pose_idx_i, int pose_idx_j,
    const float* e, const float* J, const float* Omega,
    work_memory_t work_memory)
{
    float* Jt_Omega = allocate_work_memory(&work_memory, STATE_SIZE*2 * ERROR_SIZE * sizeof(float));
    for (int r = 0; r < STATE_SIZE * 2; ++r) {
        for (int c = 0; c < ERROR_SIZE; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < ERROR_SIZE; ++k) sum += J[k*(STATE_SIZE*2) + r] * Omega[k*ERROR_SIZE + c];
            Jt_Omega[r*ERROR_SIZE + c] = sum;
        }
    }

    float* H_contrib = allocate_work_memory(&work_memory, STATE_SIZE*2 * STATE_SIZE*2 * sizeof(float));
    for (int r = 0; r < STATE_SIZE*2; ++r) {
        for (int c = 0; c < STATE_SIZE*2; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < ERROR_SIZE; ++k) sum += Jt_Omega[r*ERROR_SIZE + k] * J[k*(STATE_SIZE*2) + c];
            H_contrib[r*(STATE_SIZE*2) + c] = sum;
        }
    }

    float b_contrib[STATE_SIZE*2];
    for (int r = 0; r < STATE_SIZE*2; ++r) {
        float sum = 0.0f;
        for (int k = 0; k < ERROR_SIZE; ++k) sum += Jt_Omega[r*ERROR_SIZE + k] * e[k];
        b_contrib[r] = sum;
    }

    const int global_dim = num_poses * STATE_SIZE;
    const int offset_i = pose_idx_i * STATE_SIZE;
    const int offset_j = pose_idx_j * STATE_SIZE;
    const int pose_indices[] = {offset_i, offset_j};

    for (int br=0; br<2; ++br) for (int bc=0; bc<2; ++bc) {
        for (int r=0; r<STATE_SIZE; ++r) for (int c=0; c<STATE_SIZE; ++c) {
            Sb[(pose_indices[br]+r)*(global_dim+1) + (pose_indices[bc]+c)] -= H_contrib[(br*STATE_SIZE+r)*(STATE_SIZE*2) + (bc*STATE_SIZE+c)];
        }
    }

    for(int i=0; i<STATE_SIZE; ++i) {
        Sb[(offset_i+i)*(global_dim+1) + global_dim] -= b_contrib[i];
        Sb[(offset_j+i)*(global_dim+1) + global_dim] -= b_contrib[STATE_SIZE+i];
    }
}

/**
 * @brief Remaps IMU Jacobian columns from physical [r_bw, p_world, v_world]
 *        to the T_cw storage parameterization [r_cw, t_cw, v_world].
 *
 * Chain rule:
 *   ∂e/∂r_cw   = ∂e/∂r_bw * (-I)      (r_bw = -r_cw)
 *   ∂e/∂t_cw   = ∂e/∂p_world * (-C^T)  (p_world = -C^T * t_cw)
 *   ∂e/∂v_world unchanged
 *
 * Applied in-place for both state i (cols 0-8) and state j (cols 9-17).
 *
 * @param J_imu  9x18 Jacobian, modified in place.
 * @param C_i    R_bw for state i (world->body, row-major 3x3).
 * @param C_j    R_bw for state j (world->body, row-major 3x3).
 */
static void remap_jacobian_to_tcw(float* J_imu,
                                   const float C_i[9],
                                   const float C_j[9])
{
    const float* C[2] = {C_i, C_j};

    for (int blk = 0; blk < 2; ++blk) {
        int col_base = blk * STATE_SIZE;
        const float* Cb = C[blk];

        for (int row = 0; row < ERROR_SIZE; ++row) {
            // Rotation cols: negate
            for (int c = 0; c < 3; ++c)
                J_imu[row*(STATE_SIZE*2) + col_base+c] *= -1.0f;

            // Position cols: ∂e/∂t_cw[k] = -sum_m jp[m] * Cb[k*3+m]
            float jp[3] = {
                J_imu[row*(STATE_SIZE*2) + col_base+3],
                J_imu[row*(STATE_SIZE*2) + col_base+4],
                J_imu[row*(STATE_SIZE*2) + col_base+5]
            };
            for (int c = 0; c < 3; ++c)
                J_imu[row*(STATE_SIZE*2) + col_base+3+c] =
                    -(Cb[c*3+0]*jp[0] + Cb[c*3+1]*jp[1] + Cb[c*3+2]*jp[2]);

            // Velocity cols: unchanged
        }
    }
}

/**
 * @brief Adds a kinematic consistency prior: v_j ≈ (p_j - p_i) / dt
 *
 * Constrains v_j to be consistent with the finite-difference position
 * displacement, resolving gauge freedom in absolute velocity.
 *
 * Error:    e = v_j - (p_j - p_i) / dt                     (3x1)
 *
 * Jacobians in physical coords:
 *   ∂e/∂v_j    =  I
 *   ∂e/∂p_i    = +I / dt
 *   ∂e/∂p_j    = -I / dt
 *
 * Remapped to T_cw DOFs via ∂e/∂t_cw = ∂e/∂p_world * (-C^T):
 *   ∂e/∂t_i    = -C_i^T / dt     (J_ti[er][tc] = -C_i[tc*3+er] / dt)
 *   ∂e/∂t_j    = +C_j^T / dt     (J_tj[er][tc] = +C_j[tc*3+er] / dt)
 *   ∂e/∂v_j    =  I              (unchanged, v is already world frame)
 *
 * All contributions are subtracted into Sb to match the (-H, -b) convention.
 *
 * @param Sb         Augmented system matrix, size (N*9) x (N*9+1).
 * @param poses      Current pose estimates (T_cw convention).
 * @param velocities Current velocity estimates (world frame).
 * @param imu_factors IMU factors (for dt between consecutive keyframes).
 * @param num_poses  Number of poses in the window.
 * @param total_poses Total keyframes (for index wrapping).
 * @param lambda     Regularization strength (units: 1/(m/s)^2).
 */
void add_kinematic_velocity_prior(float* Sb,
                                  const pose_t* poses,
                                  const point3D_float_t* velocities,
                                  const imu_factor_with_bias_t* imu_factors,
                                  uint16_t num_poses,
                                  uint16_t total_poses,
                                  float lambda,
                                  work_memory_t work_memory)
{
    const int global_dim = num_poses * STATE_SIZE;
    uint8_t ignore_idx = (Max(total_poses - MAX_KEYFRAMES, 0)) % MAX_KEYFRAMES;

    float* R_data = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* t_vec = allocate_work_memory(&work_memory, 3 * sizeof(float));
    float* C_i = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* C_j = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* J_ti = allocate_work_memory(&work_memory, 9 * sizeof(float));
    float* J_tj = allocate_work_memory(&work_memory, 9 * sizeof(float));
    matrix_2D_t R_mat = {R_data, {3,3,0,0}};

    for (int curr_idx = 0; curr_idx < num_poses; ++curr_idx) {
        if (curr_idx == ignore_idx) continue;

        int prev_idx = (MAX_KEYFRAMES + curr_idx - 1) % MAX_KEYFRAMES;
        float dt = imu_factors[curr_idx].base.dt;
        if (dt < 1e-6f) continue;

        // Extract C_i and p_world_i from T_cw pose
        float p_i[3];
        matrix_2D_t prev_pose_mat = {(float*) &poses[prev_idx], {4,4,0,0}};
        rotationMatrixOfTransformation(&prev_pose_mat, &R_mat);
        translationVectorOfTransformation(&prev_pose_mat, t_vec);
        memcpy(C_i, R_data, 9 * sizeof(float));
        recover_world_position(C_i, t_vec, p_i);

        // Extract C_j and p_world_j from T_cw pose
        float p_j[3];
        matrix_2D_t curr_pose_mat = {(float*) &poses[curr_idx], {4,4,0,0}};
        rotationMatrixOfTransformation(&curr_pose_mat, &R_mat);
        translationVectorOfTransformation(&curr_pose_mat, t_vec);
        memcpy(C_j, R_data, 9 * sizeof(float));
        recover_world_position(C_j, t_vec, p_j);

        // Error: e = v_j - (p_j - p_i) / dt
        const point3D_float_t* vj = &velocities[curr_idx];
        float e[3] = {
            vj->x - (p_j[0] - p_i[0]) / dt,
            vj->y - (p_j[1] - p_i[1]) / dt,
            vj->z - (p_j[2] - p_i[2]) / dt
        };

        // DOF offsets: t_cw is [3,4,5], v_world is [6,7,8] within each STATE_SIZE block
        const int t_i_offset = prev_idx * STATE_SIZE + 3;
        const int t_j_offset = curr_idx * STATE_SIZE + 3;
        const int v_j_offset = curr_idx * STATE_SIZE + 6;

        // Remapped Jacobian blocks (3x3 each, row=error index, col=t_cw index):
        //   J_ti[er][tc] = -C_i[tc*3+er] / dt
        //   J_tj[er][tc] = +C_j[tc*3+er] / dt
        for (int er = 0; er < 3; ++er) {
            for (int tc = 0; tc < 3; ++tc) {
                J_ti[er*3+tc] = -C_i[tc*3+er] / dt;
                J_tj[er*3+tc] = +C_j[tc*3+er] / dt;
            }
        }

        // Accumulate into Sb using (-H, -b) convention: Sb -= lambda * contribution
        for (int k = 0; k < 3; ++k) {

            // v_j diagonal: (∂e/∂v_j = I) → H_vv = I
            Sb[(v_j_offset+k)*(global_dim+1) + (v_j_offset+k)] -= lambda;

            // t_i and t_j self-blocks: H_aa = J_a^T * J_a
            for (int c = 0; c < 3; ++c) {
                float hii = 0.0f, hjj = 0.0f;
                for (int m = 0; m < 3; ++m) {
                    hii += J_ti[m*3+k] * J_ti[m*3+c];
                    hjj += J_tj[m*3+k] * J_tj[m*3+c];
                }
                Sb[(t_i_offset+k)*(global_dim+1) + (t_i_offset+c)] -= lambda * hii;
                Sb[(t_j_offset+k)*(global_dim+1) + (t_j_offset+c)] -= lambda * hjj;
            }

            // v_j <-> t_i cross: H = J_vj^T * J_ti = I^T * J_ti = J_ti
            for (int c = 0; c < 3; ++c) {
                float h = J_ti[k*3+c];
                Sb[(v_j_offset+k)*(global_dim+1) + (t_i_offset+c)] -= lambda * h;
                Sb[(t_i_offset+c)*(global_dim+1) + (v_j_offset+k)] -= lambda * h;
            }

            // v_j <-> t_j cross: H = J_vj^T * J_tj = J_tj
            for (int c = 0; c < 3; ++c) {
                float h = J_tj[k*3+c];
                Sb[(v_j_offset+k)*(global_dim+1) + (t_j_offset+c)] -= lambda * h;
                Sb[(t_j_offset+c)*(global_dim+1) + (v_j_offset+k)] -= lambda * h;
            }

            // t_i <-> t_j cross: H = J_ti^T * J_tj
            for (int c = 0; c < 3; ++c) {
                float h = 0.0f;
                for (int m = 0; m < 3; ++m) h += J_ti[m*3+k] * J_tj[m*3+c];
                Sb[(t_i_offset+k)*(global_dim+1) + (t_j_offset+c)] -= lambda * h;
                Sb[(t_j_offset+c)*(global_dim+1) + (t_i_offset+k)] -= lambda * h;
            }

            // b vector: b_a = J_a^T * e
            float bv   = e[k];
            float bt_i = 0.0f, bt_j = 0.0f;
            for (int m = 0; m < 3; ++m) {
                bt_i += J_ti[m*3+k] * e[m];
                bt_j += J_tj[m*3+k] * e[m];
            }
            Sb[(v_j_offset+k)*(global_dim+1) + global_dim] -= lambda * bv;
            Sb[(t_i_offset+k)*(global_dim+1) + global_dim] -= lambda * bt_i;
            Sb[(t_j_offset+k)*(global_dim+1) + global_dim] -= lambda * bt_j;
        }
    }
}


/**
 * @brief Adds IMU factor constraints to the optimization problem.
 *
 * Pose convention: poses store T_cw (world->camera):
 *   C = rotationMatrixOfTransformation  →  R_bw (world->body)
 *   t = translationVectorOfTransformation  →  -C * p_world
 *
 * The IMU error/Jacobian is computed in physical coordinates [r_bw, p_world, v_world],
 * then the Jacobian is remapped to the T_cw parameterization [r_cw, t_cw, v_world]
 * via remap_jacobian_to_tcw() before accumulating into Sb.
 *
 * Velocities are stored and optimized in world frame throughout.
 */
float add_imu_factors(pose_t* poses, uint16_t num_poses,
                      uint16_t total_poses,
                      point3D_float_t* velocities,
                      imu_factor_with_bias_t* imu_factors,
                      float* Sb,
                      float* g,
                      work_memory_t work_memory)
{
    float* e_imu  = allocate_work_memory(&work_memory, ERROR_SIZE * sizeof(float));
    float* J_imu  = allocate_work_memory(&work_memory, ERROR_SIZE * STATE_SIZE * 2 * sizeof(float));

    float* t      = allocate_work_memory(&work_memory, 3 * sizeof(float));
    float* r      = allocate_work_memory(&work_memory, 3 * sizeof(float));
    float* R_data = allocate_work_memory(&work_memory, 9 * sizeof(float));
    matrix_2D_t R = {R_data, {3,3,0,0}};

    float* omega  = allocate_work_memory(&work_memory, ERROR_SIZE * ERROR_SIZE * sizeof(float));

    float total_error = 0.0f;
    uint8_t ignore_imu_factor_idx = (Max(total_poses-MAX_KEYFRAMES, 0)) % MAX_KEYFRAMES;

    for(int curr_idx = 0; curr_idx < num_poses; ++curr_idx) {
        if (curr_idx == ignore_imu_factor_idx) continue;

        int prev_idx = (MAX_KEYFRAMES + curr_idx - 1) % MAX_KEYFRAMES;
        float dt = imu_factors[curr_idx].base.dt;

        float info_rot = IMU_WEIGHT_SCALE / (dt * GYRO_NOISE_DENSITY * GYRO_NOISE_DENSITY);
        // Velocity variance: sensor noise (σ_a²·dt) + gravity uncertainty (σ_g²·dt²)
        float var_vel = dt * ACCEL_NOISE_DENSITY * ACCEL_NOISE_DENSITY
                      + dt * dt * GRAVITY_UNCERTAINTY * GRAVITY_UNCERTAINTY;
        float info_vel = IMU_WEIGHT_SCALE / var_vel;
        // Position variance: double-integrated sensor noise (σ_a²·dt³/3) + gravity (σ_g²·dt⁴/4)
        float var_pos = (dt * dt * dt / 3.0f) * ACCEL_NOISE_DENSITY * ACCEL_NOISE_DENSITY
                      + (dt * dt * dt * dt / 4.0f) * GRAVITY_UNCERTAINTY * GRAVITY_UNCERTAINTY;
        float info_pos = IMU_WEIGHT_SCALE / var_pos;

        memset(omega, 0, ERROR_SIZE * ERROR_SIZE * sizeof(float));
        omega[0]  = info_rot;
        omega[10] = info_rot;
        omega[20] = info_rot;
        omega[30] = info_pos;
        omega[40] = info_pos;
        omega[50] = info_pos;
        omega[60] = info_vel;
        omega[70] = info_vel;
        omega[80] = info_vel;

        // --- Extract State i (previous) ---
        // Save C_i = R_bw before converting to Rodrigues (needed for Jacobian remap)
        float C_prev[9];
        matrix_2D_t prev_pose_mat = {(float*) &poses[prev_idx], {4,4,0,0}};
        rotationMatrixOfTransformation(&prev_pose_mat, &R);
        translationVectorOfTransformation(&prev_pose_mat, t);
        memcpy(C_prev, R_data, 9 * sizeof(float));
        matrixToRodrigues(R_data, r);
        float r_prev[3] = {-r[0], -r[1], -r[2]};  // r_bw = -r_cw
        float p_prev[3];
        recover_world_position(C_prev, t, p_prev);
        float vp[3] = {velocities[prev_idx].x, velocities[prev_idx].y, velocities[prev_idx].z};
        CameraState9Dof prev_state = {{r_prev[0], r_prev[1], r_prev[2]},
                                      {p_prev[0], p_prev[1], p_prev[2]},
                                      {vp[0], vp[1], vp[2]}};

        // --- Extract State j (current) ---
        float C_curr[9];
        matrix_2D_t curr_pose_mat = {(float*) &poses[curr_idx], {4,4,0,0}};
        rotationMatrixOfTransformation(&curr_pose_mat, &R);
        translationVectorOfTransformation(&curr_pose_mat, t);
        memcpy(C_curr, R_data, 9 * sizeof(float));
        matrixToRodrigues(R_data, r);
        float r_curr[3] = {-r[0], -r[1], -r[2]};  // r_bw = -r_cw
        float p_curr[3];
        recover_world_position(C_curr, t, p_curr);
        float vc[3] = {velocities[curr_idx].x, velocities[curr_idx].y, velocities[curr_idx].z};
        CameraState9Dof curr_state = {{r_curr[0], r_curr[1], r_curr[2]},
                                      {p_curr[0], p_curr[1], p_curr[2]},
                                      {vc[0], vc[1], vc[2]}};

        // --- Compute Error and Jacobian in physical coords ---
        compute_imu_error_and_jacobian(e_imu, J_imu,
                                       &prev_state, &curr_state,
                                       &imu_factors[curr_idx].base,
                                       g,
                                       work_memory);

        // --- Accumulate Total Error ---
        for(int j = 0; j < ERROR_SIZE; ++j)
            total_error += e_imu[j] * omega[j*ERROR_SIZE+j] * e_imu[j];

        if (Sb == NULL) continue;

        // --- Remap Jacobian columns from physical to T_cw parameterization ---
        remap_jacobian_to_tcw(J_imu, C_prev, C_curr);

        // Negate e for (-H, -b) sign convention before accumulating
        for(int i = 0; i < ERROR_SIZE; ++i) e_imu[i] = -e_imu[i];

        compose_system_from_jacobian(Sb, num_poses,
                                     prev_idx, curr_idx,
                                     e_imu, J_imu, omega,
                                     work_memory);
    }
    return total_error;
}