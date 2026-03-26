// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "definitions/type_definitions.h"
#include "epnp.h"
#include "math.h"

#define SAMPLE_SIZE 4
#define JACOBI_ITERS 120

typedef struct ransac_epnp_args
{
    point3D_float_t* sample_wps;
    point2D_u16_t* sample_ips;
    matrix_2D_t* K;
    float* t_temp_all_cores;
    float* reprojection_errrors;
    work_memory_t* work_memory;
} ransac_epnp_args_t;



/**
 * @brief Accumulates the contribution of one 3D-2D correspondence into M^T*M.
 *
 * For each point, two rows of the measurement matrix M are formed from the
 * barycentric coordinates (as) and camera intrinsics K:
 *   row1 = [a_i*fu, 0,      a_i*(uc-u)]  for each control point i
 *   row2 = [0,      a_i*fv, a_i*(vc-v)]
 *
 * These are accumulated as outer products into MtM (12x12), which is later
 * decomposed via SVD to find the control point coordinates in the camera frame.
 */
void epnp_fill_MtM(matrix_2D_t* MtM,
                   matrix_2D_t* K,
                   float* as, 
                   point2D_u16_t* image_point,
                   work_memory_t work_memory)
{
    float u = image_point->x;
    float v = image_point->y;
    float fu = K->data[0];
    float fv = K->data[4];
    float uc = K->data[2];
    float vc = K->data[5];
    float* M1 = allocate_work_memory(&work_memory, 12*sizeof(float));
    float* M2 = allocate_work_memory(&work_memory, 12*sizeof(float));
    
    for(int i = 0; i < 4; i++) {
        M1[3 * i    ] = as[i] * fu;
        M1[3 * i + 1] = 0.0;
        M1[3 * i + 2] = as[i] * (uc - u);
        
        M2[3 * i    ] = 0.0;
        M2[3 * i + 1] = as[i] * fv;
        M2[3 * i + 2] = as[i] * (vc - v);
    }

    for(int i = 0; i < 12; i++){
        for(int j = 0; j < 12; j++)
        {
            MtM->data[i*12 + j] += M1[i]*M1[j]+M2[i]*M2[j];
        }
    }
}

/**
 * @brief Selects 4 EPnP control points from the world point cloud using PCA.
 *
 * C0 is set to the centroid of the world points. C1, C2, C3 are offset from
 * C0 along the principal axes of the point cloud (from SVD of the centered
 * covariance), scaled by sqrt(eigenvalue / N) to match the data spread.
 */
void epnp_choose_control_points(point3D_float_t* world_points,
                                point3D_float_t* control_points,
                                work_memory_t work_memory,
                                uint16_t number_of_correspondences)
{
    // Take C0 as the reference points centroid:
    control_points[0].x = 0;
    control_points[0].y = 0;
    control_points[0].z = 0;
    for(int i = 0; i < number_of_correspondences; i++){
        control_points[0].x += world_points[i].x;
        control_points[0].y += world_points[i].y;
        control_points[0].z += world_points[i].z;
    }
    
    control_points[0].x /= number_of_correspondences;
    control_points[0].y /= number_of_correspondences;
    control_points[0].z /= number_of_correspondences;
    
    // Take C1, C2, and C3 from PCA on the reference points:
    float* pw0 = allocate_work_memory(&work_memory, number_of_correspondences*3*sizeof(float));
    float* pw0tpw0 = allocate_work_memory(&work_memory, 3*3*sizeof(float));
    float* dc = allocate_work_memory(&work_memory, 3*1*sizeof(float));
    float* uc = allocate_work_memory(&work_memory, 3*3*sizeof(float));
    float* vc = allocate_work_memory(&work_memory, 3*3*sizeof(float));
    matrix_2D_t PW0 = {pw0,{number_of_correspondences,3,0,0}};
    matrix_2D_t PW0t = {pw0,{number_of_correspondences,3,1,0}};
    matrix_2D_t PW0tPW0 = {pw0tpw0,{3,3,0,0}};
    matrix_2D_t UC = {uc,{3,3,0,0}};
    matrix_2D_t VC = {vc,{3,3,0,0}};

    for(int i = 0; i < number_of_correspondences; i++){
        PW0.data[3*i + 0] = world_points[i].x - control_points[0].x;
        PW0.data[3*i + 1] = world_points[i].y - control_points[0].y;
        PW0.data[3*i + 2] = world_points[i].z - control_points[0].z;
    }

    matmul(&PW0t,&PW0,&PW0tPW0);
    svd(&PW0tPW0,&UC,dc,&VC,work_memory);
    
    for(int i = 1; i < 4; i++) {
        float k = sqrtf(dc[i - 1] / number_of_correspondences);
        control_points[i].x = control_points[0].x + k * uc[3 * 0 + (i - 1)];
        control_points[i].y = control_points[0].y + k * uc[3 * 1 + (i - 1)];
        control_points[i].z = control_points[0].z + k * uc[3 * 2 + (i - 1)];
    }
}

/**
 * @brief Computes barycentric coordinates of each world point w.r.t. the 4 control points.
 *
 * Each world point p_i is expressed as: p_i = sum_{j=0}^{3} alpha_{ij} * c_j
 * with sum(alpha_i) = 1. The three non-trivial coordinates alpha[1..3] are
 * solved from the 3x3 system formed by the vectors c_j - c_0, then alpha[0] = 1 - sum.
 */
void epnp_compute_barycentric_coordinates(point3D_float_t* world_points,
                                          point3D_float_t* control_points,
                                          float* alphas,
                                          work_memory_t work_memory,
                                          uint16_t number_of_correspondences)
{
    float* cc = allocate_work_memory(&work_memory, 3*3*sizeof(float));
    float* ci = allocate_work_memory(&work_memory, 3*3*sizeof(float));
    matrix_2D_t CC = {cc,{3,3,0,0}};
    matrix_2D_t CI = {ci,{3,3,0,0}};

    for(int j = 1; j < 4; j++){
        cc[3 * 0 + j - 1] = control_points[j].x - control_points[0].x;
        cc[3 * 1 + j - 1] = control_points[j].y - control_points[0].y;
        cc[3 * 2 + j - 1] = control_points[j].z - control_points[0].z;
    }

    matinv3x3(&CC,&CI);

    for(int i = 0; i < number_of_correspondences; i++) {
        point3D_float_t* pi = world_points + i;
        float* a = alphas + 4 * i;

        for(int j = 0; j < 3; j++){
            a[1 + j] =
                ci[3 * j    ] * (pi->x - control_points[0].x) +
                ci[3 * j + 1] * (pi->y - control_points[0].y) +
                ci[3 * j + 2] * (pi->z - control_points[0].z);
        }
        a[0] = 1.0f - a[1] - a[2] - a[3];
    }
}

/**
 * @brief Sorts eigenvectors by descending eigenvalue and writes the result as row vectors into ut.
 *
 * Given the column-major eigenvector matrix U and eigenvalue vector d (from jacobi_eigen),
 * this function selection-sorts the column indices by descending d, then writes
 * ut[i*12 + j] = u[j*12 + sorted_index[i]], producing the top-4 null-space vectors
 * in the last four rows (ut[8..11]) as expected by epnp_compute_L_6x10.
 */
void extract_ut(float* u, float* d, float* ut)
{
    // Sort largest to smallest entry
    uint8_t indices[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    for (int i = 0; i < 11; ++i) {
        for (int j = i + 1; j < 12; ++j) {
            if (d[indices[i]] < d[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j){
            ut[i*12+j] = u[j*12+indices[i]];
        }
    }
}

/**
 * @brief Builds the 6x10 L matrix encoding pairwise control-point distance constraints.
 *
 * The solution x = sum(beta_i * v_i) must preserve inter-control-point distances.
 * For each of the 6 pairs (a,b), each row of L contains the 10 symmetric products
 * of the difference vectors dv[i][pair] = v_i[a] - v_i[b], corresponding to the
 * 10 beta products [B11, B12, B22, B13, B23, B33, B14, B24, B34, B44].
 */
void epnp_compute_L_6x10(float* ut, float* l_6x10)
{
  float* v[4];

  v[0] = ut + 12 * 11;
  v[1] = ut + 12 * 10;
  v[2] = ut + 12 *  9;
  v[3] = ut + 12 *  8;

  float dv[4][6][3];

  for(int i = 0; i < 4; i++) {
    int a = 0, b = 1;
    for(int j = 0; j < 6; j++) {
      dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

      b++;
      if (b > 3) {
	a++;
	b = a + 1;
      }
    }
  }

  for(int i = 0; i < 6; i++) {
    float* row = l_6x10 + 10 * i;

    row[0] =        dot(dv[0][i], dv[0][i], 3);
    row[1] = 2.0f * dot(dv[0][i], dv[1][i], 3);
    row[2] =        dot(dv[1][i], dv[1][i], 3);
    row[3] = 2.0f * dot(dv[0][i], dv[2][i], 3);
    row[4] = 2.0f * dot(dv[1][i], dv[2][i], 3);
    row[5] =        dot(dv[2][i], dv[2][i], 3);
    row[6] = 2.0f * dot(dv[0][i], dv[3][i], 3);
    row[7] = 2.0f * dot(dv[1][i], dv[3][i], 3);
    row[8] = 2.0f * dot(dv[2][i], dv[3][i], 3);
    row[9] =        dot(dv[3][i], dv[3][i], 3);
  }
}

float dist2(point3D_float_t* p1, point3D_float_t* p2)
{
    float dist = 0.0;
    dist += (p1->x - p2->x) * (p1->x - p2->x);
    dist += (p1->y - p2->y) * (p1->y - p2->y);
    dist += (p1->z - p2->z) * (p1->z - p2->z);
    return dist;
}

void epnp_compute_rho(point3D_float_t* control_points, float* rho)
{
  rho[0] = dist2(control_points + 0, control_points + 1);
  rho[1] = dist2(control_points + 0, control_points + 2);
  rho[2] = dist2(control_points + 0, control_points + 3);
  rho[3] = dist2(control_points + 1, control_points + 2);
  rho[4] = dist2(control_points + 1, control_points + 3);
  rho[5] = dist2(control_points + 2, control_points + 3);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

/**
 * @brief Solves for all 4 betas assuming they are all non-zero (N=4 approximation).
 *
 * Extracts the 4 relevant columns from L_6x10 (corresponding to B11, B12, B13, B14)
 * to form L_6x4, then solves L_6x4 * [B11, B12, B13, B14]^T = rho via QR.
 * Recovers betas from the symmetric-product representation.
 */
void epnp_find_betas_approx_1(matrix_2D_t* L_6x10, matrix_2D_t* Rho,
			       float * betas, work_memory_t work_memory)
{
    float* l_6x4 = allocate_work_memory(&work_memory, 6 * 4 * sizeof(float));
    float* b4 = allocate_work_memory(&work_memory, 4 * sizeof(float));
    matrix_2D_t L_6x4 = {l_6x4,{6,4,0,0}};
    matrix_2D_t B4 = {b4,{4,1,0,0}};

    for(int i = 0; i < 6; i++) {
        L_6x4.data[i*4 + 0] = L_6x10->data[i*10+0];
        L_6x4.data[i*4 + 1] = L_6x10->data[i*10+1];
        L_6x4.data[i*4 + 2] = L_6x10->data[i*10+3];
        L_6x4.data[i*4 + 3] = L_6x10->data[i*10+6];
    }

    solve_linear_qr(&L_6x4, Rho, &B4, work_memory);

    if (b4[0] < 0) {
        betas[0] = sqrtf(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = sqrtf(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

/**
 * @brief Solves for 2 betas assuming beta3 = beta4 = 0 (N=2 approximation).
 *
 * Extracts columns for [B11, B12, B22] from L_6x10, solves via QR,
 * then recovers beta1 and beta2. beta3 and beta4 are set to zero.
 */
void epnp_find_betas_approx_2(matrix_2D_t* L_6x10, matrix_2D_t* Rho,
			       float * betas, work_memory_t work_memory)
{
    float* l_6x3 = allocate_work_memory(&work_memory, 6 * 3 * sizeof(float));
    float* b3 = allocate_work_memory(&work_memory, 3 * sizeof(float));
    matrix_2D_t L_6x3 = {l_6x3,{6,3,0,0}};
    matrix_2D_t B3 = {b3,{3,1,0,0}};

    for(int i = 0; i < 6; i++) {
        L_6x3.data[i*3 + 0] = L_6x10->data[i*10+0];
        L_6x3.data[i*3 + 1] = L_6x10->data[i*10+1];
        L_6x3.data[i*3 + 2] = L_6x10->data[i*10+2];
    }

    solve_linear_qr(&L_6x3, Rho, &B3, work_memory);

    if (b3[0] < 0) {
        betas[0] = sqrtf(-b3[0]);
        betas[1] = (b3[2] < 0) ? sqrtf(-b3[2]) : 0.0;
    } else {
        betas[0] = sqrtf(b3[0]);
        betas[1] = (b3[2] > 0) ? sqrtf(b3[2]) : 0.0;
    }

    if (b3[1] < 0) betas[0] = -betas[0];

    betas[2] = 0.0;
    betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

/**
 * @brief Solves for 3 betas assuming beta4 = 0 (N=3 approximation).
 *
 * Extracts columns for [B11, B12, B22, B13, B23] from L_6x10, solves via QR,
 * then recovers beta1, beta2, beta3. beta4 is set to zero.
 */
void epnp_find_betas_approx_3(matrix_2D_t* L_6x10, matrix_2D_t* Rho,
			       float * betas, work_memory_t work_memory)
{
    float* l_6x5 = allocate_work_memory(&work_memory, 6 * 5 * sizeof(float));
    float* b5 = allocate_work_memory(&work_memory, 5 * sizeof(float));
    matrix_2D_t L_6x5 = {l_6x5,{6,5,0,0}};
    matrix_2D_t B5 = {b5,{5,1,0,0}};

    for(int i = 0; i < 6; i++) {
        L_6x5.data[i*5 + 0] = L_6x10->data[i*10+0];
        L_6x5.data[i*5 + 1] = L_6x10->data[i*10+1];
        L_6x5.data[i*5 + 2] = L_6x10->data[i*10+2];
        L_6x5.data[i*5 + 3] = L_6x10->data[i*10+3];
        L_6x5.data[i*5 + 4] = L_6x10->data[i*10+4];
    }

    solve_linear_qr(&L_6x5, Rho, &B5, work_memory);

    if (b5[0] < 0) {
        betas[0] = sqrtf(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrtf(-b5[2]) : 0.0;
    } else {
        betas[0] = sqrtf(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrtf(b5[2]) : 0.0;
    }
    if (b5[1] < 0) betas[0] = -betas[0];
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}

/**
 * @brief Builds the linearized system (A, b) for one Gauss-Newton beta refinement step.
 *
 * The residual is r_i = rho_i - L_i(beta*beta^T), where L_i is row i of l_6x10.
 * The Jacobian row is the partial derivative of L_i(beta*beta^T) w.r.t. each beta_j,
 * which involves the symmetric cross-terms from the 10-parameter product expansion.
 */
void epnp_compute_A_and_b_gauss_newton(float* l_6x10, float* rho,
					float betas[4], matrix_2D_t* A, matrix_2D_t* b)
{
  for(int i = 0; i < 6; i++) {
    float* rowL = l_6x10 + i * 10;
    float* rowA = A->data + i * 4;

    rowA[0] = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];
    rowA[1] =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];
    rowA[2] =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];
    rowA[3] =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

    b->data[i] = rho[i] -
	   (
	    rowL[0] * betas[0] * betas[0] +
	    rowL[1] * betas[0] * betas[1] +
	    rowL[2] * betas[1] * betas[1] +
	    rowL[3] * betas[0] * betas[2] +
	    rowL[4] * betas[1] * betas[2] +
	    rowL[5] * betas[2] * betas[2] +
	    rowL[6] * betas[0] * betas[3] +
	    rowL[7] * betas[1] * betas[3] +
	    rowL[8] * betas[2] * betas[3] +
	    rowL[9] * betas[3] * betas[3]
	    );
  }
}

/**
 * @brief Refines the 4 EPnP betas with a fixed number of Gauss-Newton iterations.
 *
 * At each step, builds the linearized system A*delta = b around the current betas
 * using epnp_compute_A_and_b_gauss_newton, solves it via QR, and updates betas += delta.
 * Runs 5 iterations, which is sufficient for convergence in practice.
 */
void epnp_gauss_newton(matrix_2D_t* L_6x10, matrix_2D_t* Rho,
			float betas[4], work_memory_t work_memory)
{
  int iterations_number = 5;

  float* a = allocate_work_memory(&work_memory, 6*4*sizeof(float));
  float* b = allocate_work_memory(&work_memory, 6*sizeof(float));
  float* x = allocate_work_memory(&work_memory, 4*sizeof(float));
  matrix_2D_t A = {a,{6,4,0,0}};
  matrix_2D_t B = {b,{6,1,0,0}};
  matrix_2D_t X = {x,{4,1,0,0}};

  for(int k = 0; k < iterations_number; k++) {
    epnp_compute_A_and_b_gauss_newton(L_6x10->data, Rho->data,
				 betas, &A, &B);
    // Original uses QR solver not SVD
    solve_linear_qr(&A, &B, &X, work_memory);

    for(int i = 0; i < 4; i++)
      betas[i] += x[i];
  }
}

/**
 * @brief Reconstructs the 4 control points in the camera frame from the null-space vectors.
 *
 * Computes ccs[j] = sum_{i=0}^{3} betas[i] * v_i[j*3 : j*3+3], where v_i are the 4
 * null-space row vectors stored in ut (rows 11, 10, 9, 8 in descending eigenvalue order).
 */
void epnp_compute_ccs(float ccs[4][3], float* betas, float* ut)
{
  for(int i = 0; i < 4; i++) {
    float* v = ut + 12 * (11 - i);
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 3; k++)
	ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

/**
 * @brief Reconstructs each correspondence's camera-frame 3D position from barycentric coordinates.
 *
 * Computes pcs[i] = sum_{j=0}^{3} alphas[4*i+j] * ccs[j], expressing each world point
 * in the camera frame using the solved control point positions.
 */
void epnp_compute_pcs(float* alphas,
                      float* pcs,
                      float ccs[4][3],
                      uint16_t number_of_correspondences)
{
  for(int i = 0; i < number_of_correspondences; i++) {
    float* a = alphas + 4 * i;
    float* pc = pcs + 3 * i;

    for(int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

/**
 * @brief Enforces positive depth (cheirality) by flipping all camera-frame coordinates if needed.
 *
 * If the first reconstructed point has negative z (behind the camera), all control point
 * and point-cloud z values are negated so that points lie in front of the camera.
 */
void epnp_solve_for_sign(float* pcs,
                         float ccs[4][3],
                         uint16_t number_of_correspondences)
{
  if (pcs[2] < 0.0) {
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
	ccs[i][j] = -ccs[i][j];

    for(int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i    ] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

/**
 * @brief Estimates rotation R and translation t from camera-frame and world-frame point sets.
 *
 * Computes the cross-covariance matrix ABt = sum((pc_i - pc0) * (pw_i - pw0)^T),
 * decomposes it with SVD, and recovers R = U * V^T. Enforces det(R)=1 to avoid
 * reflections. Translation is t = pc0 - R * pw0 (centroid alignment).
 */
void epnp_estimate_R_and_t(float R[3][3],
                            float t[3],
                            float* pcs,
                            point3D_float_t* world_points,
                            work_memory_t work_memory,
                            uint16_t number_of_correspondences)
{
    float pc0[3], pw0[3];

    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;

    for(int i = 0; i < number_of_correspondences; i++) {
        float* pc = pcs + 3 * i;
        point3D_float_t* pw = world_points + i;

        for(int j = 0; j < 3; j++) {
        pc0[j] += pc[j];
        }
        pw0[0] += pw->x;
        pw0[1] += pw->y;
        pw0[2] += pw->z;
    }
    for(int j = 0; j < 3; j++) {
        pc0[j] /= number_of_correspondences;
        pw0[j] /= number_of_correspondences;
    }

    float* abt = allocate_work_memory(&work_memory, 3*3*sizeof(float));
    float* abt_d = allocate_work_memory(&work_memory, 3*sizeof(float));
    float* abt_u = allocate_work_memory(&work_memory, 3*3*sizeof(float));
    float* abt_v = allocate_work_memory(&work_memory, 3*3*sizeof(float));
    matrix_2D_t ABt = {abt,{3,3,0,0}};
    matrix_2D_t ABt_U = {abt_u,{3,3,0,0}};
    matrix_2D_t ABt_V = {abt_v,{3,3,0,0}};

    memset(abt,0,3*3*sizeof(float));
    for(int i = 0; i < number_of_correspondences; i++) {
        float* pc = pcs + 3 * i;
        point3D_float_t* pw = world_points + i;

        for(int j = 0; j < 3; j++) {
        abt[3 * j    ] += (pc[j] - pc0[j]) * (pw->x - pw0[0]);
        abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw->y - pw0[1]);
        abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw->z - pw0[2]);
        }
    }

    svd(&ABt, &ABt_U, abt_d, &ABt_V, work_memory);

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
        R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j, 3);

    const float det =
        R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
        R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

    if (det < 0) {
        R[2][0] = -R[2][0];
        R[2][1] = -R[2][1];
        R[2][2] = -R[2][2];
    }

    t[0] = pc0[0] - dot(R[0], pw0, 3);
    t[1] = pc0[1] - dot(R[1], pw0, 3);
    t[2] = pc0[2] - dot(R[2], pw0, 3);
}

/**
 * @brief Computes the mean reprojection error for a given R, t, and set of correspondences.
 *
 * Projects each world point through R and t, applies K, and accumulates the Euclidean
 * pixel error. Points behind the camera (Zc < 0) incur a large penalty (1e3 * |Zc|)
 * to disfavor degenerate solutions during beta-set selection.
 *
 * @return Mean reprojection error per point (including depth penalties).
 */
float epnp_reprojection_error(float R[3][3],
                            float t[3],
                            point3D_float_t* world_points,
                            point2D_u16_t* image_points,
                            matrix_2D_t* K,
                            uint16_t number_of_correspondences)
{
    float sum2 = 0.0;
    float pw[3];
    float fu = K->data[0];
    float fv = K->data[4];
    float uc = K->data[2];
    float vc = K->data[5];


    for(int i = 0; i < number_of_correspondences; i++) {
        pw[0] = world_points[i].x;
        pw[1] = world_points[i].y;
        pw[2] = world_points[i].z;
        float Xc = dot(R[0], pw, 3) + t[0];
        float Yc = dot(R[1], pw, 3) + t[1];
        float Zc = dot(R[2], pw, 3) + t[2];
        float inv_Zc = 1.0 / Zc;
        float ue = uc + fu * Xc * inv_Zc;
        float ve = vc + fv * Yc * inv_Zc;
        float u = image_points[i].x;
        float v = image_points[i].y;

        float z_penalty = (Zc < 0) * (-Zc) * BEHIND_CAMERA_PENALTY;
        sum2 += z_penalty;

        sum2 += sqrtf( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
    }

    return sum2 / number_of_correspondences;
}


/**
 * @brief Recovers R and t from a given set of betas and evaluates the reprojection error.
 *
 * Reconstructs camera-frame control points (ccs) and point positions (pcs) from the
 * null-space vectors and betas, enforces positive depth, estimates R and t via
 * epnp_estimate_R_and_t, and returns the resulting mean reprojection error.
 *
 * @return Mean reprojection error for this beta configuration.
 */
float epnp_compute_R_and_t(float* ut,
                           float* alphas,
                           float* betas,
			               float R[3][3],
                           float t[3],
                           point3D_float_t*  world_points,
                           point2D_u16_t* image_points,
                           matrix_2D_t* K,
                           work_memory_t work_memory,
                           uint16_t number_of_correspondences)
{
    float ccs[4][3] = {0};
    float* pcs = allocate_work_memory(&work_memory, 3*number_of_correspondences*sizeof(float));
    epnp_compute_ccs(ccs, betas, ut);
    epnp_compute_pcs(alphas, pcs, ccs, number_of_correspondences);

    epnp_solve_for_sign(pcs, ccs, number_of_correspondences);

    epnp_estimate_R_and_t(R, t, pcs, world_points, work_memory, number_of_correspondences);

    return epnp_reprojection_error(R, t, world_points, image_points, K, number_of_correspondences);
}

/**
 * @brief Packs an orthonormalized R and t into a 4x4 homogeneous transformation matrix.
 *
 * Applies orthonormalization to R_src (to correct numerical drift), then writes
 * T = [R | t; 0 0 0 1].
 */
void epnp_copy_R_and_t(float R_src[3][3], float t_src[3], matrix_2D_t* T)
{
    float r[9];
    matrix_2D_t R_dash = {(float*) R_src,{3,3,0,0}};
    matrix_2D_t R = {r,{3,3,0,0}};
    orthonormalize(&R_dash,&R);
    memset(T->data,0,4*4*sizeof(float));
    T->data[15] = 1.0;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++){
            T->data[i*4+j] = r[i*3+j];
        }
        T->data[4*i+3] = t_src[i];
    }
}
    
/**
 * @brief Solves for camera pose using the EPnP algorithm on a set of 3D-2D correspondences.
 *
 * Selects PCA-based control points, computes barycentric coordinates, builds and
 * decomposes M^T*M, then evaluates three beta approximations (N=1,2,3 non-zero betas)
 * each refined by Gauss-Newton. Returns the pose with the lowest reprojection error.
 *
 * @return Mean reprojection error of the best solution found.
 */
float epnp_compute_pose(point3D_float_t* world_points,
                        point2D_u16_t* image_points,
                        matrix_2D_t* K,
                        matrix_2D_t* T,
                        work_memory_t work_memory,
                        uint16_t number_of_correspondences)
{
    point3D_float_t* control_points = allocate_work_memory(&work_memory,4*sizeof(point3D_float_t));
    float* alphas = allocate_work_memory(&work_memory,4*number_of_correspondences*sizeof(float));
    epnp_choose_control_points(world_points, control_points, work_memory, number_of_correspondences);
    epnp_compute_barycentric_coordinates(world_points, control_points, alphas, work_memory, number_of_correspondences);
    
    float* mtm = allocate_work_memory(&work_memory, 12*12*sizeof(float));
    memset(mtm,0,12*12*sizeof(float));
    matrix_2D_t MtM = {mtm,{12,12,0,0}};
    
    for(int i = 0; i < number_of_correspondences; i++){
        epnp_fill_MtM(&MtM, K, alphas + 4 * i, image_points + i, work_memory);
    }
    
    // We are looking for Ut of the SVD of MtM = H
    // U is equvalent to the eigenvector of H*H^T 
    matrix_2D_t H = {mtm,{12,12,0,0}};
    matrix_2D_t Ht = {mtm,{12,12,1,0}};
    float* hht = allocate_work_memory(&work_memory, 12*12*sizeof(float));
    matrix_2D_t HHt = {hht,{12,12,1,0}};
    matmul(&H,&Ht,&HHt);
    float f_norm = frobenius_norm(&HHt);
    matscale(&HHt, 1/f_norm);

    float* d = allocate_work_memory(&work_memory, 12*sizeof(float));
    float* u = allocate_work_memory(&work_memory, 12*12*sizeof(float));
    matrix_2D_t U  = {u,{12,12,0,0}};
    matrix_2D_t D  = {d,{12,1,0,0}};
    jacobi_eigen(hht,u,d,12,JACOBI_ITERS,1e-8f);

    float* ut = allocate_work_memory(&work_memory, 12*12*sizeof(float));
    extract_ut(u,d,ut);
    
    float* l_6x10 = allocate_work_memory(&work_memory, 6*10*sizeof(float));
    float* rho = allocate_work_memory(&work_memory, 6*sizeof(float));
    matrix_2D_t L_6x10 = {l_6x10,{6,10,0,0}};
    matrix_2D_t Rho    = {rho,{6,1,0,0}};
    
    epnp_compute_L_6x10(ut, l_6x10);
    epnp_compute_rho(control_points, rho);
    
    float Betas[4][4], rep_errors[4];
    float Rs[4][3][3], ts[4][3];
    
    epnp_find_betas_approx_1(&L_6x10, &Rho, Betas[1], work_memory);
    epnp_gauss_newton(&L_6x10, &Rho, Betas[1], work_memory);
    rep_errors[1] = epnp_compute_R_and_t(ut, alphas, Betas[1], Rs[1], ts[1], world_points, image_points, K, work_memory, number_of_correspondences);
    
    epnp_find_betas_approx_2(&L_6x10, &Rho, Betas[2], work_memory);
    epnp_gauss_newton(&L_6x10, &Rho, Betas[2], work_memory);
    rep_errors[2] = epnp_compute_R_and_t(ut, alphas, Betas[2], Rs[2], ts[2], world_points, image_points, K, work_memory, number_of_correspondences);
    
    epnp_find_betas_approx_3(&L_6x10, &Rho, Betas[3], work_memory);
    epnp_gauss_newton(&L_6x10, &Rho, Betas[3], work_memory);
    rep_errors[3] = epnp_compute_R_and_t(ut, alphas, Betas[3], Rs[3], ts[3], world_points, image_points, K, work_memory, number_of_correspondences);
    
    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;
    
    epnp_copy_R_and_t(Rs[N], ts[N], T);
    
    return rep_errors[N];
}

float ransac_epnp_compute_pose(point3D_float_t* world_points,
                               point2D_u16_t* image_points,
                               matrix_2D_t* K,
                               matrix_2D_t* T,
                               work_memory_t work_memory,
                               uint16_t number_of_correspondences,
                               uint16_t ransac_iterations)
{
    LOG_INFO("Start ePnP RANSAC\n");
    uint16_t bestInlierCount = 0;
    float inlierThreshold = sqrtf(EPNP_INLIER_THRESHOLD_SQ);
    uint16_t offsets[4] = {0,3,7,13};
    float bestAvgInlierError = 1000.0;

    point3D_float_t* sample_wp = allocate_work_memory(&work_memory, SAMPLE_SIZE*sizeof(point3D_float_t));
    point2D_u16_t* sample_ip = allocate_work_memory(&work_memory, SAMPLE_SIZE*sizeof(point2D_u16_t));

    float* t_temp = allocate_work_memory(&work_memory, 4*4*sizeof(float));
    matrix_2D_t T_temp = {t_temp,{4,4,0,0}};

    uint16_t* all_indices = allocate_work_memory(&work_memory, number_of_correspondences * sizeof(uint16_t));
    for (uint16_t i = 0; i < number_of_correspondences; i++) {
        all_indices[i] = i;
    }

    for (int iter = 0; iter < ransac_iterations; iter++) {
        // Randomly select 8 point
        // Fisher-Yates shuffle for first sample_size elements
        for (uint8_t i = 0; i < SAMPLE_SIZE; i++) {
            // Generate unbiased random index in range [i, number_of_correspondences-1]
            uint16_t j = i + rand_range(number_of_correspondences - i);
            // Swap elements
            uint16_t temp = all_indices[i];
            all_indices[i] = all_indices[j];
            all_indices[j] = temp;

            uint16_t index = all_indices[i];
            sample_wp[i] = world_points[index];
            sample_ip[i] = image_points[index];
        }

        float avgInlierError = epnp_compute_pose(sample_wp, sample_ip, K, &T_temp, work_memory, SAMPLE_SIZE);
        
        uint16_t inlierCount = 0;
        for (int i = 0; i < number_of_correspondences; i++) {
            float error = reprojection_error(&world_points[i], &image_points[i], &T_temp, K);
            if (error < inlierThreshold) {
                inlierCount++;
            }
        }

        if ((inlierCount > bestInlierCount) || (inlierCount == bestInlierCount && avgInlierError < bestAvgInlierError)) {
            bestInlierCount = inlierCount;
            bestAvgInlierError = avgInlierError;
            for (int i = 0; i < 16; i++){
                    T->data[i] = T_temp.data[i];
            }
            LOG_DEBUG("Iteration %d\n",iter);
            LOG_DEBUG("Inliers count %d\n",inlierCount);
            matprint(T, DEBUG_LEVEL);
        }
    }

    LOG_INFO("Best inlier error: %f\n", bestAvgInlierError);
    return bestAvgInlierError;
}

/**
 * @brief Cluster kernel: each core runs epnp_compute_pose on its pre-drawn sample.
 *
 * Each core reads SAMPLE_SIZE correspondences from sample_wps/sample_ips[core_id*SAMPLE_SIZE],
 * computes the pose, and writes the 4x4 result to t_temp_all_cores[core_id*16] and
 * the reprojection error to reprojection_errors[core_id].
 */
void ransac_epnp_iteration(void* args)
{
    ransac_epnp_args_t* iter_args = (ransac_epnp_args_t*) args;
    uint16_t core_id = pi_core_id();
    work_memory_t work_memory = split_work_memory(iter_args->work_memory, NB_CORES, core_id);

    matrix_2D_t T_temp = {&iter_args->t_temp_all_cores[core_id*4*4], {4,4,0,0}};
    iter_args->reprojection_errrors[core_id] = epnp_compute_pose(&iter_args->sample_wps[core_id*SAMPLE_SIZE],
                                                                 &iter_args->sample_ips[core_id*SAMPLE_SIZE],
                                                                 iter_args->K,
                                                                 &T_temp,
                                                                 work_memory,
                                                                 SAMPLE_SIZE);
}

float ransac_epnp_compute_pose_multicore(point3D_float_t* world_points,
                                         point2D_u16_t* image_points,
                                         matrix_2D_t* K,
                                         matrix_2D_t* T,
                                         work_memory_t work_memory,
                                         uint16_t number_of_correspondences,
                                         uint16_t ransac_iterations,
                                         uint8_t nb_cores)
{
    LOG_INFO("Start ePnP RANSAC\n");
    uint16_t bestInlierCount = 0;
    float inlierThreshold = sqrtf(EPNP_INLIER_THRESHOLD_SQ);
    uint16_t offsets[4] = {0,3,7,13};
    float bestAvgInlierError = 1000.0;

    point3D_float_t* sample_wp = allocate_work_memory(&work_memory, NB_CORES*SAMPLE_SIZE*sizeof(point3D_float_t));
    point2D_u16_t* sample_ip = allocate_work_memory(&work_memory, NB_CORES*SAMPLE_SIZE*sizeof(point2D_u16_t));

    float* t_temp = allocate_work_memory(&work_memory, NB_CORES*4*4*sizeof(float));
    float* reprojection_errrors = allocate_work_memory(&work_memory, NB_CORES*sizeof(float));

    uint16_t* all_indices = allocate_work_memory(&work_memory, number_of_correspondences * sizeof(uint16_t));
    for (uint16_t i = 0; i < number_of_correspondences; i++) {
        all_indices[i] = i;
    }

    ransac_epnp_args_t args = {sample_wp,
                               sample_ip,
                               K,
                               t_temp,
                               reprojection_errrors,
                               &work_memory};

    for (int iter = 0; iter < ransac_iterations/nb_cores; iter++) {
        // Randomly select 8 point
        for (uint8_t core_id = 0; core_id < nb_cores; core_id++)
        {
            // Fisher-Yates shuffle for first sample_size elements
            for (uint8_t i = 0; i < SAMPLE_SIZE; i++) {
                // Generate unbiased random index in range [i, number_of_correspondences-1]
                uint16_t j = i + rand_range(number_of_correspondences - i);
                // Swap elements
                uint16_t temp = all_indices[i];
                all_indices[i] = all_indices[j];
                all_indices[j] = temp;

                uint16_t index = all_indices[i];
                sample_wp[core_id*SAMPLE_SIZE+i] = world_points[index];
                sample_ip[core_id*SAMPLE_SIZE+i] = image_points[index];
            }
        }

        pi_cl_team_fork(nb_cores, ransac_epnp_iteration, &args);

        for (int core_id = 0; core_id < nb_cores; ++core_id) {
            matrix_2D_t T_temp = {&t_temp[core_id*4*4], {4,4,0,0}};
            uint16_t inlierCount = 0;
            for (int i = 0; i < number_of_correspondences; i++) {
                float error = reprojection_error(&world_points[i], &image_points[i], &T_temp, K);
                if (error < inlierThreshold) {
                    inlierCount++;
                }
            }

            if ((inlierCount > bestInlierCount) || (inlierCount == bestInlierCount && reprojection_errrors[core_id] < bestAvgInlierError)) {
                bestInlierCount = inlierCount;
                bestAvgInlierError = reprojection_errrors[core_id];
                for (int i = 0; i < 16; i++){
                        T->data[i] = T_temp.data[i];
                }
            }
        }
    }

    LOG_INFO("Best inlier error: %f\n", bestAvgInlierError);
    return bestAvgInlierError;
}