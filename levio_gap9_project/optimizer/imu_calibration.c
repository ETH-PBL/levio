// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

/**
 * imu_stationary_calib.c
 *
 * Derives gyro bias and gravity vector (body frame) from a stationary
 * pre-integration window, using the pre-computed bias Jacobians stored
 * in imu_factor_with_bias_t.
 *
 * Correction model (Forster et al. TRO 2017, eq. 44):
 *
 *   delta_R_corr = delta_R * Exp(J_r_bg * b_g)
 *   delta_v_corr = delta_v + J_v_bg * b_g + J_v_ba * b_a
 *   delta_p_corr = delta_p + J_p_bg * b_g + J_p_ba * b_a
 *
 * Stationary constraints (true motion = zero, gravity = g_b):
 *
 *   (1) delta_R_corr = I
 *         => J_r_bg * b_g = -Log(delta_R)
 *         => b_g = -J_r_bg^{-1} * Log(delta_R)
 *
 *   (2) delta_v_corr = g_b * T
 *         => g_b = (delta_v + J_v_bg * b_g) / T        [ignores b_a term]
 *         or with b_a known: subtract J_v_ba * b_a first
 *
 *   (3) delta_p_corr = 0.5 * g_b * T^2
 *         => g_b = 2 * (delta_p + J_p_bg * b_g) / T^2  [ignores b_a term]
 *
 * Note on accel bias:
 *   With stationary data only, g_b and b_a are not individually separable.
 *   The J_v_ba / J_p_ba terms require a known b_a to use. Without it, the
 *   estimates from (2) and (3) give  (g_b - J_v_ba*b_a / T)  and similarly
 *   for position -- i.e. the accel bias component along gravity is absorbed.
 *   If b_a is available from a prior calibration, pass it in; otherwise zero.
 */

#include <math.h>
#include "pmsis.h"

#include "imu_calibration.h"


/* ------------------------------------------------------------------ */
/* Internal vector / matrix helpers                                     */
/* ------------------------------------------------------------------ */

#define PI      3.14159265358979f
#define GRAVITY GRAVITY_MAGNITUDE
#define EPS     1e-8f

typedef point3D_float_t v3_t;

static inline float  v3_norm (v3_t v)           { return sqrtf(v.x*v.x+v.y*v.y+v.z*v.z); }
static inline v3_t   v3_scale(v3_t v, float s)  { return (v3_t){v.x*s, v.y*s, v.z*s}; }
static inline v3_t   v3_add  (v3_t a, v3_t b)   { return (v3_t){a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline v3_t   v3_sub  (v3_t a, v3_t b)   { return (v3_t){a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline float  v3_dot  (v3_t a, v3_t b)   { return a.x*b.x+a.y*b.y+a.z*b.z; }

/* M[row][col] accessed as flat[row*3+col] */
#define M(m,r,c) (m)[(r)*3+(c)]

/* y = M * x  (M is flat 9-float row-major) */
static v3_t mat_vmul(const float *M9, v3_t x) {
    return (v3_t){
        M9[0]*x.x + M9[1]*x.y + M9[2]*x.z,
        M9[3]*x.x + M9[4]*x.y + M9[5]*x.z,
        M9[6]*x.x + M9[7]*x.y + M9[8]*x.z
    };
}

/* Invert a 3x3 stored as flat[9] row-major. Returns 0 if singular. */
static int mat3_inv(const float *A, float *Ai) {
    float det =
        A[0]*(A[4]*A[8]-A[5]*A[7])
       -A[1]*(A[3]*A[8]-A[5]*A[6])
       +A[2]*(A[3]*A[7]-A[4]*A[6]);
    if (fabsf(det) < EPS) return 0;
    float id = 1.f / det;
    Ai[0] = id*(A[4]*A[8]-A[5]*A[7]);
    Ai[1] = id*(A[2]*A[7]-A[1]*A[8]);
    Ai[2] = id*(A[1]*A[5]-A[2]*A[4]);
    Ai[3] = id*(A[5]*A[6]-A[3]*A[8]);
    Ai[4] = id*(A[0]*A[8]-A[2]*A[6]);
    Ai[5] = id*(A[2]*A[3]-A[0]*A[5]);
    Ai[6] = id*(A[3]*A[7]-A[4]*A[6]);
    Ai[7] = id*(A[1]*A[6]-A[0]*A[7]);
    Ai[8] = id*(A[0]*A[4]-A[1]*A[3]);
    return 1;
}

/*
 * SO(3) matrix logarithm.
 * R is flat[9] row-major.  Returns rotation vector phi = theta * axis.
 */
static v3_t so3_log(const float *R) {
    float tr = R[0]+R[4]+R[8];
    float ct = 0.5f*(tr-1.f);
    if (ct >  1.f) ct =  1.f;
    if (ct < -1.f) ct = -1.f;
    float theta = acosf(ct);

    float rx = R[7]-R[5];   /* R[2][1]-R[1][2] */
    float ry = R[2]-R[6];   /* R[0][2]-R[2][0] */
    float rz = R[3]-R[1];   /* R[1][0]-R[0][1] */

    float scale;
    if (fabsf(theta) < EPS) {
        scale = 0.5f;
    } else if (fabsf(theta - PI) < 1e-4f) {
        /* theta ~ pi: recover axis from symmetric part */
        float diag[3] = { R[0], R[4], R[8] };
        int im = 0;
        if (diag[1] > diag[im]) im = 1;
        if (diag[2] > diag[im]) im = 2;
        float n[3] = {0};
        switch (im) {
            case 0:
                n[0] = sqrtf(0.5f*(1.f+R[0]));
                n[1] = R[1]/(2.f*n[0]);
                n[2] = R[2]/(2.f*n[0]);
                break;
            case 1:
                n[1] = sqrtf(0.5f*(1.f+R[4]));
                n[0] = R[3]/(2.f*n[1]);
                n[2] = R[5]/(2.f*n[1]);
                break;
            default:
                n[2] = sqrtf(0.5f*(1.f+R[8]));
                n[0] = R[6]/(2.f*n[2]);
                n[1] = R[7]/(2.f*n[2]);
                break;
        }
        float sgn = (rx*n[0]+ry*n[1]+rz*n[2]) >= 0.f ? 1.f : -1.f;
        return (v3_t){ sgn*n[0]*theta, sgn*n[1]*theta, sgn*n[2]*theta };
    } else {
        scale = theta / (2.f * sinf(theta));
    }
    return (v3_t){ scale*rx, scale*ry, scale*rz };
}
 
/* Skew-symmetric matrix [v]x written into flat[9] row-major */
static void skew(v3_t v, float S[9])
{
    S[0] =  0.f;   S[1] = -v.z;  S[2] =  v.y;
    S[3] =  v.z;   S[4] =  0.f;  S[5] = -v.x;
    S[6] = -v.y;   S[7] =  v.x;  S[8] =  0.f;
}
 
/* C = s*I */
static void mat_identity_scale(float C[9], float s)
{
    memset(C, 0, 9 * sizeof(float));
    C[0] = s;  C[4] = s;  C[8] = s;
}
 
/* C = A + B  (all flat[9]) */
static void mat_add(const float A[9], const float B[9], float C[9])
{
    for (int i = 0; i < 9; i++) C[i] = A[i] + B[i];
}
 
/* C = s * A */
static void mat_scale(const float A[9], float s, float C[9])
{
    for (int i = 0; i < 9; i++) C[i] = A[i] * s;
}
 
/* C = A * B  (3x3 row-major) */
static void mat_mul(const float A[9], const float B[9], float C[9])
{
    float tmp[9] = {0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                tmp[i*3+j] += A[i*3+k] * B[k*3+j];
    memcpy(C, tmp, 9 * sizeof(float));
}
 

/* ------------------------------------------------------------------ */
/* Main calibration function                                            */
/* ------------------------------------------------------------------ */

/**
 * imu_calib_stationary() - estimate gyro bias and gravity from a
 * stationary pre-integration window, using pre-computed Jacobians.
 *
 * @factor:    pre-integration factor with bias Jacobians
 * @b_a_prior: prior accel bias estimate [m/s^2]; pass zero-vector if unknown
 * @w_vel:     weight for velocity-derived gravity estimate  (0..1)
 * @w_pos:     weight for position-derived gravity estimate  (w_vel+w_pos=1)
 * @out:       output calibration result
 *
 * Returns 0 on success, -1 if J_r_bg is singular (cannot solve b_g).
 */
int imu_calib_stationary_with_jacobians(const imu_factor_with_bias_t *factor,
                                        v3_t                              b_a_prior,
                                        float                             w_vel,
                                        float                             w_pos,
                                        imu_calib_result_t               *out)
{
    const imu_factor_t *b = &factor->base;
    float T = b->dt;

    /* ----------------------------------------------------------------
     * Step 1 — Gyro bias
     *
     *   J_r_bg * b_g = -Log(delta_R)
     *   b_g          = -J_r_bg^{-1} * Log(delta_R)
     * ---------------------------------------------------------------- */
    v3_t  phi     = so3_log(b->dR);              /* Log(delta_R)          */
    v3_t  neg_phi = v3_scale(phi, -1.f);

    float J_r_inv[9];
    if (!mat3_inv(factor->J_r_bg, J_r_inv))
        return -1;                                /* singular -- bad data  */

    out->gyro_bias = mat_vmul(J_r_inv, neg_phi);

    /* ----------------------------------------------------------------
     * Step 2 — Gravity from delta_v
     *
     *   delta_v_corr = delta_v + J_v_bg * b_g + J_v_ba * b_a
     *   g_b          = delta_v_corr / T
     * ---------------------------------------------------------------- */
    v3_t dv_corr = b->dv;
    dv_corr = v3_add(dv_corr, mat_vmul(factor->J_v_bg, out->gyro_bias));
    dv_corr = v3_add(dv_corr, mat_vmul(factor->J_v_ba, b_a_prior));

    v3_t g_from_v = v3_scale(dv_corr, 1.f / T);

    /* ----------------------------------------------------------------
     * Step 3 — Gravity from delta_p
     *
     *   delta_p_corr = delta_p + J_p_bg * b_g + J_p_ba * b_a
     *   g_b          = 2 * delta_p_corr / T^2
     * ---------------------------------------------------------------- */
    v3_t dp_corr = b->dp;
    dp_corr = v3_add(dp_corr, mat_vmul(factor->J_p_bg, out->gyro_bias));
    dp_corr = v3_add(dp_corr, mat_vmul(factor->J_p_ba, b_a_prior));

    v3_t g_from_p = v3_scale(dp_corr, 2.f / (T * T));

    /* ----------------------------------------------------------------
     * Step 4 — Weighted combination
     * ---------------------------------------------------------------- */
    v3_t g_comb = v3_add(v3_scale(g_from_v, w_vel),
                         v3_scale(g_from_p, w_pos));

    /* ----------------------------------------------------------------
     * Step 5 — Enforce ||g_b|| = 9.80665 m/s^2
     * ---------------------------------------------------------------- */
    float norm = v3_norm(g_comb);
    if (norm > EPS)
        out->gravity_b = v3_scale(g_comb, GRAVITY / norm);
    else
        out->gravity_b = (v3_t){0.f, 0.f, GRAVITY};

    return 0;
}

/* ------------------------------------------------------------------ */
/* Residual diagnostics                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    float rotation_rad;   /* ||Log(delta_R) + J_r_bg * b_g||      */
    float velocity_ms;    /* ||delta_v_corr - g_b * T||            */
    float position_m;     /* ||delta_p_corr - 0.5 * g_b * T^2||   */
    float g_norm_err;     /* | ||g_comb|| - 9.80665 |              */
    float consistency;    /* ||g_from_v - g_from_p||               */
} imu_calib_residuals_t;

/**
 * @brief Computes diagnostic residuals for a stationary IMU calibration result.
 *
 * Applies first-order bias corrections to the preintegrated delta_v and delta_p,
 * then checks them against the gravity-only predictions for a stationary interval:
 *   - rotation_rad:  ||Log(dR) + J_r_bg * b_g||   (should be zero)
 *   - velocity_ms:   ||dv_corr - g_b * T||          (should be zero)
 *   - position_m:    ||dp_corr - 0.5*g_b*T^2||      (should be zero)
 *   - g_norm_err:    | ||blended g estimate|| - 9.80665 |
 *   - consistency:   ||g_from_v - g_from_p||         (agreement between v- and p-derived gravity)
 */
void imu_calib_get_residuals(const imu_factor_with_bias_t *factor,
                             v3_t                              b_a_prior,
                             const imu_calib_result_t         *calib,
                             imu_calib_residuals_t            *res)
{
    const imu_factor_t   *b = &factor->base;
    float T = b->dt;
    v3_t  bg = calib->gyro_bias;
    v3_t  gb = calib->gravity_b;

    /* rotation residual */
    v3_t phi     = so3_log(b->dR);
    v3_t J_r_bg_times_bg = mat_vmul(factor->J_r_bg, bg);
    v3_t rot_err = v3_add(phi, J_r_bg_times_bg);    /* should be zero */
    res->rotation_rad = v3_norm(rot_err);

    /* corrected delta_v and delta_p */
    v3_t dv_corr = v3_add(b->dv,
                   v3_add(mat_vmul(factor->J_v_bg, bg),
                          mat_vmul(factor->J_v_ba, b_a_prior)));

    v3_t dp_corr = v3_add(b->dp,
                   v3_add(mat_vmul(factor->J_p_bg, bg),
                          mat_vmul(factor->J_p_ba, b_a_prior)));

    /* velocity residual: dv_corr should equal g_b * T */
    v3_t pred_v = v3_scale(gb, T);
    res->velocity_ms = v3_norm(v3_sub(dv_corr, pred_v));

    /* position residual: dp_corr should equal 0.5 * g_b * T^2 */
    v3_t pred_p = v3_scale(gb, 0.5f * T * T);
    res->position_m = v3_norm(v3_sub(dp_corr, pred_p));

    /* gravity magnitude error */
    v3_t g_comb = v3_add(v3_scale(v3_scale(dv_corr, 1.f/T),  0.5f),
                         v3_scale(v3_scale(dp_corr, 2.f/(T*T)), 0.5f));
    res->g_norm_err = fabsf(v3_norm(g_comb) - GRAVITY);

    /* consistency: how much do the two gravity estimates agree */
    v3_t g_from_v = v3_scale(dv_corr, 1.f / T);
    v3_t g_from_p = v3_scale(dp_corr, 2.f / (T * T));
    res->consistency = v3_norm(v3_sub(g_from_v, g_from_p));
}


/* ------------------------------------------------------------------ */
/* Rotation integral matrices                                           */
/* ------------------------------------------------------------------ */
 
/*
 * A_v = integral_0^T Exp(b_g * t) dt
 *     = T*I  +  c1*[b_g]x  +  c2*[b_g]x²
 *
 * c1 = (1 - cos(w*T)) / w²
 * c2 = (w*T - sin(w*T)) / w³
 */
static void build_Av(v3_t bg, float T, float Av[9])
{
    float w  = v3_norm(bg);
    float S[9], S2[9], t1[9], t2[9], t3[9], tmp[9];
 
    mat_identity_scale(t1, T);
    skew(bg, S);
    mat_mul(S, S, S2);
 
    float c1, c2;
    if (w < EPS) {
        /* Taylor: c1 ~ T²/2,  c2 ~ T³/6 */
        c1 = 0.5f  * T * T;
        c2 = T * T * T / 6.f;
    } else {
        float wT = w * T;
        c1 = (1.f   - cosf(wT)) / (w * w);
        c2 = (wT    - sinf(wT)) / (w * w * w);
    }
 
    mat_scale(S,  c1, t2);
    mat_scale(S2, c2, t3);
    mat_add(t1, t2, tmp);
    mat_add(tmp, t3, Av);
}
 
/*
 * A_p = integral_0^T integral_0^t Exp(b_g * s) ds dt
 *     = T²/2*I  +  d1*[b_g]x  +  d2*[b_g]x²
 *
 * d1 = (w*T - sin(w*T)) / w³
 * d2 = (0.5*(w*T)² + cos(w*T) - 1) / w⁴
 */
static void build_Ap(v3_t bg, float T, float Ap[9])
{
    float w  = v3_norm(bg);
    float S[9], S2[9], t1[9], t2[9], t3[9], tmp[9];
 
    mat_identity_scale(t1, 0.5f * T * T);
    skew(bg, S);
    mat_mul(S, S, S2);
 
    float d1, d2;
    if (w < EPS) {
        /* Taylor: d1 ~ T³/6,  d2 ~ T⁴/24 */
        d1 = T * T * T / 6.f;
        d2 = T * T * T * T / 24.f;
    } else {
        float wT = w * T;
        float w2 = w*w, w3 = w2*w, w4 = w3*w;
        d1 = (wT - sinf(wT)) / w3;
        d2 = (0.5f * wT * wT + cosf(wT) - 1.f) / w4;
    }
 
    mat_scale(S,  d1, t2);
    mat_scale(S2, d2, t3);
    mat_add(t1, t2, tmp);
    mat_add(tmp, t3, Ap);
}


/**
 * imu_calib_stationary() - estimate gyro bias and gravity from a
 * stationary pre-integration window, without Jacobians.
 *
 * @factor:  raw pre-integration factor (dt, dp, dv, dR)
 * @w_vel:   weight for the velocity-derived gravity estimate  (0..1)
 * @w_pos:   weight for the position-derived gravity estimate  (0..1)
 *           w_vel + w_pos should equal 1.
 * @out:     calibration result
 *
 * Returns  0  on success.
 * Returns -1  if A_v is singular (numerically degenerate input).
 * Returns -2  if A_p is singular (numerically degenerate input).
 */
int imu_calib_stationary(const imu_factor_t *factor,
                         float                   w_vel,
                         float                   w_pos,
                         imu_calib_result_t     *out)
{
    float T = factor->dt;
 
    /* -------------------------------------------------------------- */
    /* Step 1 — Gyro bias                                              */
    /*   delta_R = Exp(b_g * T)  =>  b_g = Log(delta_R) / T          */
    /* -------------------------------------------------------------- */
    v3_t phi       = so3_log(factor->dR);
    out->gyro_bias = v3_scale(phi, 1.f / T);
 
    /* -------------------------------------------------------------- */
    /* Step 2 — Gravity from delta_v                                   */
    /*   delta_v = A_v * g_eff  =>  g_eff_v = A_v^{-1} * delta_v     */
    /* -------------------------------------------------------------- */
    float Av[9], Av_inv[9];
    build_Av(out->gyro_bias, T, Av);
    if (!mat3_inv(Av, Av_inv)) return -1;
 
    v3_t g_from_v = mat_vmul(Av_inv, factor->dv);
 
    /* -------------------------------------------------------------- */
    /* Step 3 — Gravity from delta_p                                   */
    /*   delta_p = A_p * g_eff  =>  g_eff_p = A_p^{-1} * delta_p     */
    /* -------------------------------------------------------------- */
    float Ap[9], Ap_inv[9];
    build_Ap(out->gyro_bias, T, Ap);
    if (!mat3_inv(Ap, Ap_inv)) return -2;
 
    v3_t g_from_p = mat_vmul(Ap_inv, factor->dp);
 
    /* -------------------------------------------------------------- */
    /* Step 4 — Weighted combination                                   */
    /* -------------------------------------------------------------- */
    v3_t g_comb = v3_add(v3_scale(g_from_v, w_vel),
                         v3_scale(g_from_p, w_pos));
 
    /* -------------------------------------------------------------- */
    /* Step 5 — Enforce ||g_b|| = 9.80665 m/s²                        */
    /* -------------------------------------------------------------- */
    float norm = v3_norm(g_comb);
    out->gravity_b = (norm > EPS)
                   ? v3_scale(g_comb, GRAVITY / norm)
                   : (v3_t){0.f, 0.f, GRAVITY};
 
    return 0;
}

void imu_make_zero_motion_factor(float                  T,
                                 point3D_float_t        bg,
                                 point3D_float_t        g_b,
                                 imu_factor_with_bias_t *out)
{
    memset(out, 0, sizeof(*out));
 
    /* ---- base measurements ---- */
    out->base.dt = T;
 
    /* dR = I  (zero true rotation) */
    out->base.dR[0] = 1.f;
    out->base.dR[4] = 1.f;
    out->base.dR[8] = 1.f;
 
    /* dv = g_b * T */
    out->base.dv = (point3D_float_t){ g_b.x * T, g_b.y * T, g_b.z * T };
 
    /* dp = 0.5 * g_b * T² */
    float h = 0.5f * T * T;
    out->base.dp = (point3D_float_t){ g_b.x * h, g_b.y * h, g_b.z * h };
 
    /* ---- Jacobians ---- */
 
    /* J_r_bg = -T * I
     * (right Jacobian of Log at phi=0 is I; the sensitivity of phi=b_g*T
     *  to b_g is T*I, with a sign flip because dR_corr = dR * Exp(J_r_bg*db_g)) */
    mat_identity_scale(out->J_r_bg, -T);
 
    /* J_v_bg = A_v - T*I   (bias-dependent deviation from the naive T*I term) */
    float Av[9], TI[9];
    build_Av(bg, T, Av);
    mat_identity_scale(TI, T);
    for (int i = 0; i < 9; i++)
        out->J_v_bg[i] = Av[i] - TI[i];
 
    /* J_v_ba = -T * I
     * (accelerometer bias integrated over the identity rotation trajectory) */
    mat_identity_scale(out->J_v_ba, -T);
 
    /* J_p_bg = A_p - T²/2 * I */
    float Ap[9], hI[9];
    build_Ap(bg, T, Ap);
    mat_identity_scale(hI, 0.5f * T * T);
    for (int i = 0; i < 9; i++)
        out->J_p_bg[i] = Ap[i] - hI[i];
 
    /* J_p_ba = -T²/2 * I */
    mat_identity_scale(out->J_p_ba, -0.5f * T * T);
}