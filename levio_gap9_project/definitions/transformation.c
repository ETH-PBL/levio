// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "transformation.h"

#include "math.h"

void composeTransformation(matrix_2D_t* R, float t[3], matrix_2D_t* T)
{
    for(uint8_t i = 0; i < 3; ++i)
    {
        T->data[i*4+0] = R->data[i*3+0];
        T->data[i*4+1] = R->data[i*3+1];
        T->data[i*4+2] = R->data[i*3+2];
        T->data[i*4+3] = t[i];
        T->data[3*4+i] = 0.0;
    }
    T->data[15] = 1.0;
}

void rotationMatrixOfTransformation(matrix_2D_t* T, matrix_2D_t* R)
{
    for(uint8_t i = 0; i < 3; ++i)
    {
        R->data[i*3+0] = T->data[i*4+0];
        R->data[i*3+1] = T->data[i*4+1];
        R->data[i*3+2] = T->data[i*4+2];
    }
    return;
}

void translationVectorOfTransformation(matrix_2D_t* T, float t[3])
{
    t[0] = T->data[0*4+3];
    t[1] = T->data[1*4+3];
    t[2] = T->data[2*4+3];
    return;
}

void scaleTranslationOfTransformation(matrix_2D_t* T, float scale)
{
    for(uint8_t i = 0; i < 3; ++i)
    {
        T->data[i*4+3] *= scale;
    }
}

void rodriguesToMatrix(const float* r, float* R) {
    float theta = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);

    if (theta < 1e-8f) {
        // If angle is close to zero, return the identity matrix
        R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
        R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
        R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
    } else {
        // Get the unit axis n
        float n[3] = { r[0]/theta, r[1]/theta, r[2]/theta };

        // Skew-symmetric matrix K
        float K[9] = {
            0.0f, -n[2],  n[1],
            n[2],  0.0f, -n[0],
           -n[1],  n[0],  0.0f
        };

        // K^2
        float K2[9];
        K2[0] = K[0]*K[0] + K[1]*K[3] + K[2]*K[6];
        K2[1] = K[0]*K[1] + K[1]*K[4] + K[2]*K[7];
        K2[2] = K[0]*K[2] + K[1]*K[5] + K[2]*K[8];
        K2[3] = K[3]*K[0] + K[4]*K[3] + K[5]*K[6];
        K2[4] = K[3]*K[1] + K[4]*K[4] + K[5]*K[7];
        K2[5] = K[3]*K[2] + K[4]*K[5] + K[5]*K[8];
        K2[6] = K[6]*K[0] + K[7]*K[3] + K[8]*K[6];
        K2[7] = K[6]*K[1] + K[7]*K[4] + K[8]*K[7];
        K2[8] = K[6]*K[2] + K[7]*K[5] + K[8]*K[8];

        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);
        float one_minus_cos = 1.0f - cos_theta;

        // R = I + sin(theta)*K + (1-cos(theta))*K^2
        R[0] = 1.0f + sin_theta * K[0] + one_minus_cos * K2[0];
        R[1] = 0.0f + sin_theta * K[1] + one_minus_cos * K2[1];
        R[2] = 0.0f + sin_theta * K[2] + one_minus_cos * K2[2];
        R[3] = 0.0f + sin_theta * K[3] + one_minus_cos * K2[3];
        R[4] = 1.0f + sin_theta * K[4] + one_minus_cos * K2[4];
        R[5] = 0.0f + sin_theta * K[5] + one_minus_cos * K2[5];
        R[6] = 0.0f + sin_theta * K[6] + one_minus_cos * K2[6];
        R[7] = 0.0f + sin_theta * K[7] + one_minus_cos * K2[7];
        R[8] = 1.0f + sin_theta * K[8] + one_minus_cos * K2[8];
    }
}

void matrixToRodrigues(const float* R, float* r) {
    float trace = R[0] + R[4] + R[8];
    float cos_theta = (trace - 1.0f) / 2.0f;
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, cos_theta))); // Clamp for safety

    if (theta < 1e-8f) {
        // Case 1: Angle is close to 0
        r[0] = 0.0f;
        r[1] = 0.0f;
        r[2] = 0.0f;
    } else if (fabsf(theta - M_PI) < 1e-6f) {
        // Case 2: Angle is close to 180 degrees (pi)
        // R is symmetric. Find the axis from a column of R + I
        float xx = (R[0] + 1.0f) / 2.0f;
        float yy = (R[4] + 1.0f) / 2.0f;
        float zz = (R[8] + 1.0f) / 2.0f;
        float xy = (R[1] + R[3]) / 4.0f;
        float xz = (R[2] + R[6]) / 4.0f;
        float yz = (R[5] + R[7]) / 4.0f;

        float n[3];
        if ((xx > yy) && (xx > zz)) { // xx is the largest
            n[0] = sqrtf(xx);
            n[1] = xy / n[0];
            n[2] = xz / n[0];
        } else if (yy > zz) { // yy is the largest
            n[1] = sqrtf(yy);
            n[0] = xy / n[1];
            n[2] = yz / n[1];
        } else { // zz is the largest
            n[2] = sqrtf(zz);
            n[0] = xz / n[2];
            n[1] = yz / n[2];
        }
        
        // The sign is ambiguous, but the resulting rotation is the same.
        r[0] = n[0] * theta;
        r[1] = n[1] * theta;
        r[2] = n[2] * theta;
    } else {
        // General case
        float sin_theta = sinf(theta);
        float factor = theta / (2.0f * sin_theta);

        r[0] = (R[7] - R[5]) * factor;
        r[1] = (R[2] - R[6]) * factor;
        r[2] = (R[3] - R[1]) * factor;
    }
}