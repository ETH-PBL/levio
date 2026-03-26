// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __TRANSFORMATION_H__
#define __TRANSFORMATION_H__

#include "pmsis.h"
#include "matrix.h"


/**
 * @brief Composes a transformation matrix from a rotation matrix and a translation vector.
 * 
 * @param R Pointer to the 3x3 input rotation matrix (matrix_2D_t*).
 * @param t Input translation vector (float[3]).
 * @param T Pointer to the 4x4 output transformation matrix (matrix_2D_t*).
 */
void composeTransformation(matrix_2D_t* R, float t[3], matrix_2D_t* T);

/**
 * @brief Extracts the rotation matrix from a transformation matrix.
 * 
 * @param T Pointer to the 4x4 input transformation matrix (matrix_2D_t*).
 * @param R Pointer to the 3x3 output rotation matrix (matrix_2D_t*).
 */
void rotationMatrixOfTransformation(matrix_2D_t* T, matrix_2D_t* R);

/**
 * @brief Extracts the translation vector from a transformation matrix.
 * 
 * @param T Pointer to the 4x4 input transformation matrix (matrix_2D_t*).
 * @param t Output translation vector (float[3]).
 */
void translationVectorOfTransformation(matrix_2D_t* T, float t[3]);

/**
 * @brief Scales the translation component of a transformation matrix by a given factor.
 * 
 * @param T Pointer to the 4x4 transformation matrix to be modified (matrix_2D_t*).
 * @param scale Scaling factor for the translation component.
 */
void scaleTranslationOfTransformation(matrix_2D_t* T, float scale);

/**
 * @brief Converts a Rodrigues vector to a 3x3 rotation matrix.
 *
 * @param r Pointer to a 3-element float array representing the Rodrigues vector.
 * @param R Pointer to a 9-element float array to store the resulting rotation matrix (row-major).
 */
void rodriguesToMatrix(const float* r, float* R);

/**
 * @brief Converts a 3x3 rotation matrix to a Rodrigues vector.
 *
 * @param R Pointer to a 9-element float array representing the rotation matrix (row-major).
 * @param r Pointer to a 3-element float array to store the resulting Rodrigues vector.
 */
void matrixToRodrigues(const float* R, float* r);


#endif /* __TRANSFORMATION_H__ */