// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __PRNG_H__
#define __PRNG_H__

#include "pmsis.h"

/**
 * @brief Generates a pseudo-random 32-bit unsigned integer.
 * 
 * @return A pseudo-random value of type uint32_t.
 */
uint32_t rand();
 
/**
 * @brief Generates a pseudo-random number in the range [0, range).
 * 
 * @param range The exclusive upper bound for the random number.
 * @return A pseudo-random value in the range [0, range).
 */
uint32_t rand_range(uint32_t range);
 
/**
 * @brief Seeds the pseudo-random number generator.
 * 
 * @param seed The seed value to initialize the random number generator.
 */
void srand(uint32_t seed);

#endif // __PRNG_H__