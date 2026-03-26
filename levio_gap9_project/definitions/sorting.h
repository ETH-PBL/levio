// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __SORTING_H__
#define __SORTING_H__

#include "pmsis.h"

/**
 * @brief Sorts an array using the comb sort algorithm.
 *
 * This function sorts the array pointed to by `base` containing `nmemb` elements,
 * each of size `size` bytes, using the comb sort algorithm. The order of the sort
 * is determined by the comparison function `compar`.
 *
 * @param base Pointer to the first element of the array to be sorted.
 * @param nmemb Number of elements in the array.
 * @param size Size in bytes of each element in the array.
 * @param compar Pointer to a comparison function which returns a negative integer,
 *        zero, or a positive integer as the first argument is considered to be
 *        respectively less than, equal to, or greater than the second.
 */
void combsort(void *base, size_t nmemb, size_t size,
	   int (*compar) (const void *, const void *));

#endif // __SORTING_H__