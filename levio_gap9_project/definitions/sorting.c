// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "sorting.h"

static inline void memswap(void *m1, void *m2, size_t n)
{
	uint8_t* p = m1;
	uint8_t* q = m2;
	uint8_t tmp[n];

	memcpy(tmp, m1, n);
    memcpy(m1, m2, n);
    memcpy(m2, tmp, n);
}

static inline size_t newgap(size_t gap)
{
	gap = (gap * 10) / 13;
	if (gap == 9 || gap == 10)
		gap = 11;

	if (gap < 1)
		gap = 1;
	return gap;
}

void combsort(void *base, size_t nmemb, size_t size,
	   int (*compar) (const void *, const void *))
{
	size_t gap = nmemb;
	size_t i, j;
	char *p1, *p2;
	int swapped;

	if (!nmemb)
		return;

	do {
		gap = newgap(gap);
		swapped = 0;

		for (i = 0, p1 = base; i < nmemb - gap; i++, p1 += size) {
			j = i + gap;
			if (compar(p1, p2 = (char *)base + j * size) > 0) {
				memswap(p1, p2, size);
				swapped = 1;
			}
		}
	} while (gap > 1 || swapped);
}