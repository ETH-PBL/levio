// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "prng.h"

#define MAX_RAND 0xFFFFFFFFu

uint32_t g_lfsr_state = 0x356167u;

static uint32_t lfsr_step(uint32_t reg)
{
  // From https://www.schneier.com/paper-pseudorandom-sequence.html, example 1
  // This LFSR has a period of 2^32-1

  /*Register should be initialized with some random value.*/
  reg = ((((reg >> 31)  /*Shift the 32nd bit to the first
                                    bit*/
             ^ (reg >> 6)    /*XOR it with the seventh bit*/
             ^ (reg >> 4)    /*XOR it with the fifth bit*/
             ^ (reg >> 2)    /*XOR it with the third bit*/
             ^ (reg >> 1)    /*XOR it with the second bit*/
             ^ reg)          /*and XOR it with the first bit.*/
             & 0x0000001)         /*Strip all the other bits off and*/
             <<31)                /*move it back to the 32nd bit.*/
             | (reg >> 1);   /*Or with the register shifted
                                    right.*/
  return reg;
}

uint32_t rand()
{
    g_lfsr_state = lfsr_step(g_lfsr_state);
    return g_lfsr_state;
}

uint32_t rand_r(uint32_t * seedp)
{
    *seedp = lfsr_step(*seedp);
    return *seedp;
}

uint32_t rand_range(uint32_t range)
{
    if (range <= 1) return 0;
    
    // Calculate the largest multiple of range that fits in uint32_t
    uint32_t max_valid = MAX_RAND - (MAX_RAND % range);
    uint32_t value;
    
    while((value = rand()) >= max_valid);
    
    return (value % range);
}

void srand(uint32_t seed)
{
    g_lfsr_state = seed;
}

uint32_t * generate_random_data(uint32_t * rnd_state, uint32_t data_nb, uint32_t * buffer)
{
    uint32_t * pcur = (uint32_t *)buffer;

    for (uint32_t i = 0; i < data_nb; i++)
    {
        *pcur = rand_r(rnd_state);
        pcur++;
    }

    return pcur;
}