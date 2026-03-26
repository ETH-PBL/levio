// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "work_memory.h"
#include "logging.h"

void* allocate_work_memory(work_memory_t* work_memory, size_t size)
{
    if(work_memory->size_left < size)
    {
        LOG_ERROR("\nOUT OF MEMORY!\n\n");
        return NULL;
    }
    void* cur_work_ptr = work_memory->memory_ptr;
    work_memory->memory_ptr += size;
    work_memory->size_left -= size;
    return cur_work_ptr;
}

void print_work_memory(work_memory_t* work_memory)
{
    printf("Memory Available %d, ptr %d \n",work_memory->size_left,work_memory->memory_ptr);
}

work_memory_t split_work_memory(work_memory_t* work_memory, uint8_t total_chunks, uint8_t current_id)
{
    uint32_t subset_size = work_memory->size_left/total_chunks;
    work_memory_t subset_work_memory = {work_memory->memory_ptr+current_id*subset_size,subset_size};
    return subset_work_memory;
}