// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __WORK_MEMORY_H__
#define __WORK_MEMORY_H__

#include "pmsis.h"

typedef struct{
    void* memory_ptr;
    uint32_t size_left;
} work_memory_t;


/**
 * @brief Allocates a block of memory from the given work memory structure.
 *
 * @param work_memory Pointer to the work_memory_t structure managing the memory pool.
 * @param size The size of the memory block to allocate, in bytes.
 * @return Pointer to the allocated memory block, or NULL if allocation fails.
 */
void* allocate_work_memory(work_memory_t* work_memory, size_t size);

/**
 * @brief Prints the available memory and current pointer of the work memory structure.
 *
 * @param work_memory Pointer to the work_memory_t structure to be printed.
 */
void print_work_memory(work_memory_t* work_memory);
 
/**
 * @brief Splits the work memory into multiple chunks and returns the chunk corresponding to the current ID.
 *
 * @param work_memory Pointer to the work_memory_t structure to split.
 * @param total_chunks The total number of chunks to split the memory into.
 * @param current_id The ID of the current chunk to retrieve (0-based index).
 * @return The work_memory_t structure representing the current chunk.
 */
work_memory_t split_work_memory(work_memory_t* work_memory, uint8_t total_chunks, uint8_t current_id);


#endif /* __WORK_MEMORY_H__ */