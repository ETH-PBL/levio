// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __VO__H__
#define __VO__H__

#include "pmsis.h"
#include "definitions/config.h"
#include "definitions/type_definitions.h"

/**
 * @brief Processes the visual odometry pipeline.
 *
 * This function executes the visual odometry pipeline using the provided image buffer
 * and working memory, updating the pose graph statistics as needed.
 *
 * @param graph_stats Pointer to a pose_graph_stats_t structure where the function will
 *        store or update statistics related to the pose graph.
 * @param img_l2_buffer Pointer to the image buffer (in L2 memory) containing
 *        the input image data for processing.
 * @param work_memory Working memory structure required for intermediate computations
 *        during the pipeline execution.
 *
 * @return uint8_t Returns a status code indicating whether the processed image was
 *         added to the pose graph as a key frame (1) or not (0).
 */
uint8_t process_vo_pipeline(pose_graph_stats_t* graph_stats,
                            uint8_t* img_l2_buffer,
                            work_memory_t work_memory);

#endif /* __VO__H__ */