// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __BF_FEATURE_MATCHER_H__
#define __BF_FEATURE_MATCHER_H__

#include "pmsis.h"
#include "orb.h"
#include "definitions/type_definitions.h"

/**
 * @brief Performs brute-force matching between two sets of ORB features with a maximum flow constraint.
 *
 * @param matches            Pointer to an array where the resulting feature matches will be stored.
 * @param features0          Pointer to the first set of ORB features.
 * @param features1          Pointer to the second set of ORB features.
 * @param max_flow           Maximum displacement of two features (max flow).
 * @param hamming_threshold  Maximum allowed Hamming distance for a valid match.
 * @return                   The number of matches found.
 */
uint16_t bf_match_max_flow(feature_match_t* matches,
                           orb_features_t* features0,
                           orb_features_t* features1,
                           uint16_t max_flow,
                           uint8_t hamming_threshold);

/**
 * @brief Performs two-way brute-force matching between two sets of ORB features with a maximum flow constraint.
 *
 * This function ensures that matches are mutual (i.e., feature A matches feature B and vice versa).
 *
 * @param matches            Pointer to an array where the resulting feature matches will be stored.
 * @param features0          Pointer to the first set of ORB features.
 * @param features1          Pointer to the second set of ORB features.
 * @param max_flow           Maximum displacement of two features (max flow).
 * @param hamming_threshold  Maximum allowed Hamming distance for a valid match.
 * @return                   The number of mutual matches found.
 */
uint16_t bf_match_two_way_max_flow(feature_match_t* matches,
                                   orb_features_t* features0,
                                   orb_features_t* features1,
                                   uint16_t max_flow,
                                   uint8_t hamming_threshold);

/**
 * @brief Performs two-way brute-force matching with a maximum flow constraint using multicore processing.
 *
 * This function is similar to bf_match_two_way_max_flow but utilizes multiple cores for parallel processing.
 *
 * @param matches            Pointer to an array where the resulting feature matches will be stored.
 * @param features0          Pointer to the first set of ORB features.
 * @param features1          Pointer to the second set of ORB features.
 * @param max_flow           Maximum displacement of two features (max flow).
 * @param hamming_threshold  Maximum allowed Hamming distance for a valid match.
 * @param nb_cores           Number of available cores for the parallel execution.
 * @return                   The number of mutual matches found.
 */
uint16_t bf_match_two_way_max_flow_multicore(feature_match_t* matches,
                                             orb_features_t* features0,
                                             orb_features_t* features1,
                                             uint16_t max_flow,
                                             uint8_t hamming_threshold,
                                             uint8_t nb_cores);


#endif /* __BF_FEATURE_MATCHER_H__ */