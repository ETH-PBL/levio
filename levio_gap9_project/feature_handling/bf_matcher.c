// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "bf_matcher.h"

#define Abs(a)          (((int)(a)<0)?(-(a)):(a))
#define Min(a, b)       (((a)<(b))?(a):(b))

static inline uint8_t calculate_hamming_distance(orb_descriptor_t desc0, orb_descriptor_t desc1)
{
    uint8_t hamming_score = 0;
    hamming_score += __builtin_popcount(desc0[0] ^ desc1[0]);
    hamming_score += __builtin_popcount(desc0[1] ^ desc1[1]);
    hamming_score += __builtin_popcount(desc0[2] ^ desc1[2]);
    hamming_score += __builtin_popcount(desc0[3] ^ desc1[3]);
    hamming_score += __builtin_popcount(desc0[4] ^ desc1[4]);
    hamming_score += __builtin_popcount(desc0[5] ^ desc1[5]);
    hamming_score += __builtin_popcount(desc0[6] ^ desc1[6]);
    hamming_score += __builtin_popcount(desc0[7] ^ desc1[7]);
    return hamming_score;
}

uint16_t bf_match_max_flow(feature_match_t* matches,
                           orb_features_t* features0,
                           orb_features_t* features1,
                           uint16_t max_flow,
                           uint8_t hamming_threshold)
{
    uint16_t match_counter = 0;
    for(uint16_t idx1 = 0; idx1 < features1->kpt_counter; ++idx1)
    {
        uint8_t current_distance = hamming_threshold;
        /* Use length value to indicate no match */
        uint16_t matching_idx0 = features0->kpt_counter;
        for(uint16_t idx0 = 0; idx0 < features0->kpt_counter; ++idx0)
        {
            uint8_t new_distance = calculate_hamming_distance(features0->descs[idx0],features1->descs[idx1]);
            if(new_distance < current_distance)
            {
                point2D_u16_t kpt0 = features0->kpts[idx0];
                point2D_u16_t kpt1 = features1->kpts[idx1];
                if(max_flow == 0 || (Abs((int16_t)kpt0.x-kpt1.x) <= max_flow && Abs((int16_t)kpt0.y-kpt1.y) <= max_flow))
                {
                    current_distance = new_distance;
                    matching_idx0 = idx0;
                }
            }
        }
        if(matching_idx0 < features0->kpt_counter)
        {
            feature_match_t match = {matching_idx0, idx1, current_distance};
            matches[match_counter] = match;
            ++match_counter;
        }
    }
    return match_counter;
}

/**
 * @brief Removes one-way matches from a slice of the matches array.
 *
 * Retains only matches where no other match in the full set claims the same
 * feat_idx0 with an equal or better score. Matches are compacted in-place
 * starting at start_offset.
 *
 * @return Number of mutual matches remaining in the slice.
 */
uint16_t two_way_filter(feature_match_t* matches,
                        uint16_t match_counter,
                        uint16_t start_offset,
                        uint16_t slice_size)
{
    uint16_t two_way_match_counter = 0;
    for (uint16_t i = start_offset; i < start_offset+slice_size; i++)
    {
        feature_match_t match = matches[i];
        uint8_t keep = 1;
        for (uint16_t j = 0; j < match_counter; j++)
        {
            if(i == j)
            {
                continue;
            }
            feature_match_t match_iter = matches[j];
            /* Comparing for equal or bigger score can lead to different results between single and multicore execution */
            if(match.feat_idx0 == match_iter.feat_idx0 && match.match_score >= match_iter.match_score)
            {
                keep = 0;
            }
        }
        if (keep == 1)
        {
            matches[start_offset+two_way_match_counter] = match;
            two_way_match_counter++;
        }
    }
    return two_way_match_counter;
}

uint16_t bf_match_two_way_max_flow(feature_match_t* matches,
                                   orb_features_t* features0,
                                   orb_features_t* features1,
                                   uint16_t max_flow,
                                   uint8_t hamming_threshold)
{
    uint16_t match_counter = bf_match_max_flow(matches, features0, features1, 
                                               max_flow, hamming_threshold);
    LOG_INFO("Match counter %d\n",match_counter);
    uint16_t two_way_match_counter = two_way_filter(matches, match_counter, 0 , match_counter);
    LOG_INFO("Two Way Match counter %d\n",two_way_match_counter);
    return two_way_match_counter;
}

typedef struct bf_matcher_arguments
{
    feature_match_t* matches;
    orb_features_t* features0;
    orb_features_t* features1;
    uint16_t* match_counter;
    uint16_t slice_size;
    uint16_t max_flow;
    uint8_t hamming_threshold;
    uint8_t nb_cores;
} bf_matcher_arguments_t;

/**
 * @brief Compacts per-core match results into a single contiguous array.
 *
 * After a parallel bf_match_max_flow, each core wrote its matches starting at
 * core_id * slice_size. This function moves them into a packed layout starting
 * at matches[0], preserving core-0 results in place and shifting the rest.
 *
 * @return Total number of matches across all cores.
 */
uint16_t reorganize_matches_memory(feature_match_t* matches, uint16_t* match_counters, uint16_t slice_size, uint8_t nb_cores)
{
    uint16_t match_counter = match_counters[0];
    for(uint8_t index = 1; index < nb_cores; ++index)
    {
        uint16_t nb_match_pool = match_counters[index];
        uint16_t offset = slice_size*index;
        for (uint8_t match_nr = 0; match_nr < nb_match_pool; ++match_nr)
        {
            matches[match_counter] = matches[offset+match_nr];
            ++match_counter;
        }
    }
    return match_counter;
}

/**
 * @brief Cluster kernel: each core runs bf_match_max_flow on its assigned slice of features1.
 *
 * The features1 keypoints are divided into equal slices across nb_cores. Each core
 * writes its matches into matches[core_id * slice_size] and records the count.
 */
void bf_match_max_flow_subset(void* args)
{
    bf_matcher_arguments_t* bfm_args = (bf_matcher_arguments_t*) args;
    uint16_t core_id = pi_core_id();
	uint16_t nb_cores = bfm_args->nb_cores;
    uint16_t nb_features1 = bfm_args->features1->kpt_counter;
    uint16_t slice_size = bfm_args->slice_size;
    uint32_t offset = slice_size*core_id;
    if (offset >= nb_features1)
    {
        bfm_args->match_counter[core_id] = 0;
        return;
    }
    slice_size = Min(nb_features1-offset, slice_size);
    orb_features_t temp_features1;
    temp_features1.kpts = bfm_args->features1->kpts + offset;
    temp_features1.descs = bfm_args->features1->descs + offset;
    temp_features1.kpt_counter = slice_size;
    bfm_args->match_counter[core_id] = bf_match_max_flow(bfm_args->matches+offset,
                                                         bfm_args->features0,
                                                         &temp_features1,
                                                         bfm_args->max_flow,
                                                         bfm_args->hamming_threshold);
    for (uint16_t i = 0; i < bfm_args->match_counter[core_id]; ++i)
    {
        /* Re-Index Matches */
        bfm_args->matches[offset+i].feat_idx1 += offset;
    }
}

typedef struct two_way_filter_arguments
{
    feature_match_t* matches;
    uint16_t* match_counter;
    uint16_t total_one_way_matches;
    uint16_t slice_size;
    uint8_t nb_cores;
} two_way_filter_arguments_t;

/**
 * @brief Cluster kernel: each core applies two_way_filter to its assigned slice of matches.
 *
 * The total one-way matches are divided into equal slices across nb_cores. Each core
 * filters its slice against the full match set and writes results back in-place,
 * recording the surviving count in match_counter[core_id].
 */
void two_way_filter_subset(void* args)
{
    two_way_filter_arguments_t* twf_args = (two_way_filter_arguments_t*) args;
    uint16_t core_id = pi_core_id();
	uint16_t nb_cores = twf_args->nb_cores;
    uint16_t slice_size = twf_args->slice_size;
    uint32_t offset = slice_size*core_id;
    if (offset >= twf_args->total_one_way_matches)
    {
        twf_args->match_counter[core_id] = 0;
        return;
    }
    slice_size = Min(twf_args->total_one_way_matches-offset, slice_size);
    twf_args->match_counter[core_id] = two_way_filter(twf_args->matches,
                                                      twf_args->total_one_way_matches,
                                                      offset,
                                                      slice_size);
}

uint16_t bf_match_two_way_max_flow_multicore(feature_match_t* matches,
                                             orb_features_t* features0,
                                             orb_features_t* features1,
                                             uint16_t max_flow,
                                             uint8_t hamming_threshold,
                                             uint8_t nb_cores)
{
    uint16_t slice_size;
    if(features1->kpt_counter % nb_cores == 0)
    {
        slice_size = features1->kpt_counter/nb_cores;
    }
    else
    {
        slice_size = features1->kpt_counter/nb_cores+1;
    }
    uint16_t utility_counters[8];
    bf_matcher_arguments_t bfm_args = {matches, features0, features1, utility_counters, slice_size, max_flow, hamming_threshold, nb_cores};
    pi_cl_team_fork(nb_cores, bf_match_max_flow_subset, &bfm_args);
    uint16_t match_counter = reorganize_matches_memory(matches, utility_counters, slice_size, nb_cores);
    LOG_INFO("Match counter %d\n",match_counter);

    if(match_counter % nb_cores == 0)
    {
        slice_size = match_counter/nb_cores;
    }
    else
    {
        slice_size = match_counter/nb_cores+1;
    }
    two_way_filter_arguments_t twf_args = {matches, utility_counters, match_counter, slice_size, nb_cores};
    pi_cl_team_fork(nb_cores, two_way_filter_subset, &twf_args);
    uint16_t two_way_match_counter = reorganize_matches_memory(matches, utility_counters, slice_size, nb_cores);
    LOG_INFO("Two Way Match counter %d\n",two_way_match_counter);
    return two_way_match_counter;
}