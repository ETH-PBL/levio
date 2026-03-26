// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "orb.h"
#include "math.h"

#include "orb_bit_pattern.h"

#define Abs(a)          (((int)(a)<0)?(-(a)):(a))
#define Min(a, b)       (((a)<(b))?(a):(b))
#define Max(a, b)       (((a)>(b))?(a):(b))
#define Div256_Clip15(a)  ((a<0)?(-Min((-a+128)/256,15)):(Min((a+128)/256,15)))


int16_t* fast_offsets;
int16_t* harris_offsets;

/**
 * @brief Precomputes linear pixel offsets for the 16-point FAST corner detector ring.
 *
 * Stores flattened offsets (row*step + col) for each of the 16 pixels on the
 * Bresenham circle of radius 3 around a candidate corner. `step` is the image row stride.
 */
void precompute_fast_offsets(int16_t step)
{
    int16_t offsets[16] =   {(+0*step)-3,(+1*step)-3,(+2*step)-2,(+3*step)-1,
                             (+3*step)+0,(+3*step)+1,(+2*step)+2,(+1*step)+3,
                             (+0*step)+3,(-1*step)+3,(-2*step)+2,(-3*step)+1,
                             (-3*step)+0,(-3*step)-1,(-2*step)-2,(-1*step)-3};
    for(uint8_t i = 0; i < 16; ++i)
    {
        fast_offsets[i] = offsets[i];
    }
}

void precompute_harris_offsets(int16_t step)
{
    uint8_t block_size = HARRIS_BLOCK_SIZE;
    for(uint8_t k = 0; k < block_size*block_size; ++k)
    {
        harris_offsets[k] = ((int16_t)k%block_size-block_size/2) + ((int16_t)k/block_size-block_size/2)*step;
    }
}

/**
 * @brief Evaluates the FAST corner response at a pixel and returns its score.
 *
 * Checks whether 9 or more consecutive pixels on the 16-pixel Bresenham ring
 * are all brighter or all darker than the center by more than fast_threshold.
 * Wrap-around is handled by re-checking the first 8 pixels after the full ring.
 * The score is the sum of absolute intensity differences on the ring (used
 * for Harris-based filtering downstream), or 0 if not a corner.
 */
uint16_t calculate_fast_score(image_data_t* img, uint16_t x, uint16_t y, uint8_t fast_threshold)
{
    uint8_t* center_pointer = img->pixels+y*img->width+x;
    int16_t center = (int16_t)center_pointer[0];
    uint16_t sum = 0;
    uint8_t is_bigger[16];
    uint8_t is_smaller[16];
    uint8_t bigger = 0;
    uint8_t smaller = 0;
    uint8_t is_fast_corner = 0;
    for(uint8_t i = 0; i < 16; ++i)
    {
        int16_t val = ((int16_t)center_pointer[fast_offsets[i]])-center;
        sum += Abs(val);
        is_bigger[i] = (val > fast_threshold);
        is_smaller[i] = (val < -fast_threshold);
        bigger = is_bigger[i]*(bigger+1);
        smaller = is_smaller[i]*(smaller+1);
        is_fast_corner |= ((smaller == 9) || (bigger ==9));
    }
    /* Deal with wrap around */
    for(uint8_t i = 0; i < 8; ++i)
    {
        bigger = is_bigger[i]*(bigger+1);
        smaller = is_smaller[i]*(smaller+1);
        is_fast_corner |= ((smaller == 9) || (bigger ==9));
    }
    return is_fast_corner * sum;
}

/**
 * @brief Computes the Harris corner response score at a pixel.
 *
 * Sums the squared image gradients (Ix^2, Iy^2, IxIy) over a HARRIS_BLOCK_SIZE x HARRIS_BLOCK_SIZE
 * window using 5-tap finite differences. The response is:
 *   R = det(M) - 0.04 * trace(M)^2  (k=1/25 is the Harris factor)
 * where M = [[a, c], [c, b]] is the structure tensor.
 * Intermediate values are right-shifted to stay within int32_t range.
 */
int32_t calculate_harris_score(image_data_t* img, uint16_t x, uint16_t y)
{
    /* Open CV Version with block size 7 */
    int32_t a = 0; int32_t b = 0; int32_t c = 0;
    uint8_t block_size_sq = (HARRIS_BLOCK_SIZE*HARRIS_BLOCK_SIZE);
    int32_t step = (int32_t)img->width;
    for(uint8_t k = 0; k < block_size_sq; ++k)
    {
        const uint8_t* ptr = img->pixels + y*step + x + harris_offsets[k];
        /* [+/-4*255] */
        int16_t Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
        int16_t Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
        /* [+/-49*4*4*255*255 < +/-2^(6+2+2+8+8) < +/-2^26] */
        a += (int32_t)Ix*Ix;
        b += (int32_t)Iy*Iy;
        c += (int32_t)Ix*Iy;
    }
    /* 2^(26-11) = 2^(15) */
    a = a/(1 << 11);
    b = b/(1 << 11);
    c = c/(1 << 11);
    /* 2^15*2^15 = 2^30 */
    /* Division by 25 is equivalent to a harris factor of 0.04 */
    /* Original harris is scaled (by the patch size and input), this implementation uses
       the int32_t value range */
    int32_t R = a * b - c * c - ((a + b) * (a + b))/25;
    /* Scaled and rounded version of the Open CV computation */
    return R;
}

void initialize_orb(int16_t* fast_offsets_storage, int16_t* harris_offsets_storage, uint16_t image_width)
{
    fast_offsets = fast_offsets_storage;
    harris_offsets = harris_offsets_storage;
    precompute_fast_offsets(image_width);
    precompute_harris_offsets(image_width);
}

/**
 * @brief Detects up to features_per_patch ORB keypoints within a single spatial patch.
 *
 * Scans all pixels in [start_row,end_row) x [start_column,end_column). For each
 * FAST corner, the Harris score is computed and only the top-scoring candidates
 * are retained (maintained as a fixed-size min-heap by score). Results are written
 * into features->kpts starting at feature_offset.
 *
 * @return Number of valid keypoints found in this patch.
 */
uint16_t run_orb_detector_on_patch(image_data_t* img, orb_features_t* features, uint16_t feature_offset, uint16_t start_row, uint16_t end_row, uint16_t start_column, uint16_t end_column, uint8_t features_per_patch, uint8_t fast_threshold)
{
    uint16_t keypoint_counter = 0;
    int32_t harris_scores[features_per_patch];
    int32_t smallest_score = 0;
    uint8_t smallest_score_index = 0;
    memset(harris_scores,0,features_per_patch*sizeof(uint32_t));
    for(uint16_t y = start_row; y < end_row; ++y)
    {
        for(uint16_t x = start_column; x < end_column; ++x)
        {
            uint16_t fast_score = calculate_fast_score(img, x, y, FAST_THRESHOLD);
            if(fast_score > 0)
            {
                int32_t harris_score = calculate_harris_score(img, x, y);
                if(harris_score > HARRIS_THRESHOLD && harris_score > smallest_score)
                {
                    features->kpts[feature_offset+smallest_score_index].x = x;
                    features->kpts[feature_offset+smallest_score_index].y = y;
                    harris_scores[smallest_score_index] = harris_score;
                    smallest_score = harris_score;
                    for(uint8_t i = 0; i < features_per_patch; ++i)
                    {
                        if(harris_scores[i] < smallest_score)
                        {
                            smallest_score = harris_scores[i];
                            smallest_score_index = i;
                        }
                    }
                }
            }
        }
    }
    for(uint8_t i = 0; i < features_per_patch; ++i)
    {
        if(harris_scores[i] > 0)
        {
            keypoint_counter++;
        }
    }
    return keypoint_counter;
}

void run_orb_detector_spatial(image_data_t* img, orb_features_t* features, uint16_t start_bin, uint16_t end_bin, uint8_t fast_threshold)
{
    uint16_t keypoint_counter = 0;
    uint16_t fast_counter = 0;
    uint16_t x_bins = (img->width - 2*(ORB_PATCH_RADIUS+1)) / PATCH_SIZE;
    uint16_t y_bins = (img->height - 2*(ORB_PATCH_RADIUS+1)) / PATCH_SIZE;
    uint16_t total_bins = x_bins * y_bins;

    for(uint16_t idx = start_bin; idx < end_bin; ++idx)
    {
        uint16_t y = idx / x_bins;
        uint16_t x = idx % x_bins;
        uint16_t start_r = ORB_PATCH_RADIUS+1+y*PATCH_SIZE;
        uint16_t start_c = ORB_PATCH_RADIUS+1+x*PATCH_SIZE;
        keypoint_counter += run_orb_detector_on_patch(img, features, keypoint_counter, start_r, start_r+PATCH_SIZE, start_c, start_c+PATCH_SIZE,FEATURES_PER_PATCH,fast_threshold);
    }
    LOG_INFO("Keypoint counter %d\n",keypoint_counter);
    features->kpt_counter = keypoint_counter;
}

void apply_gaussian_blurr(image_data_t* img_in, image_data_t* img_out, uint16_t start_row, uint16_t end_row)
{
    uint8_t gauss_c[25] =  { 1,  4,  6,  4, 1,
                             4, 16, 24, 16, 4,
                             6, 24, 36, 24, 6,
                             4, 16, 24, 16, 4,
                             1,  4,  6,  4, 1};
    uint16_t scalar = 256;

    uint16_t step = img_in->width;
    uint16_t height = img_in->height;
    uint16_t start_y = start_row;
    uint16_t end_y = end_row;
    if(start_y < 2)
    {
        for(uint16_t x = 0; x < img_in->width; ++x)
        {
            /* Deal with top boundaries */
            img_out->pixels[0*step+x] = img_in->pixels[0*step+x];
            img_out->pixels[1*step+x] = img_in->pixels[1*step+x];
        }
        start_y = 2;
    }
    if(end_y >= img_in->height-2)
    {
        for(uint16_t x = 0; x < img_in->width; ++x)
        {
            /* Deal with bottom boundaries */
            img_out->pixels[(height-2)*step+x] = img_in->pixels[(height-2)*step+x];
            img_out->pixels[(height-1)*step+x] = img_in->pixels[(height-1)*step+x];
        }
        end_y = img_in->height-2;
    }
    for(uint16_t y = start_y; y < end_y; ++y)
    {
        int32_t offset = y*step;
        /* Deal with left and right boundaries */
        img_out->pixels[offset+0] = img_in->pixels[offset+0];
        img_out->pixels[offset+1] = img_in->pixels[offset+1];
        img_out->pixels[offset+step-2] = img_in->pixels[offset+step-2];
        img_out->pixels[offset+step-1] = img_in->pixels[offset+step-1];
        for(uint16_t x = 2; x < (img_in->width-2); ++x)
        {
            uint8_t* ptr_in = img_in->pixels+x+y*step;
            uint16_t filtered_value = 0;
            for(uint8_t k = 0; k < 5; ++k)
            {
                filtered_value +=    gauss_c[0+k*5]*ptr_in[(-2+k)*step-2]
                                    +gauss_c[1+k*5]*ptr_in[(-2+k)*step-1]
                                    +gauss_c[2+k*5]*ptr_in[(-2+k)*step+0]
                                    +gauss_c[3+k*5]*ptr_in[(-2+k)*step+1]
                                    +gauss_c[4+k*5]*ptr_in[(-2+k)*step+2];
            }
            img_out->pixels[x+y*step] = (uint8_t)(filtered_value/scalar);
        }
    }
}

void calculate_orb_decriptor(image_data_t* blurred_img, orb_features_t* features)
{
    for(uint16_t i = 0; i < features->kpt_counter; ++i)
    {
        /* Determine the dominant orientation */
        int32_t sum_ix = 0;
        int32_t sum_iy = 0;
        int16_t step = blurred_img->width;
        uint8_t* base_ptr = blurred_img->pixels + features->kpts[i].x + features->kpts[i].y*step;
        for(int16_t dy = -ORB_PATCH_RADIUS+1; dy < ORB_PATCH_RADIUS; ++dy)
        {
            for(int16_t dx = -ORB_PATCH_RADIUS+1; dx < ORB_PATCH_RADIUS; ++dx)
            {
                if((dx*dx + dy*dy) < (ORB_PATCH_RADIUS*ORB_PATCH_RADIUS))
                {
                    sum_ix += dx * base_ptr[dy*step+dx];
                    sum_iy += dy * base_ptr[dy*step+dx];
                }
            }
        }
        float alpha = atan2f((float)sum_iy,(float)sum_ix);
        float a_float = cosf(alpha);
        float b_float = sinf(alpha);
        int16_t a = a_float*256;
        int16_t b = b_float*256;

        for(uint8_t idx = 0; idx < 8; ++idx)
        {
            uint32_t sub_desc = 0;
            for(uint8_t sub_idx = 0; sub_idx < 32; ++sub_idx)
            {
                int8_t x0_o = bit_pattern_31_[(idx*32+sub_idx)*4+0];
                int8_t y0_o = bit_pattern_31_[(idx*32+sub_idx)*4+1];
                int8_t x1_o = bit_pattern_31_[(idx*32+sub_idx)*4+2];
                int8_t y1_o = bit_pattern_31_[(idx*32+sub_idx)*4+3];
                int8_t y0 = Div256_Clip15(x0_o*b + y0_o*a);
                int8_t x0 = Div256_Clip15(x0_o*a - y0_o*b);
                int8_t x1 = Div256_Clip15(x1_o*a - y1_o*b);
                int8_t y1 = Div256_Clip15(x1_o*b + y1_o*a);
                uint8_t t0 = base_ptr[y0*step+x0];
                uint8_t t1 = base_ptr[y1*step+x1];
                sub_desc |= ((t0<t1) << sub_idx);
            }
            features->descs[i][idx] = sub_desc;
        }
    }
}