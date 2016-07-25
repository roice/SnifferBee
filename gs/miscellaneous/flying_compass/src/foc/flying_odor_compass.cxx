/*
 * Flying Odor Compass
 *
 * This technique digs information from three MOX gas sensors which are
 * equally spaced under propellers of a quadrotor
 *
 * Author:
 *      Roice Luo
 * Date:
 *      2016.06.17
 */
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

#include "foc/flying_odor_compass.h"
#include "foc/noise_suppression.h"

#include <samplerate.h> // require libsamplerate

#define SIGN(n) (n >= 0? 1:-1)

/* The sensors are placed at vertices of a regular polygon
 * Example: 3 sensors
 *
 *              sensor_1   --------
 *                /  \        R
 *               /    \    --------
 *              /      \
 *      sensor_2 ------ sensor_3
 */

Flying_Odor_Compass::Flying_Odor_Compass(void)
{
    // data
    foc_input.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    foc_ukf_out.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    foc_interp_out.reserve(FOC_RECORD_LEN*FOC_MOX_INTERP_FREQ);
    foc_diff_out.reserve(FOC_RECORD_LEN*FOC_MOX_INTERP_FREQ);
/* init UKF filtering */
    noise_suppression_ukf_init();
}

void Flying_Odor_Compass::update(FOC_Input_t& new_in)
{
    foc_input.push_back(new_in); // save record
/* Step 0: Pre-processing */
    
/* Step 1: UKF filtering */
    FOC_Reading_t ukf_out = noise_suppression_ukf_update(new_in);
    // save record
    foc_ukf_out.push_back(ukf_out);

/* Step 3 */

#if 0
    // sample rate converter
    int error;
    SRC_STATE* rate_conv = src_new(SRC_SINC_BEST_QUALITY, 1, &error);

    // sample rate convert data struct
    SRC_DATA src_data;
    src_data.src_ratio = (float)FOC_MOX_INTERP_FREQ/(float)FOC_MOX_DAQ_FREQ;
    src_data.input_frames = foc_ukf_out.size();
    src_data.output_frames = src_data.input_frames*src_data.src_ratio;
    src_data.end_of_input = 1;

    // interpolation input and output array
    float* interp_in[FOC_NUM_SENSORS];
    float* interp_out[FOC_NUM_SENSORS];

    // convert (interpolate)
    for (int sidx = 0; sidx < FOC_NUM_SENSORS; sidx++) {
        interp_in[sidx] = (float*)malloc(src_data.input_frames*sizeof(float));
        interp_out[sidx] = (float*)malloc(src_data.output_frames*sizeof(float));

        // prepare input
        for (int i = 0; i < src_data.input_frames; i++)
            interp_in[sidx][i] = foc_ukf_out.at(i).reading[sidx];
        src_data.data_in = interp_in[sidx];
        src_data.data_out = interp_out[sidx];
        
        // interpolate
        src_reset(rate_conv);
        src_process(rate_conv, &src_data); 
    }

    // save to vector
    FOC_Reading_t new_interp = {0};
    foc_interp_out.clear();
    for (int i = 0; i < src_data.output_frames; i++)
    {
        new_interp.time += (double)i/(double)FOC_MOX_INTERP_FREQ;
        for (int sidx = 0; sidx < FOC_NUM_SENSORS; sidx++)
            new_interp.reading[sidx] = interp_out[sidx][i];
        foc_interp_out.push_back(new_interp);
    }

    for (int sidx = 0; sidx < FOC_NUM_SENSORS; sidx++) {
        free(interp_in[sidx]);
        free(interp_out[sidx]);
    }
    
    

    

    

    
    
/* Step 3: compute 1st derivative */
    noise_suppression_gaussian_filter(&foc_interp_out, &foc_diff_out, 2, 10.0);
/*
    if(noise_suppression_gaussian_smooth_update(&foc_interp_out, &foc_diff_out, 10.0)) { // smooth
        for (int i = 0; i < foc_diff_out.size()-1; i++) // 1st derivative
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
                foc_diff_out.at(i).reading[idx] = (foc_diff_out.at(i+1).reading[idx] - foc_diff_out.at(i).reading[idx])*FOC_MOX_INTERP_FREQ;
    }
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        foc_diff_out.back().reading[idx] = 0;
    // normalize
    double sum[FOC_NUM_SENSORS] = {0};
    float std_var[FOC_NUM_SENSORS] = {0};
    for (int i = 0; i < foc_diff_out.size(); i++)
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            sum[idx] += pow(foc_diff_out.at(i).reading[idx],2);
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        std_var[idx] = sqrt(sum[idx]/((double)foc_diff_out.size()));
    for (int i = 0; i < foc_diff_out.size(); i++)
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            foc_diff_out.at(i).reading[idx] = foc_diff_out.at(i).reading[idx]/std_var[idx];
*/

/* Step 4: find change points */
    double peak_time;
    if (foc_diff_out.size() >= 2) {
        for (int i = 0; i < FOC_NUM_SENSORS; i++)
            foc_peak_time[i].clear();
        for (int i = 0; i < foc_diff_out.size()-1; i++) {
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
                //if (SIGN(foc_diff_out.at(i).reading[idx]) != SIGN(foc_diff_out.at(i+1).reading[idx]))
                if (foc_diff_out.at(i).reading[idx] < 0 && foc_diff_out.at(i+1).reading[idx] > 0)
                {
                    peak_time = foc_diff_out.at(i).time + 
                        (0 - foc_diff_out.at(i).reading[idx])/
                        (foc_diff_out.at(i+1).reading[idx] - foc_diff_out.at(i).reading[idx])*
                        (foc_diff_out.at(i+1).time - foc_diff_out.at(i).time);
                    foc_peak_time[idx].push_back(peak_time);
                }
            }
        }
    }
/* Step : Save record */
#endif
}
