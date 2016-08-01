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
#include "foc/foc_noise_reduction.h"
#include "foc/foc_interp.h"
#include "foc/foc_smooth.h"
#include "foc/foc_diff.h"
#include "foc/foc_delta.h"
#include "foc/foc_estimate.h"

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
    data_wind.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    data_raw.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    data_denoise.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    data_interp.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
    data_smooth.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
    data_diff.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
    data_delta.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    data_est.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
/* init UKF filtering */
    foc_noise_reduction_ukf_init();
/* init FIR interpolation */
    foc_interp_init(data_interp, FOC_MOX_INTERP_FACTOR, FOC_MOX_DAQ_FREQ*1, 60);
/* init FIR smoothing
 * h_len = 5 s * sampling_freq, fc = 0.2 Hz */
    foc_smooth_init(data_smooth, 5*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR, 0.2f/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR*2, 60, 0.0);
/* init Differentiation 
 * order = 3 */
    foc_diff_init(data_diff, 3);
/* init feature extraction */
    foc_delta_init(data_delta);
/* init direction estimation */
    foc_estimate_init(data_est);
}

/* FOC update
 * Bool output:
 *      true    Odor direction updated 
 *      false   Haven't dug out useful information
 */
bool Flying_Odor_Compass::update(FOC_Input_t& new_in)
{
    data_raw.push_back(new_in); // save record
/* Step 0: Pre-processing */

/* Step 1: Noise reduction through UKF filtering */
    FOC_Reading_t ukf_out = foc_noise_reduction_ukf_update(new_in);
    data_denoise.push_back(ukf_out); // save record

/* Step 2: FIR interpolation (`zero-stuffing' upsampling + filtering) */
    if (!foc_interp_update(ukf_out, data_interp))
        return false;

/* Step 3: Smoothing through FIR filtering */
    if (!foc_smooth_update(data_interp, data_smooth))
        return false;

/* Step 4: Derivative */
    if (!foc_diff_update(data_smooth, data_diff))
        return false;

/* Step 5: Extracting features: time diff and variance */
    if (!foc_delta_update(data_diff, data_delta))
        return false;

/* Step 6: Estimate the direction the odor comes from 
 * Warning: This step is only suitable for 3 sensors (FOC_NUM_SENSORS = 3)*/
    if (!foc_estimate_update(data_delta, data_est))
        return false;

    return true;
}
