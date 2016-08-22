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
#include "foc/foc_edge.h"
#include "foc/foc_tdoa.h"
#include "foc/foc_std.h"
#include "foc/foc_estimate.h"
#include "foc/foc_wind.h"

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
    for (int i = 0; i < FOC_DIFF_LAYERS; i++) {
        data_diff[i].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
        data_edge_max[i].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
        data_edge_min[i].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
        data_cp_max[i].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
        data_cp_min[i].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
        data_tdoa[i].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
    }
    data_std.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    data_est.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_DIFF_LAYERS);
/* init wind filtering */
    foc_wind_smooth_init(data_wind); // FOC_DELAY s delay
/* init UKF filtering */
    foc_noise_reduction_ukf_init();
/* init FIR interpolation */
    foc_interp_init(data_interp, FOC_MOX_INTERP_FACTOR, FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ, 60); // FOC_DELAY s delay, consistent with wind smoothing
/* init FIR smoothing
 * h_len = FOC_SIGNAL_DELAY s * sampling_freq, fc = 1.0 Hz */
    foc_smooth_init(data_smooth, FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR, 1.0f/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR*2, 60, 0.0);
/* init Differentiation */
    foc_diff_init(data_diff);
/* init Edge finding */
    foc_edge_init(data_edge_max, data_edge_min);
/* init feature extraction: std */
    foc_std_init(data_std);
/* init feature extraction: tdoa */
    foc_tdoa_init(data_cp_max, data_cp_min, data_tdoa);
#if 0
/* init gradient */
    foc_gradient_init(data_gradient);
/* init edge */
    foc_edge_init(data_edge_max, data_edge_min);
/* init feature extraction */
    foc_delta_init(data_cp_max, data_cp_min, data_delta);
#endif

/* init direction estimation */
    foc_estimate_source_direction_init(data_est);
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
    foc_wind_smooth_update(new_in, data_wind); // smooth wind data

#if 0
/* Step 1: Noise reduction through UKF filtering */
    FOC_Reading_t ukf_out = foc_noise_reduction_ukf_update(new_in);
    data_denoise.push_back(ukf_out); // save record

/* Step 2: FIR interpolation (`zero-stuffing' upsampling + filtering) */
    if (!foc_interp_update(ukf_out, data_interp))
        return false;
#endif
    FOC_Reading_t new_rd;
    memcpy(&new_rd.reading, &new_in.mox_reading, FOC_NUM_SENSORS*sizeof(float));
    if (!foc_interp_update(new_rd, data_interp))
        return false;

/* Step 3: Smoothing through FIR filtering */
    if (!foc_smooth_update(data_interp, data_smooth))
        return false;

/* Step 4: Derivative */
    if (!foc_diff_update(data_smooth, data_diff))
        return false;

/* Step 5: Find edges */
    if (!foc_edge_update(data_diff, data_edge_max, data_edge_min))
        return false;

/* Step 6: Extracting features: standard deviation */
    if (!foc_std_update(data_diff, data_std))
        return false;

/* Step 7: Extracting features: time diff */
    if (!foc_tdoa_update(data_diff, data_edge_max, data_edge_min, data_cp_max, data_cp_min, data_tdoa))
        return false;

#if 0
/* Step 4: Calculate gradient */
    if (!foc_gradient_update(data_smooth, data_gradient))
        return false;

/* Step 5: Calculate edge */
    if (!foc_edge_update(data_gradient, data_edge_max, data_edge_min))
        return false;

/* Step 5: Extracting features: time diff and variance */
    if (!foc_delta_update(data_smooth, data_edge_max, data_edge_min, data_cp_max, data_cp_min, data_delta))
        return false;
#endif
/* Step 6: Estimate the direction the odor comes from 
 * Warning: This step is only suitable for 3 sensors (FOC_NUM_SENSORS = 3)*/
    if (!foc_estimate_source_direction_update(data_raw, data_tdoa, data_wind, data_est))
        return false;

    return true;
}
