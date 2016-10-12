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
#include <string.h>
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
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP+1; j++)
            data_smooth[i*(FOC_DIFF_LAYERS_PER_GROUP+1)+j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++) { 
            data_diff[i*FOC_DIFF_LAYERS_PER_GROUP+j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
            data_edge_max[i*FOC_DIFF_LAYERS_PER_GROUP+j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
            data_edge_min[i*FOC_DIFF_LAYERS_PER_GROUP+j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
            data_cp_max[i*FOC_DIFF_LAYERS_PER_GROUP+j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
            data_cp_min[i*FOC_DIFF_LAYERS_PER_GROUP+j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
            data_tdoa[i*FOC_DIFF_LAYERS_PER_GROUP+j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
            data_std[i*FOC_DIFF_LAYERS_PER_GROUP+j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
        } 
    data_est.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP);
/* init wind filtering */
    foc_wind_smooth_init(data_wind); // FOC_DELAY s delay
/* init UKF filtering */
    foc_noise_reduction_ukf_init();
/* init FIR interpolation
 * delay = FOC_SIGNAL_DELAY/2 s */
    foc_interp_init(data_interp, FOC_MOX_INTERP_FACTOR, (int)((float)FOC_SIGNAL_DELAY*(float)FOC_MOX_DAQ_FREQ/2.0), 60); // FOC_DELAY s delay, consistent with wind smoothing
/* init FIR smoothing
 * h_len = FOC_SIGNAL_DELAY s * sampling_freq 
 * delay = FOC_SIGNAL_DELAY/2 s , because the delay of FIR filter = (N-1)/(2*Fs) */
    foc_smooth_init(data_smooth);
/* init Differentiation */
    foc_diff_init(data_diff);
/* init Edge finding */
    foc_edge_init(data_edge_max, data_edge_min);
/* init feature extraction: std */
    foc_std_init(data_std);
/* init feature extraction: tdoa */
    foc_tdoa_init(data_cp_max, data_cp_min, data_tdoa);
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
    FOC_Wind_t  new_wind;
    memcpy(new_wind.wind, new_in.wind, 3*sizeof(float));
    data_wind.push_back(new_wind);

#if 0
/* Step 1: Noise reduction through UKF filtering */
    FOC_Reading_t ukf_out = foc_noise_reduction_ukf_update(new_in);
    data_denoise.push_back(ukf_out); // save record
#endif

/* Step 2: FIR interpolation (`zero-stuffing' upsampling + filtering)
 *         delay = FOC_SIGNAL_DELAY/2 s */
    FOC_Reading_t ukf_out;
    memcpy(ukf_out.reading, new_in.mox_reading, FOC_NUM_SENSORS*sizeof(float));
    if (!foc_interp_update(ukf_out, data_interp))
        return false;

/* Step 3: Smoothing through FIR filtering
 *         delay ~ FOC_SIGNAL_DELAY/2 s */
    if (!foc_smooth_update(data_interp, data_smooth))
        return false;

/* Step 4: Differences
 *         truncate first FOC_SIGNAL_DELAY s data */
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

/* Step 8: Estimate the direction the odor comes from 
 * Warning: This step is only suitable for 3 sensors (FOC_NUM_SENSORS = 3) */
    if (!foc_estimate_source_direction_update(data_raw, data_std, data_tdoa, data_wind, data_est))
        return false;

    return true;
}
