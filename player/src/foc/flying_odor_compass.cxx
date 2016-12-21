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
#include "foc/foc_interp.h"
#include "foc/foc_wind.h"
#include "foc/foc_wavelet.h"

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
    // make space for data
    data_wind.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    data_raw.reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        data_interp[i].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR); 
    }
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        for (int j = 0; j < FOC_WT_LEVELS; j++)
            data_wt_out[i][j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        for (int j = 0; j < FOC_WT_LEVELS; j++)
            for (int k = 0; k < 2; k++)
                data_modmax[i][j][k].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_WT_LEVELS); // a bit exaggerated
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        for (int j = 0; j < 2; j++)
            data_maxline[i][j].reserve(FOC_RECORD_LEN*FOC_MOX_DAQ_FREQ);

/* init FIR interpolation
 * delay = FOC_SIGNAL_DELAY/2 s */
    foc_interp_init(data_interp, FOC_MOX_INTERP_FACTOR, (int)((float)FOC_SIGNAL_DELAY*(float)FOC_MOX_DAQ_FREQ/2.0), 60);

# if 0 // Scale space method. Smooth+Diff+Edges+TDOA
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
#endif

    foc_cwt_init(data_wvs, data_wvs_idx, data_wt_out);
    foc_identify_modmax_init(data_modmax);
    foc_chain_maxline_init(data_maxline);
}

/* FOC update
 * Bool output:
 *      true    Odor direction updated 
 *      false   Haven't dug out useful information
 */
bool Flying_Odor_Compass::update(FOC_Input_t& new_in)
{ 
/* Step 0: Pre-processing */
    // save some info to record
    data_raw.push_back(new_in); // save raw data
    FOC_Wind_t  new_wind;
    memcpy(new_wind.wind, new_in.wind, 3*sizeof(float));
    data_wind.push_back(new_wind); // save wind data

/* Step 1: FIR interpolation (`zero-stuffing' upsampling + filtering)
 *         delay = FOC_SIGNAL_DELAY/2 s */
    if (!foc_interp_update(new_in.mox_reading, data_interp))
        return false;

/* Step 2: Wavelet Transformation */
    if (!foc_cwt_update(data_interp, data_wt_out))
        return false;

/* Step 3: Find (Identify) Modulus Maxima */
    if (!foc_identify_modmax_update(data_wt_out, data_modmax))
        return false;

/* Step 4: Chain maxima lines */
    if (!foc_chain_maxline_update(data_modmax, data_maxline, data_wt_out[0][0].size()))
        return false;

#if 0
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
#endif
    return true;
}
