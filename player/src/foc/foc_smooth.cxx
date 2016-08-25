#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include "flying_odor_compass.h"
#include "liquid.h"

#define FOC_SMOOTH_BASE_FREQ    1.5     // Hz
#define H_LEN   FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR    // filter length 

static firfilt_rrrf f[FOC_DIFF_LAYERS+1][FOC_NUM_SENSORS];
static float h[FOC_DIFF_LAYERS+1][FOC_NUM_SENSORS][H_LEN];
static float g_tune[FOC_DIFF_LAYERS+1][FOC_NUM_SENSORS];

static int index_in_reading = 0;

/* FIR filtering
 * Args:
 *      out     filtered data
 */
void foc_smooth_init(std::vector<FOC_Reading_t>* out)
{
    float As = 60.0f;   // stop-band attenuation [dB]
    float mu = 0.0f;    // timing offset
    float fc[FOC_DIFF_LAYERS+1][FOC_NUM_SENSORS];   // filter cutoff

    for (int i = 0; i < FOC_DIFF_LAYERS+1; i++)
        for (int j = 0; j < FOC_NUM_SENSORS; j++)
            fc[i][j] = (FOC_SMOOTH_BASE_FREQ*std::pow(2.0, (float)i/(float)(FOC_DIFF_LAYERS+1)))/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR*2; 

    // create filter from prototype
    for (int i = 0; i < FOC_DIFF_LAYERS+1; i++)
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
            //f[i][idx] = firfilt_rrrf_create_kaiser(h_len, fc[i][idx], As, mu);
            liquid_firdes_kaiser(H_LEN, fc[i][idx], As, mu, h[i][idx]);
            f[i][idx] = firfilt_rrrf_create(h[i][idx], H_LEN);
            }

    // calculate filter gain and tune them to the same gain
    float gain[FOC_DIFF_LAYERS+1][FOC_NUM_SENSORS];
    float temp_g;
    for (int i = 0; i < FOC_DIFF_LAYERS+1; i++)
        for (int j = 0; j < FOC_NUM_SENSORS; j++) {
            temp_g = 0;
            for (int k = 0; k < H_LEN; k++)
                temp_g += h[i][j][k];
            gain[i][j] = temp_g;
        }
    for (int i = 0; i < FOC_DIFF_LAYERS+1; i++)
        for (int j = 0; j < FOC_NUM_SENSORS; j++) {
            g_tune[i][j] = gain[0][0]/gain[i][j];
        }


    for (int i = 0; i < FOC_DIFF_LAYERS+1; i++)
        out[i].clear();

    index_in_reading = 0;
}

/* 
 * Args:
 *      in          reading vector
 *      out         filtered reading
 * Return:
 *      false       an error occured
 *      true        interpolation successful
 */
bool foc_smooth_update(std::vector<FOC_Reading_t>& in, std::vector<FOC_Reading_t>* out)
{
    // because smoothing is after interpolation
    if (in.size() < index_in_reading + FOC_MOX_INTERP_FACTOR)
        return false;

    FOC_Reading_t sp; sp.time = 0;
    
    for (int i = index_in_reading; i < in.size(); i++)
    {
        for (int j = 0; j < FOC_DIFF_LAYERS+1; j++)
        {
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            {
                firfilt_rrrf_push(f[j][idx], in.at(i).reading[idx]);
                firfilt_rrrf_execute(f[j][idx], &sp.reading[idx]);
                // adjust gain
                sp.reading[idx] *= g_tune[j][idx];
            }
            out[j].push_back(sp);
        }
    }

    index_in_reading = in.size();

    return true;
}

void foc_smooth_terminate(void)
{
    // destroy interpolator object
    for (int i = 0; i < FOC_DIFF_LAYERS+1; i++)
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            firfilt_rrrf_destroy(f[i][idx]);
}
