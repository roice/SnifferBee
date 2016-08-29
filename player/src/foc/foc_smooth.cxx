#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include "flying_odor_compass.h"
#include "liquid.h"

#define FOC_SMOOTH_BASE_SIGMA   2.0
#define H_LEN       FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR    // filter length 

static firfilt_rrrf f[FOC_DIFF_GROUPS][FOC_DIFF_LAYERS_PER_GROUP+1][FOC_NUM_SENSORS];
static float h[FOC_DIFF_GROUPS][FOC_DIFF_LAYERS_PER_GROUP+1][FOC_NUM_SENSORS][H_LEN];
static float g_tune[FOC_DIFF_GROUPS][FOC_DIFF_LAYERS_PER_GROUP+1][FOC_NUM_SENSORS];

static int index_in_reading = 0;

static void CreateGaussianKernel(float sigma, float* gKernel, int nWindowSize);

/* FIR filtering
 * Args:
 *      out     filtered data
 */
void foc_smooth_init(std::vector<FOC_Reading_t>* out)
{
    float As = 60.0f;   // stop-band attenuation [dB]
    float mu = 0.0f;    // timing offset

    // calculate response
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP+1; j++)
            for (int k = 0; k < FOC_NUM_SENSORS; k++)
                CreateGaussianKernel(std::pow(2.0, i)*std::pow(std::pow(2.0, 1.0/(FOC_DIFF_LAYERS_PER_GROUP+1)), j)*FOC_SMOOTH_BASE_SIGMA,
                        h[i][j][k], H_LEN);

    // create filter from response
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP+1; j++)
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
                f[i][j][idx] = firfilt_rrrf_create(h[i][j][idx], H_LEN);

    // calculate filter gain and tune them to the same gain
    float gain[FOC_DIFF_GROUPS][FOC_DIFF_LAYERS_PER_GROUP+1][FOC_NUM_SENSORS];
    float temp_g;
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP+1; j++)
            for (int k = 0; k < FOC_NUM_SENSORS; k++) {
                temp_g = 0;
                for (int idx = 0; idx < H_LEN; idx++)
                    temp_g += h[i][j][k][idx];
                gain[i][j][k] = temp_g;
            }
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP+1; j++)
            for (int k = 0; k < FOC_NUM_SENSORS; k++)
                g_tune[i][j][k] = gain[0][0][0]/gain[i][j][k];

    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP+1; j++)
            out[i*(FOC_DIFF_LAYERS_PER_GROUP+1)+j].clear();

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
        for (int j = 0; j < FOC_DIFF_GROUPS; j++)
            for (int k = 0; k < FOC_DIFF_LAYERS_PER_GROUP+1; k++)
            {
                for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
                {
                    firfilt_rrrf_push(f[j][k][idx], in.at(i).reading[idx]);
                    firfilt_rrrf_execute(f[j][k][idx], &sp.reading[idx]);
                // adjust gain
                sp.reading[idx] *= g_tune[j][k][idx];
                }
                out[j*(FOC_DIFF_LAYERS_PER_GROUP+1)+k].push_back(sp);
            }
    }

    index_in_reading = in.size();

    return true;
}

void foc_smooth_terminate(void)
{
    // destroy interpolator object
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP+1; j++)
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
                firfilt_rrrf_destroy(f[i][j][idx]);
}

static void CreateGaussianKernel(float sigma, float* gKernel, int nWindowSize) {
    
    int nCenter = nWindowSize/2;
    double Value, Sum;
   
    // default no derivative
    gKernel[nCenter] = 1.0;
    Sum = 1.0;
    for (int i = 1; i <= nCenter; i++) {
        Value = 1.0/sigma*exp(-0.5*i*i/(sigma*sigma));
        if (nCenter+i < nWindowSize)
            gKernel[nCenter+i] = Value;
        gKernel[nCenter-i] = Value;
        Sum += 2.0*Value;
    }
    // normalize
    for (int i = 0; i < nWindowSize; i++)
        gKernel[i] = gKernel[i]/Sum;
}
