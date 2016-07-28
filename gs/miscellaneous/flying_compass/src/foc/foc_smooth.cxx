#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "flying_odor_compass.h"
#include "liquid.h"

static firfilt_rrrf f[FOC_NUM_SENSORS];

static int index_in_reading = 0;

/* FIR filtering
 * Args:
 *      out     filtered data
 *      h_len       filter length
 *      fc          filter cutoff
 *      As          stop-band attenuation [dB]
 *      mu          timing offset
 */
void foc_smooth_init(std::vector<FOC_Reading_t>& out, int h_len = FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR, float fc = 0.2f/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR, float As = 60.0f, float mu = 0.0f)
{
    // check if args are valid
    if (h_len < 1) {
        fprintf(stderr, "error: filter length must be greater than 1\n");
        exit(1);
    } else if (fc <= 0) {
        fprintf(stderr, "error: filter cutoff must be greater than 0\n");
        exit(1);
    }

    // create filter from prototype
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        f[idx] = firfilt_rrrf_create_kaiser(h_len, fc, As, mu);
 
    out.clear();

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
bool foc_smooth_update(std::vector<FOC_Reading_t>& in, std::vector<FOC_Reading_t>& out)
{
    // because smoothing is after interpolation
    if (in.size() < index_in_reading + FOC_MOX_INTERP_FACTOR)
        return false;

    FOC_Reading_t sp; sp.time = 0;
    
    for (int i = index_in_reading; i < in.size(); i++)
    {
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        {
            firfilt_rrrf_push(f[idx], in.at(i).reading[idx]);
            firfilt_rrrf_execute(f[idx], &sp.reading[idx]);
        }
        out.push_back(sp);
    }

    index_in_reading = in.size();

    return true;
}

void foc_smooth_terminate(void)
{
    // destroy interpolator object
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        firfilt_rrrf_destroy(f[idx]);
}
