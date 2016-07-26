#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "flying_odor_compass.h"
#include "liquid.h"

static firinterp_rrrf q[FOC_NUM_SENSORS];
static int interp_factor = 4;

/* FIR interpolation
 * Args:
 *      samples     interpolated data
 *      k           interp factor (samples/symbol), k > 1, default: 4
 *      m           filter delay (symbols), m > 0, default: 3
 *      s           filter stop-band attenuation [dB], default: 60 
 */
void foc_interp_init(std::vector<FOC_Reading_t>& samples, int k = 4, int m = 3, float s = 60)
{
    // check if args are valid
    if (k < 2) {
        fprintf(stderr, "error: interp factor must be greater than 1\n");
        exit(1);
    } else if (m < 1) {
        fprintf(stderr, "error: filter delay must be greater than 0\n");
        exit(1);
    }

    // create interpolator from prototype
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        q[idx] = firinterp_rrrf_create_kaiser(k,m,s);
 
    samples.clear();

    interp_factor = k;
}

/* 
 * Args:
 *      symbol      reading
 *      samples     interpolated reading
 * Return:
 *      false       an error occured
 *      true        interpolation successful
 */
bool foc_interp_update(FOC_Reading_t& symbol, std::vector<FOC_Reading_t>& samples)
{
    float samples_array[FOC_NUM_SENSORS][interp_factor] = {0};

    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
    {
        firinterp_rrrf_execute(q[idx], symbol.reading[idx], &samples_array[idx][0]);
    }

    FOC_Reading_t sp; sp.time = 0;
    for (int i = 0; i < interp_factor; i++)
    {
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            sp.reading[idx] = samples_array[idx][i];
        samples.push_back(sp);
    }

    return true;
}

void foc_interp_terminate(void)
{
    // destroy interpolator object
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        firinterp_rrrf_destroy(q[idx]);
}
