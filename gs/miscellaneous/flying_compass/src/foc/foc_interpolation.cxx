#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "flying_odor_compass.h"
#include "liquid.h"

/* FIR interpolation
 * Args:
 *      symbols     data to be interpolated
 *      samples     interpolated data
 *      k           interp factor (samples/symbol), k > 1, default: 4
 *      m           filter delay (symbols), m > 0, default: 3
 *      s           filter stop-band attenuation [dB], default: 60
 * Output:
 *      false       length of symbols are shorter than FOC_DELAY*FOC_MOX_DAQ_FREQ
 *      true        interpolation successful
 */
bool foc_interpolation(std::vector<FOC_Reading_t>& symbols, std::vector<FOC_Reading_t>& samples, int k = 4, int m = 3, float s = 60)
{
    // check if the length of symbols is enough long
    if (symbols.size() < FOC_DELAY*FOC_MOX_DAQ_FREQ)
        return false;

    // check if args are valid
    if (k < 2) {
        fprintf(stderr, "error: interp factor must be greater than 1\n");
        exit(1);
    } else if (m < 1) {
        fprintf(stderr, "error: filter delay must be greater than 0\n");
        exit(1);
    }

    int len_symbols = FOC_DELAY*FOC_MOX_DAQ_FREQ;
    int num_symbols = len_symbols + 2*m; // compensate for filter delay
    int num_samples = k*num_symbols;
    float samples_array[FOC_NUM_SENSORS][num_samples] = {0};

    // create interpolator from prototype
    firinterp_rrrf q[FOC_NUM_SENSORS];
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        q[idx] = firinterp_rrrf_create_kaiser(k,m,s);

    // interpolate symbols
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
    {
        for (int i = 0; i < len_symbols; i++)
            firinterp_rrrf_execute(q[idx], symbols.at(i+symbols.size()-len_symbols).reading[idx], &samples_array[idx][k*i]);
        for (int i = len_symbols; i < num_symbols; i++)
            firinterp_rrrf_execute(q[idx], 0, &samples_array[idx][k*i]);
    }

    // destroy interpolator object
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        firinterp_rrrf_destroy(q[idx]);
    
    // save results
    FOC_Reading_t sp;
    sp.time = 0;
    samples.clear(); // data_interp only stores recent FOC_DELAY data
    for (int i = 0; i < num_samples; i++)
    {
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            sp.reading[idx] = samples_array[idx][i];
        samples.push_back(sp);
    }

    return true;
}
