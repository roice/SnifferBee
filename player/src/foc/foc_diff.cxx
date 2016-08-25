#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "flying_odor_compass.h"
#include "liquid.h"

// index of latest differentiated reading
static int index_in_reading = 2*FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR; // skip FIR init fluctuation

/* Difference of Gaussian */
void foc_diff_init(std::vector<FOC_Reading_t>* out)
{
    for (int i = 0; i < FOC_DIFF_LAYERS; i++)
        out[i].clear();

    index_in_reading = 2*FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
}

bool foc_diff_update(std::vector<FOC_Reading_t>* in, std::vector<FOC_Reading_t>* out)
{
    // check if there are new data to be diffed
    if (in[0].size() < index_in_reading + FOC_MOX_INTERP_FACTOR) // in arrays have the same size
        return false;

    FOC_Reading_t sp; sp.time = 0;
    for (int layer = 0; layer < FOC_DIFF_LAYERS; layer++)
    {
        for (int i = index_in_reading; i < in[layer].size(); i++)
        {
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            {
                sp.reading[idx] = in[layer].at(i).reading[idx] - in[layer+1].at(i).reading[idx];
            }
            out[layer].push_back(sp);
        }
    }

    // update index, in arrays have the same size
    index_in_reading = in[0].size();

    return true;
}

/* Derivatives */
#if 0

// index of latest differentiated reading
static int index_in_reading = 0;

static float x[FOC_DIFF_LAYERS][FOC_DIFF_LAYERS+1];
static float y[FOC_DIFF_LAYERS];// support upto 6-th order derivative

static firfilt_rrrf f[FOC_DIFF_LAYERS][FOC_NUM_SENSORS];

/* Derivative + smoothing
 * Args:
 *      out     difference vector array
 */
void foc_diff_init(std::vector<FOC_Reading_t>* out)
{
    // create filter from prototype
    // len = 0.1 s, freq = 10 Hz
    for (int order = 1; order <= FOC_DIFF_LAYERS; order++)
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            f[order-1][idx] = firfilt_rrrf_create_kaiser(FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR, (order*1.0)/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR*2, 60.0, 0.0);

    for (int i = 0; i < FOC_DIFF_LAYERS; i++)
        out[i].clear();

    // for alignment
    FOC_Reading_t   d; memset(&d, 0, sizeof(d));
    for (int order = 1; order <= FOC_DIFF_LAYERS; order++)
        for (int i = 0; i < order; i++)
            out[order-1].push_back(d);

    memset(&x, 0, FOC_DIFF_LAYERS*(FOC_DIFF_LAYERS+1)*sizeof(float));
    memset(&y, 0, FOC_DIFF_LAYERS*sizeof(float));

    index_in_reading = 0;
}

/* Differentiate signals
 * Args:
 *      in      input data
 *      out     output data, vector array
 * Return:
 *      false   an error happend
 *      true    diff successful
 */
bool foc_diff_update(std::vector<FOC_Reading_t>& in, std::vector<FOC_Reading_t>* out)
{
    // check if there are new data to be diffed
    if (in.size() < index_in_reading + FOC_MOX_INTERP_FACTOR)
        return false;

    // diff, 1 <= order <= 3
    FOC_Reading_t sp; sp.time = 0;
    for (int order = 1; order <= (FOC_DIFF_LAYERS > 3 ? 3 : FOC_DIFF_LAYERS); order++)
    {
        for (int i = index_in_reading; i < in.size(); i++)
        {
            if (i < order) continue; // first run, x[order-1][] should init 0
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            {
                for (int j = 0; j < order+1; j++)
                    x[order-1][j] = in.at(i-order+j).reading[idx];
                switch (order) {
                    case 1:
                        y[order-1] = (x[order-1][1] - x[order-1][0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                    case 2:
                        y[order-1] = (x[order-1][2] - 2*x[order-1][1] + x[order-1][0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break; 
                    case 3:
                        y[order-1] = (x[order-1][3] - 3*x[order-1][2] + 3*x[order-1][1] - x[order-1][0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                    /*
                    case 4:
                        y[order-1] = (x[order-1][4] - 4*x[order-1][3] + 6*x[order-1][2] - 4*x[order-1][1] + x[order-1][0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                    case 5:
                        y[order-1] = (x[order-1][5] - 5*x[order-1][4] + 10*x[order-1][3] - 10*x[order-1][2] + 5*x[order-1][1] - x[order-1][0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                    case 6:
                        y[order-1] = (x[order-1][6] - 6*x[order-1][5] + 15*x[order-1][4] - 20*x[order-1][3] + 15*x[order-1][2] - 6*x[order-1][1] + x[order-1][0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                    */
                    default: // 2-nd
                        y[order-1] = (x[order-1][2] - 2*x[order-1][1] + x[order-1][0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                }

                // smooth
                firfilt_rrrf_push(f[order-1][idx], y[order-1]);
                firfilt_rrrf_execute(f[order-1][idx], &sp.reading[idx]);
            }
            // save results
            out[order-1].push_back(sp);
        }
    }

    // diff, order >= 4
    if (FOC_DIFF_LAYERS > 3)
    {
        for (int order = 4; order <= FOC_DIFF_LAYERS; order++)
        {
            for (int i = index_in_reading; i < in.size(); i++)
            {
                if (i < order) continue;
                for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
                {
                    for (int j = 0; j < 2; j++)
                        x[order-1][j] = out[order-2].at(i-1+j).reading[idx];
                    y[order-1] = (x[order-1][1] - x[order-1][0]);
                    // smooth
                    firfilt_rrrf_push(f[order-1][idx], y[order-1]);
                    firfilt_rrrf_execute(f[order-1][idx], &sp.reading[idx]);
                }
                // save results
                out[order-1].push_back(sp);
            }
        }
    }

    // update index
    index_in_reading = in.size();

    return true;
}

#endif
