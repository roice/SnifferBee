#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "flying_odor_compass.h"
#include "liquid.h"

// index of latest differentiated reading
static int index_in_reading = 0;

static int diff_order = 2;
static float x[4+1] = {0}, y = 0;// support upto 4-th order derivative

static firfilt_rrrf f[FOC_NUM_SENSORS];

/* Derivative + smoothing
 * Args:
 *      order   order of diff, 1 <= order <= 4, default: 2
 */
void foc_diff_init(std::vector<FOC_Reading_t>& out, int order=2)
{
    // check if args valid for differentiation
    if (order < 1 || order > 4) {
        fprintf(stderr, "error: diff order %d not valid.\n", order);
        exit(1);
    }

    // create filter from prototype
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        f[idx] = firfilt_rrrf_create_kaiser(FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR, 0.5f/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR*2, 60.0, 0.0);

    out.clear();

    diff_order = order;
    index_in_reading = 0;
}

/* Differentiate signals
 * Args:
 *      in      input data
 *      out     output data 
 * Return:
 *      false   an error happend
 *      true    diff successful
 */
bool foc_diff_update(std::vector<FOC_Reading_t>& in, std::vector<FOC_Reading_t>& out)
{
    // check if args valid for differentiation
    if (in.size() < diff_order+1)
        return false; // not contain enough data to diff

    // check if there are new data to be diffed
    if (in.size() < index_in_reading + FOC_MOX_INTERP_FACTOR)
        return false;

    // diff
    FOC_Reading_t sp; sp.time = 0;
    for (int i = index_in_reading; i < in.size(); i++)
    {
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        {
            for (int j = 0; j < diff_order+1; j++)
            {
                if (i < diff_order) // first run
                    continue;   // x[] should init 0
                else
                    x[j] = in.at(i-diff_order+j).reading[idx];
            }
            switch (diff_order) {
                case 1:
                    y = (x[1] - x[0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                case 2:
                    y = (x[2] - 2*x[1] + x[0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break; 
                case 3:
                    y = (x[3] - 3*x[2] + 3*x[1] - x[0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                case 4:
                    y = (x[4] - 4*x[3] + 6*x[2] - 4*x[1] + x[0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
                default: // 2-nd
                    y = (x[2] - 2*x[1] + x[0])*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
                    break;
            }

            // smooth
            firfilt_rrrf_push(f[idx], y);
            firfilt_rrrf_execute(f[idx], &sp.reading[idx]);
        }

        // save results
        out.push_back(sp);
    }

    // update index
    index_in_reading = in.size();

    return true;
}
