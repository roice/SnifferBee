#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "flying_odor_compass.h"

// index of latest calculated reading
static int index_in_reading = 1;

/* Calculate gradient
 * Args:
 *      out     containing results
 */
void foc_gradient_init(std::vector<FOC_Reading_t>& out)
{
    out.clear();
}

/* Calculate gradient
 * Args:
 *      in      input data
 *      out     output data
 * Return:
 *      false   an error happend
 */
bool foc_gradient_update(std::vector<FOC_Reading_t>& in, std::vector<FOC_Reading_t>& out)
{
    // check if args valid to calculate gradient
    if (in.size() < 3) // at least 3 point to calculate gradient
        return false; // not contain enough data to diff

    // check if there are new data to be diffed
    if (in.size() < index_in_reading + FOC_MOX_INTERP_FACTOR)
        return false;

    // calculate gradient
    FOC_Reading_t g; memset(&g, 0, sizeof(g));
    float kernel[3] = {-FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR/2, 0, FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR/2};
    for (int i = index_in_reading; i < in.size()-1; i++)
    {
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        {
            g.reading[idx] = kernel[0]*in.at(i-1).reading[idx] + kernel[2]*in.at(i+1).reading[idx]; // kernel[1] == 0.
        }
        out.push_back(g);
    }
    index_in_reading = in.size()-1;
    return true;
}
