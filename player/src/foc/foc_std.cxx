#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include "flying_odor_compass.h"

#define     N       (FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR)

/* Calculate standard deviation
 * Args:
 *      out         output data, standard deviation
 */
void foc_std_init(std::vector<FOC_STD_t>& out)
{ 
    out.clear();
}

/* Calculate standard deviation
 * Args:
 *      diff        difference vectors
 *      out         output data, standard deviation
 */
bool foc_std_update(std::vector<FOC_Reading_t>* diff, std::vector<FOC_STD_t>& out)
{
    for (int i = 0; i < FOC_DIFF_LAYERS; i++) {
        if (diff[i].size() < N)
            return false;
    }

    FOC_STD_t   new_out;
    double sum[FOC_NUM_SENSORS] = {0};
    for (int i = diff[1].size() - N; i < diff[1].size(); i++) {
        for (int j = 0; j < FOC_NUM_SENSORS; j++) {
            sum[j] += std::abs(diff[1].at(i).reading[j]);
        }
    }
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        new_out.std[i] = sum[i]/N;

    out.push_back(new_out);

    return true;
}
