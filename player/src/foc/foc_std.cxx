#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <vector>
#include "flying_odor_compass.h"

#define     N       (FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR)

/* Calculate standard deviation
 * Args:
 *      out         output data, standard deviation
 */
void foc_std_init(std::vector<FOC_STD_t>* out)
{
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++)
            out[i*FOC_DIFF_LAYERS_PER_GROUP+j].clear();
}

/* Calculate standard deviation
 * Args:
 *      diff        difference vectors
 *      out         output data, standard deviation
 */
bool foc_std_update(std::vector<FOC_Reading_t>* diff, std::vector<FOC_STD_t>* out)
{
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++)
            if (diff[i*FOC_DIFF_LAYERS_PER_GROUP+j].size() < N)
                return false;

    FOC_STD_t   new_out;
    double sum[FOC_NUM_SENSORS] = {0};

    for (int grp = 0; grp < FOC_DIFF_GROUPS; grp++)
        for (int lyr = 0; lyr < FOC_DIFF_LAYERS_PER_GROUP; lyr++) {
            memset(sum, 0, sizeof(sum));
            for (int i = diff[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size() - N; i < diff[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size(); i++) {
                for (int j = 0; j < FOC_NUM_SENSORS; j++)
                    sum[j] += std::abs(diff[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[j]);
            }
            for (int j = 0; j < FOC_NUM_SENSORS; j++)
                new_out.std[j] = sum[j];
            out[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].push_back(new_out);
        }

    return true;
}
