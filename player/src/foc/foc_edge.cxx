#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "flying_odor_compass.h"

// index of latest calculated reading
static int index_in_reading[FOC_DIFF_GROUPS][FOC_DIFF_LAYERS_PER_GROUP] = {0};

/* Find edge
 * Args:
 *      out     containing results
 */
void foc_edge_init(std::vector<FOC_Reading_t>* out_max, std::vector<FOC_Reading_t>* out_min)
{
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++) {
        out_max[i*FOC_DIFF_LAYERS_PER_GROUP+j].clear();
        out_min[i*FOC_DIFF_LAYERS_PER_GROUP+j].clear();
    }

    // for alignment
    FOC_Reading_t g; memset(&g, 0, sizeof(g));
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
        for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++) {
            out_max[i*FOC_DIFF_LAYERS_PER_GROUP+j].push_back(g);
            out_min[i*FOC_DIFF_LAYERS_PER_GROUP+j].push_back(g);
            index_in_reading[i][j] = 1;
        }
}

/* Find edge from gradient
 * Args:
 *      in          input data
 *      out_max     output data, after non-maximum suppression
 *      out_min     output data, after non-minimum suppression
 * Return:
 *      false   an error happend
 */
bool foc_edge_update(std::vector<FOC_Reading_t>* in, std::vector<FOC_Reading_t>* out_max, std::vector<FOC_Reading_t>* out_min)
{
    float* grad_max[FOC_NUM_SENSORS];
    float* grad_min[FOC_NUM_SENSORS];

    int i;

    FOC_Reading_t g;
   
    for (int grp = 0; grp < FOC_DIFF_GROUPS; grp++)
        for (int lyr = 0; lyr < FOC_DIFF_LAYERS_PER_GROUP; lyr++)
        {
            // check if args valid to find edge
            if (in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size() < 3) // at least 3 point to find edge
                continue;

            // check if there are new data
            if (in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size() < index_in_reading[grp][lyr] + FOC_MOX_INTERP_FACTOR)
                continue;

            // Non-maximum suppression
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
                grad_max[idx] = new float[in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1-index_in_reading[grp][lyr]];
                memset(grad_max[idx], 0, (in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1-index_in_reading[grp][lyr])*sizeof(float));
            }
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            {
                i = index_in_reading[grp][lyr];
                while (i < in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1) // reserve last element for next update
                {      
                    if (in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx] > in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i+1).reading[idx]) {
                    if (in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx] >= in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i-1).reading[idx])
                        grad_max[idx][i-index_in_reading[grp][lyr]] = in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx];
                }
                else {
                    i++;
                    while (i < in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1 and in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx] <= in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i+1).reading[idx])
                        i++;
                    if (i < in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1)
                        grad_max[idx][i-index_in_reading[grp][lyr]] = in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx];
                }
                i=i+2;
            }
        }

        // Non-minimum suppression
    
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
            grad_min[idx] = new float[in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1-index_in_reading[grp][lyr]];
            memset(grad_min[idx], 0, (in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1-index_in_reading[grp][lyr])*sizeof(float));
        }
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        {
            i = index_in_reading[grp][lyr];
            while (i < in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1) // reserve last element for next update
            {      
                if (in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx] < in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i+1).reading[idx]) {
                    if (in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx] <= in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i-1).reading[idx])
                        grad_min[idx][i-index_in_reading[grp][lyr]] = in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx];
                }
                else {
                    i++;
                    while (i < in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1 and in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx] >= in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i+1).reading[idx])
                        i++;
                    if (i < in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1)
                        grad_min[idx][i-index_in_reading[grp][lyr]] = in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(i).reading[idx];
                }
                i=i+2;
            }
        }

        // save results
        memset(&g, 0, sizeof(g));
        for (i = index_in_reading[grp][lyr]; i < in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1; i++) {
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
                //g.reading[idx] = grad_max[idx][i-index_in_reading[grp][lyr]];
                
                if (grad_max[idx][i-index_in_reading[grp][lyr]] > 0)
                    g.reading[idx] = grad_max[idx][i-index_in_reading[grp][lyr]];
                else
                    g.reading[idx] = 0;
                
            }
            out_max[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].push_back(g);
            for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
                //g.reading[idx] = grad_min[idx][i-index_in_reading[grp][lyr]];
                
                if (grad_min[idx][i-index_in_reading[grp][lyr]] < 0)
                    g.reading[idx] = grad_min[idx][i-index_in_reading[grp][lyr]];
                else
                    g.reading[idx] = 0;
                
            }
            out_min[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].push_back(g);
        }

        // free memories
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
            delete [] grad_max[idx];
            delete [] grad_min[idx];
        }

        index_in_reading[grp][lyr] = in[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size()-1;
    }
    
    return true;
}
