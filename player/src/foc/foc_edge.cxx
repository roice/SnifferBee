#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "flying_odor_compass.h"

// index of latest calculated reading
static int index_in_reading = 1;

/* Find edge
 * Args:
 *      out     containing results
 */
void foc_edge_init(std::vector<FOC_Reading_t>& out_max, std::vector<FOC_Reading_t>& out_min)
{
    out_max.clear();
    out_min.clear();

    // for alignment
    FOC_Reading_t g; memset(&g, 0, sizeof(g));
    out_max.push_back(g);
    out_min.push_back(g);
}

/* Find edge from gradient
 * Args:
 *      in          input data
 *      out_max     output data, after non-maximum suppression
 *      out_min     output data, after non-minimum suppression
 * Return:
 *      false   an error happend
 */
bool foc_edge_update(std::vector<FOC_Reading_t>& in, std::vector<FOC_Reading_t>& out_max, std::vector<FOC_Reading_t>& out_min)
{
    // check if args valid to find edge
    if (in.size() < 3) // at least 3 point to find edge
        return false; // not contain enough data

    // check if there are new data
    if (in.size() < index_in_reading + FOC_MOX_INTERP_FACTOR)
        return false;

    // Non-maximum suppression
    float* grad_max[FOC_NUM_SENSORS];
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        grad_max[idx] = new float[in.size()-1-index_in_reading];
        memset(grad_max[idx], 0, (in.size()-1-index_in_reading)*sizeof(float));
    }
    int i;
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
    {
        i = index_in_reading;
        while (i < in.size()-1) // reserve last element for next update
        {      
            if (in.at(i).reading[idx] > in.at(i+1).reading[idx]) {
                if (in.at(i).reading[idx] >= in.at(i-1).reading[idx])
                    grad_max[idx][i-index_in_reading] = in.at(i).reading[idx];
            }
            else {
                i++;
                while (i < in.size()-1 and in.at(i).reading[idx] <= in.at(i+1).reading[idx])
                    i++;
                if (i < in.size()-1)
                    grad_max[idx][i-index_in_reading] = in.at(i).reading[idx];
            }
            i=i+2;
        }
    }

    // Non-minimum suppression
    float* grad_min[FOC_NUM_SENSORS];
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        grad_min[idx] = new float[in.size()-1-index_in_reading];
        memset(grad_min[idx], 0, (in.size()-1-index_in_reading)*sizeof(float));
    }
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
    {
        i = index_in_reading;
        while (i < in.size()-1) // reserve last element for next update
        {      
            if (in.at(i).reading[idx] < in.at(i+1).reading[idx]) {
                if (in.at(i).reading[idx] <= in.at(i-1).reading[idx])
                    grad_min[idx][i-index_in_reading] = in.at(i).reading[idx];
            }
            else {
                i++;
                while (i < in.size()-1 and in.at(i).reading[idx] >= in.at(i+1).reading[idx])
                    i++;
                if (i < in.size()-1)
                    grad_min[idx][i-index_in_reading] = in.at(i).reading[idx];
            }
            i=i+2;
        }
    }

    // save results
    FOC_Reading_t g; memset(&g, 0, sizeof(FOC_Reading_t));
    for (i = index_in_reading; i < in.size()-1; i++) {
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
            //g.reading[idx] = grad_max[idx][i-index_in_reading];
            if (grad_max[idx][i-index_in_reading] > 0)
                g.reading[idx] = grad_max[idx][i-index_in_reading];
            else
                g.reading[idx] = 0;
        }
        out_max.push_back(g);
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
            //g.reading[idx] = grad_min[idx][i-index_in_reading];
            if (grad_min[idx][i-index_in_reading] < 0)
                g.reading[idx] = grad_min[idx][i-index_in_reading];
            else
                g.reading[idx] = 0;
        }
        out_min.push_back(g);
    }

    // free memories
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        delete [] grad_max[idx];
        delete [] grad_min[idx];
    }

    index_in_reading = in.size()-1;
    
    return true;
}
