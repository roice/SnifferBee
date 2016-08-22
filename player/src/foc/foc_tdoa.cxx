#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <vector>
#include "flying_odor_compass.h"

#define     N   (FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR)

/* Feature extraction
 * Args:
 *      cp_max  change points extracted from non-maximum suppressed gradient
 *      cp_min  change points extracted from non-minimum suppressed gradient
 *      out     features (time of arrival & standard deviation) extracted to output vector
 */
void foc_tdoa_init(std::vector<FOC_ChangePoints_t>* cp_max, std::vector<FOC_ChangePoints_t>* cp_min, std::vector<FOC_TDOA_t>* out)
{
    for (int i = 0; i < FOC_DIFF_LAYERS; i++) {
        cp_max[i].clear();
        cp_min[i].clear();
        out[i].clear();
    }
}
/* Feature extraction
 * Args:
 *      grad    gradient of mox signals, to calculate std
 *      edge    non-extremum suppressioned gradient, to calculate tdoa
 * Return:
 *      false   an error happend
 *      true    feature extraction successful
 */

static void reorganize_edge_to_a_vector(std::vector<FOC_Reading_t>&, int, std::vector<FOC_Edge_t>&);
static void find_pairs_of_change_points(std::vector<FOC_Edge_t>&, std::vector<FOC_ChangePoints_t>&);
static bool calculate_delta(std::vector<FOC_ChangePoints_t>&, int, std::vector<FOC_TDOA_t>&, std::vector<FOC_Reading_t>&);

bool foc_tdoa_update(std::vector<FOC_Reading_t>* diff, std::vector<FOC_Reading_t>* edge_max, std::vector<FOC_Reading_t>* edge_min, std::vector<FOC_ChangePoints_t>* cp_max, std::vector<FOC_ChangePoints_t>* cp_min, std::vector<FOC_TDOA_t>* out)
{
    for (int i = 0; i < FOC_DIFF_LAYERS; i++)
        // check if args valid for differentiation
        if (diff[i].size() < N or edge_max[i].size() < N or edge_min[i].size() < N)
            return false;
     
/* Calculate time difference of arrival according to non-extremum suppressioned gradient */

    static std::vector<FOC_Edge_t> edge_max_cps;
    static std::vector<FOC_Edge_t> edge_min_cps;

    int cp_max_size, cp_min_size;
    bool ret_max, ret_min, ret = false;

    for (int order = 1; order <= FOC_DIFF_LAYERS; order++) {
        /* Phase 0: organize change points to a vector */
        reorganize_edge_to_a_vector(edge_max[order-1], N, edge_max_cps);
        reorganize_edge_to_a_vector(edge_min[order-1], N, edge_min_cps);

        /* Phase 1: find contiguous change points of different mox sensors */
        cp_max_size = cp_max[order-1].size(); cp_min_size = cp_min[order-1].size();
        find_pairs_of_change_points(edge_max_cps, cp_max[order-1]);
        find_pairs_of_change_points(edge_min_cps, cp_min[order-1]);

        /* Phase 2: update delta */ 
        ret_max = calculate_delta(cp_max[order-1], cp_max_size, out[order-1], diff[order-1]);
        ret_min = calculate_delta(cp_min[order-1], cp_min_size, out[order-1], diff[order-1]);
        if (ret_max or ret_min)
            ret = true;
    }

    return ret;
}

static bool calculate_delta(std::vector<FOC_ChangePoints_t>& cps, int previous_size, 
        std::vector<FOC_TDOA_t>& delta, std::vector<FOC_Reading_t>& grad)
{
    if (cps.size() <= previous_size)
        return false;

    FOC_TDOA_t new_delta; memset(&new_delta, 0, sizeof(new_delta));
    for (int i = previous_size; i < cps.size(); i++) {
        new_delta.abs[0] = std::abs(grad.at(cps.at(i).index[0]).reading[0]);
        for (int j = 1; j < FOC_NUM_SENSORS; j++) {
            new_delta.toa[j] = ((float)(cps.at(i).index[0] - cps.at(i).index[j]))/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR;
            new_delta.abs[j] = std::abs(grad.at(cps.at(i).index[j]).reading[j]);
        }
        delta.push_back(new_delta);
    }

    return true;
}

static int get_dispersion_of_the_pair(std::vector<FOC_Edge_t>& edge_cps, int start)
{
    int p = 0;
    for (int i = start; i < start+FOC_NUM_SENSORS-1; i++)
        p += abs(edge_cps.at(i).index_time - edge_cps.at(i+1).index_time);
    return p;
}

static bool are_change_points_of_different_sensors(std::vector<FOC_Edge_t>& edge_cps, int start)
{
    if (edge_cps.size() < start + FOC_NUM_SENSORS)
        return false;

    bool are_different_sensors = true;
    for (int idx_sensor = 0; idx_sensor < FOC_NUM_SENSORS-1; idx_sensor++) {
        for (int i = start + idx_sensor + 1; i < start + FOC_NUM_SENSORS; i++) {
            if (edge_cps.at(start+idx_sensor).index_sensor == edge_cps.at(i).index_sensor)
                are_different_sensors = false;
        }
    }

    return are_different_sensors;
}

static bool are_changepoints_overlapping(FOC_ChangePoints_t& a, FOC_ChangePoints_t& b)
{
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        for (int j = 0; j < FOC_NUM_SENSORS; j++) {
            if (a.index[i] == b.index[j])
                return true;
        }
    }

    return false;
}

static bool check_if_its_new_cps(FOC_ChangePoints_t& cp, std::vector<FOC_ChangePoints_t>& cps)
{
    if (cps.size() < 1)
        return true;

    int num_trace_back = FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR/FOC_NUM_SENSORS;

    for (int i = int(cps.size()) - num_trace_back >= 0 ? cps.size() - num_trace_back: 0; i < cps.size(); i++) {
        if (are_changepoints_overlapping(cp, cps.at(i))) {
            if (cps.at(i).disp > cp.disp) {
                memcpy(&cps.at(i), &cp, sizeof(cp));
                return false; // return false to prevent adding again
            }
            else
                return false;
        }
    }

    return true;
}

static void find_pairs_of_change_points(std::vector<FOC_Edge_t>& edge_cps, std::vector<FOC_ChangePoints_t>& cps)
{
    if (edge_cps.size() < FOC_NUM_SENSORS)
        return;

    FOC_ChangePoints_t new_cp;
    bool is_best_pair;
    int i = 0;
    while (i <= edge_cps.size() - FOC_NUM_SENSORS) {
        if (are_change_points_of_different_sensors(edge_cps, i)) { // find a pair, need to check if it is the best pair
            is_best_pair = true;
            for (int j = 1; j < FOC_NUM_SENSORS; j++) {
                if (i+j <= edge_cps.size()-FOC_NUM_SENSORS) {
                    if (are_change_points_of_different_sensors(edge_cps, i+j)) {
                        if (get_dispersion_of_the_pair(edge_cps, i+j) < get_dispersion_of_the_pair(edge_cps, i)) {
                            is_best_pair = false;
                            break;
                        }
                    }
                }
            }
            if (is_best_pair) { // find best pair
                for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
                    new_cp.index[edge_cps.at(i+idx).index_sensor] = edge_cps.at(i+idx).index_time;
                new_cp.disp = get_dispersion_of_the_pair(edge_cps,i);
                if (check_if_its_new_cps(new_cp, cps))
                    cps.push_back(new_cp);
                i = i+FOC_NUM_SENSORS;
            }
            else
                i++;
        }
        else
            i++;
    }
}

static void reorganize_edge_to_a_vector(std::vector<FOC_Reading_t>& edge, int num, std::vector<FOC_Edge_t>& v) 
{
    v.clear();
    FOC_Edge_t  new_edge_cp;
    for (int i = edge.size() - num; i < edge.size(); i++) {
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
            if (edge.at(i).reading[idx] != 0) {
                new_edge_cp.reading = edge.at(i).reading[idx];
                new_edge_cp.index_time = i;
                new_edge_cp.index_sensor = idx;
                v.push_back(new_edge_cp);
            }
        }
    }
}


#if 0
/* Feature extraction
 * Args:
 *      out     features (time of arrival & standard deviation) extracted to output vector
 */
void foc_delta_init(std::vector<FOC_Delta_t>& out)
{
    out.clear();
}
/* Feature extraction
 * Args:
 *      in      input mox signal (smoothed & differentiated)
 *      out     toa & std vector
 * Return:
 *      false   an error happend
 *      true    feature extraction successful
 */
bool foc_delta_update(std::vector<FOC_Reading_t>& in, std::vector<FOC_Delta_t>& out)
{
    // check if args valid for differentiation
    if (in.size() < N)
        return false;

    FOC_Delta_t new_out;

    // standard deviation
    double sum[FOC_NUM_SENSORS] = {0};
    float mean[FOC_NUM_SENSORS] = {0};
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
    {
        //for (int i = in.size() - N; i < in.size(); i++)
        //    sum[idx] += in.at(i).reading[idx];
        //mean[idx] = sum[idx]/N;
        //sum[idx] = 0;
        for (int i = in.size() - N; i < in.size(); i++)
            sum[idx] += pow((in.at(i).reading[idx] - mean[idx]), 2);
        new_out.std[idx] = sqrt(sum[idx]/N);
    }

    // time of arrival
    float reading[FOC_NUM_SENSORS][N];
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        for (int i = in.size() - N; i < in.size(); i++)
            reading[idx][i+N-in.size()] = (in.at(i).reading[idx] - mean[idx]) / new_out.std[idx]; // normalize
    float time[2*N-1];
    for (int i = 1-N; i < N; i++)
        time[i-1+N] = float(i)/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR; // time diff index
    double xcorr[2*N-1]; double temp; int index;
    for (int idx = 1; idx < FOC_NUM_SENSORS; idx++) // sensor_1, sensor_2, ... compare with sensor_0
    {
        // calculate correlation
        for (int t = 1-N; t < N; t++)
        {
            temp = 0;
            for (int i = 0; i < N; i++)
            {
                if (i+t < 0 || i+t >= N)
                    continue;
                else
                    temp += reading[idx][i]*reading[0][i+t];
            }
            xcorr[t+N-1] = temp;
        }
        // find the index of max
        temp = xcorr[0]; index = 0;
        for (int i = 0; i < 2*N-1; i++)
        {
            if (xcorr[i] > temp) {
                temp = xcorr[i];
                index = i;
            }
        }
        // get time diff
        new_out.toa[idx] = time[index];
    }

    // save results
    out.push_back(new_out);

    return true;
}
#endif
