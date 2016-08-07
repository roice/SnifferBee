#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include "flying_odor_compass.h"

#define N   (FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR)

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
