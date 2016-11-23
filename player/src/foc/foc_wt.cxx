#include <vector>
#include <string>
#include "wavelet2s.h"
#include "flying_odor_compass.h"

static std::vector<double> recent_readings[FOC_NUM_SENSORS];

void foc_wt_init(std::vector<double>* out, std::vector<int>* length, std::vector<double>* flag)
{
    // clear outputs
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        out[idx].clear();
        length[idx].clear();
        flag[idx].clear();
    }

    // reserve space for recent readings
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        recent_readings[idx].reserve(FOC_LEN_RECENT_INFO);
    }
}

bool foc_wt_update(std::vector<double>* readings, std::vector<double>* out, std::vector<int>* length, std::vector<double>* flag)
{
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        if (readings[idx].size() < FOC_LEN_RECENT_INFO)
            return false;
    }

    // clear outputs
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        out[idx].clear();
        length[idx].clear();
        flag[idx].clear();
    }

    
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        recent_readings[idx].clear();
        std::copy(readings[idx].end()-FOC_LEN_RECENT_INFO, readings[idx].end(), back_inserter(recent_readings[idx]));
    }

    // wavelet transform
    std::string wavelet_name = "coif4";
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        dwt(recent_readings[idx], FOC_WT_LEVEL, wavelet_name, out[idx], flag[idx], length[idx]);
    }

    return true;
}
