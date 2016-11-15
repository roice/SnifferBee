#include <vector>
#include <string>
#include "wavelet2s.h"
#include "flying_odor_compass.h"

void foc_wt_init(std::vector<double>* out, std::vector<int>* length, std::vector<double>* flag)
{
    // clear outputs
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        out[idx].clear();
        length[idx].clear();
        flag[idx].clear();
    }
}

bool foc_wt_update(std::vector<double>* readings, std::vector<double>* out, std::vector<int>* length, std::vector<double>* flag)
{
    // clear outputs
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        out[idx].clear();
        length[idx].clear();
        flag[idx].clear();
    }

    // wavelet transform
    std::string wavelet_name = "haar";
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        dwt(readings[idx], int(FOC_WT_LEVEL), wavelet_name, out[idx], flag[idx], length[idx]);
    }

    return true;
}
