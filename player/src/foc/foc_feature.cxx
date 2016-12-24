#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "foc/flying_odor_compass.h"

#define FOC_FEATURE_ML_LEVELS_THRESHOLD     30
#define FOC_FEATURE_ML_VALUE_THRESHOLD      0.6
#define FOC_FEATURE_MLS_T_THRESHOLD         (FOC_RADIUS/FOC_WIND_MIN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR) // 1 s

void foc_feature_extraction_init(std::vector<FOC_Feature_t> data_feature)
{
    data_feature.clear();
}

typedef struct {
    int idx[FOC_NUM_SENSORS]; // index of data_maxline
    float sum_abs_tdoa; // sum of abs(tdoa), s
    float sum_llh_mls_t; // sum of likelihood of time of maxlines, 0. ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float sum_llh_mls_value; // sum of likelihood of value of maxlines, 0. ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float sum_diff_mls_levels; // sum of diff of levels of maxlines, 0. ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float belief;   // 0. ~ 1. 
    bool possible; // true or not
} Comb_Maxlines_t;

static float calculate_likelihood_t_of_two_mls(FOC_Maxline_t &ml_a, FOC_Maxline_t &ml_b)
{
}

bool foc_feature_extraction_update(std::vector<FOC_Maxline_t> data_maxline[FOC_NUM_SENSORS][2], std::vector<FOC_Feature_t> data_feature, int size_of_signal)
{
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        for (int j = 0; j < 2; j++)
            if (data_maxline[i][j].size() == 0) return false;

    data_feature.clear();

/* Cluster, combination of maxlines of different sensors */
    // step 1: traverse combinations
    static std::vector<Comb_Maxlines_t> comb_mls[2]; // combination of maxlines/minlines
    static std::vector<int> idx_mls[FOC_NUM_SENSORS];
    int temp_idx[FOC_NUM_SENSORS];
    int iterator;
    bool have_maxlines_after_thresholding;
    Comb_Maxlines_t new_comb_mls;
    std::vector<bool> mask(FOC_NUM_SENSORS); // sort 2 from FOC_NUM_SENSORS 
    int temp_t;
    for (int sign = 0; sign < 2; sign++) {
        // threshold maxlines
        for (int idx_s = 0; idx_s < FOC_NUM_SENSORS; idx_s++) {
            idx_mls[idx_s].clear();
            for (int i = data_maxline[idx_s][sign].size()-1; i >= 0; i--) {
                // recent info
                if (data_maxline[idx_s][sign].at(i).t[0] < size_of_signal - FOC_LEN_RECENT_INFO)
                    break;
                // maxline threshold, level & value
                if (data_maxline[idx_s][sign].at(i).levels < FOC_FEATURE_ML_LEVELS_THRESHOLD or data_maxline[idx_s][sign].at(i).value[data_maxline[idx_s][sign].at(i).levels-1] < FOC_FEATURE_ML_VALUE_THRESHOLD)
                    continue;
                idx_mls[idx_s].push_back(i);
            } 
        }
        have_maxlines_after_thresholding = true;
        for (int idx_s = 0; idx_s < FOC_NUM_SENSORS; idx_s++)
            if (idx_mls[idx_s].size() == 0) {
                have_maxlines_after_thresholding = false; // no suitable maxlines
                break;
            }
        if (!have_maxlines_after_thresholding)
            break;
        
        // all combinations
        comb_mls[sign].clear();
        memset(temp_idx, 0, sizeof(temp_idx));
        while (true)
        {
            for (int i = 0; i < FOC_NUM_SENSORS; i++)
                new_comb_mls.idx[i] = idx_mls[i].at(temp_idx[i]);
            comb_mls[sign].push_back(new_comb_mls);
            for (iterator = FOC_NUM_SENSORS-1 ; iterator >= 0 ; iterator--) {
                if (++temp_idx[iterator] < idx_mls[iterator].size())
                    break;
                else
                    temp_idx[iterator]=0;
            }
            if (iterator < 0)
                break;
        }

        // t thresholding 
        for (int i = 0; i < comb_mls[sign].size(); i++) {
            std::fill(mask.begin(), mask.end(), false);
            std::fill(mask.begin(), mask.begin()+2, true); // FOC_NUM_SENSORS > 2
            comb_mls[sign].at(i).sum_abs_tdoa = 0;
            comb_mls[sign].at(i).possible = true;
            do {
                temp_t = 0;
                for (int j = 0; j < FOC_NUM_SENSORS; j++) {
                    if (mask[j])
                        temp_t = comb_mls[sign].at(i).idx[j] - temp_t;
                }
                if (std::abs(temp_t) > FOC_FEATURE_MLS_T_THRESHOLD)
                    comb_mls[sign].at(i).possible = false;
                comb_mls[sign].at(i).sum_abs_tdoa += ((float)std::abs(temp_t))/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR;
            } while (std::prev_permutation(mask.begin(), mask.end()));
        }

#if 0
// DEBUG
for (int i = 0; i < comb_mls[sign].size(); i++) {
    printf("comb_mls[%d].at(%d).t = ", sign, i);
    for (int j = 0; j < FOC_NUM_SENSORS; j++)
        printf("%d ", comb_mls[sign].at(i).idx[j]);
    if (comb_mls[sign].at(i).possible)
        printf(", true\n");
    else
        printf(", false\n");
}
#endif
        // calculate likelihood of maxlines in a combination
        for (int i = 0; i < comb_mls[sign].size(); i++) {
            if (!comb_mls[sign].at(i).possible)
                continue;
            std::fill(mask.begin(), mask.end(), false);
            std::fill(mask.begin(), mask.begin()+2, true); // FOC_NUM_SENSORS > 2
            comb_mls[sign].at(i).sum_llh_mls_t = 0;
            comb_mls[sign].at(i).sum_llh_mls_value = 0;
            comb_mls[sign].at(i).sum_diff_mls_levels = 0;
            do { // sort 2 mls
                iterator = 0;
                for (int j = 0; j < FOC_NUM_SENSORS; j++) { // sort 2 mls
                    if (mask[j])
                        temp_idx[iterator++] = j;
                } // temp_idx[0 ~ 1] contains index of sensors
                // calculate likelihood of t
                calculate_likelihood_t_of_two_mls(data_maxline[temp_idx[0]][sign].at(comb_mls[sign].at(i).idx[temp_idx[0]]), data_maxline[temp_idx[1]][sign].at(comb_mls[sign].at(i).idx[temp_idx[1]]));
            } while (std::prev_permutation(mask.begin(), mask.end()));
        }


    }

    if (data_feature.size() > 0)
        return true;
    else
        return false;
}
