#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "foc/flying_odor_compass.h"

#define FOC_FEATURE_ML_LEVELS_THRESHOLD     30
#define FOC_FEATURE_ML_VALUE_THRESHOLD      0.1
#define FOC_FEATURE_MLS_T_THRESHOLD         (FOC_RADIUS/FOC_WIND_MIN*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR) // 1 s

void foc_feature_extraction_init(std::vector<FOC_Feature_t> &data_feature)
{
    data_feature.clear();
}

typedef struct {
    int idx[FOC_NUM_SENSORS]; // index of data_maxline
    float sum_abs_tdoa; // sum of abs(tdoa), s
    float sum_llh_mls_t; // sum of likelihood of time of maxlines, -FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)  ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float sum_llh_mls_value; // sum of likelihood of value of maxlines, -FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!) ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float sum_llh_mls_levels; // sum of diff of levels of maxlines, 0. ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float belief;   // 0. ~ 1. 
    bool possible; // true or not
} Comb_Maxlines_t;

// Pearson Correlation Coefficient
// Out:  -1. ~ 1.
static float calculate_likelihood_t_of_two_mls(FOC_Maxline_t &ml_x, FOC_Maxline_t &ml_y)
{
    float x_mean = 0, y_mean = 0;
    float temp_x[FOC_WT_LEVELS] = {0}, temp_y[FOC_WT_LEVELS] = {0};
    // \bar{x}, \bar{y}
    for (int i = 0; i < ml_x.levels; i++) // levels > 0
        x_mean += ml_x.t[i];
    x_mean /= ml_x.levels;
    for (int i = 0; i < ml_y.levels; i++) // levels > 0
        y_mean += ml_y.t[i];
    y_mean /= ml_y.levels;
    // x_i-\bar{x}, y_i-\bar{y}
    for (int i = 0; i < ml_x.levels; i++)
        temp_x[i] = ml_x.t[i] - x_mean;
    for (int i = 0; i < ml_y.levels; i++)
        temp_y[i] = ml_y.t[i] - y_mean;
    // sqrt(sum{temp_x^2})...
    float sqrt_sum_xx = 0, sqrt_sum_yy = 0;
    for (int i = 0; i < ml_x.levels; i++)
        sqrt_sum_xx += temp_x[i]*temp_x[i];
    sqrt_sum_xx = std::sqrt(sqrt_sum_xx);
    for (int i = 0; i < ml_y.levels; i++)
        sqrt_sum_yy += temp_y[i]*temp_y[i];
    sqrt_sum_yy = std::sqrt(sqrt_sum_yy);
    // sum{temp_x*temp_y}
    float sum_xy = 0;
    for (int i = 0; i < FOC_WT_LEVELS; i++)
        sum_xy += temp_x[i]*temp_y[i];

    if (sqrt_sum_xx == 0. and sqrt_sum_yy == 0.)
        return 1.;
    else if (sqrt_sum_xx == 0. or sqrt_sum_yy == 0.) {
        if (sqrt_sum_xx+sqrt_sum_yy > FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR/2.)
            return 0.;
        else
            return 1 - (sqrt_sum_xx+sqrt_sum_yy)/(FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR/2.);
    }

    return sum_xy/(sqrt_sum_xx*sqrt_sum_yy);
}

static float calculate_likelihood_value_of_two_mls(FOC_Maxline_t &ml_x, FOC_Maxline_t &ml_y)
{
    float x_mean = 0, y_mean = 0;
    float temp_x[FOC_WT_LEVELS] = {0}, temp_y[FOC_WT_LEVELS] = {0};
    // \bar{x}, \bar{y}
    for (int i = 0; i < ml_x.levels; i++) // levels > 0
        x_mean += ml_x.value[i];
    x_mean /= ml_x.levels;
    for (int i = 0; i < ml_y.levels; i++) // levels > 0
        y_mean += ml_y.value[i];
    y_mean /= ml_y.levels;
    // x_i-\bar{x}, y_i-\bar{y}
    for (int i = 0; i < ml_x.levels; i++)
        temp_x[i] = ml_x.value[i] - x_mean;
    for (int i = 0; i < ml_y.levels; i++)
        temp_y[i] = ml_y.value[i] - y_mean;
    // sqrt(sum{temp_x^2})...
    float sqrt_sum_xx = 0, sqrt_sum_yy = 0;
    for (int i = 0; i < ml_x.levels; i++)
        sqrt_sum_xx += temp_x[i]*temp_x[i];
    sqrt_sum_xx = std::sqrt(sqrt_sum_xx);
    for (int i = 0; i < ml_y.levels; i++)
        sqrt_sum_yy += temp_y[i]*temp_y[i];
    sqrt_sum_yy = std::sqrt(sqrt_sum_yy);
    // sum{temp_x*temp_y}
    float sum_xy = 0;
    for (int i = 0; i < FOC_WT_LEVELS; i++)
        sum_xy += temp_x[i]*temp_y[i];

    if (sqrt_sum_xx == 0. and sqrt_sum_yy == 0.)
        return 0.;
    else if (sqrt_sum_xx == 0. or sqrt_sum_yy == 0.)
        return 0.;

    return sum_xy/(sqrt_sum_xx*sqrt_sum_yy);
}

int factorial(int n)
{
    if(n > 1)
        return n * factorial(n - 1);
    else
        return 1;
}

bool foc_feature_extraction_update(std::vector<FOC_Maxline_t> data_maxline[FOC_NUM_SENSORS][2], std::vector<FOC_Feature_t> &data_feature, int size_of_signal)
{
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        for (int j = 0; j < 2; j++)
            if (data_maxline[i][j].size() == 0) return false;

    data_feature.clear();
    FOC_Feature_t new_feature;

    int num_permutation = factorial(FOC_NUM_SENSORS)/factorial(2)/factorial(FOC_NUM_SENSORS-2);

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
    float temp_belief_comb; int temp_idx_comb;
    for (int sign = 0; sign < 2; sign++) {
        // threshold maxlines
        for (int idx_s = 0; idx_s < FOC_NUM_SENSORS; idx_s++) {
            idx_mls[idx_s].clear();
            for (int i = data_maxline[idx_s][sign].size()-1; i >= 0; i--) {
                // recent info
                if (data_maxline[idx_s][sign].at(i).t[0] < size_of_signal - FOC_LEN_RECENT_INFO)
                    break;
                // maxline threshold, level & value
                if (data_maxline[idx_s][sign].at(i).levels < FOC_FEATURE_ML_LEVELS_THRESHOLD or std::abs(data_maxline[idx_s][sign].at(i).value[data_maxline[idx_s][sign].at(i).levels-1]) < FOC_FEATURE_ML_VALUE_THRESHOLD)
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
            continue;
        
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
                        temp_t = data_maxline[j][sign].at(comb_mls[sign].at(i).idx[j]).t[0] - temp_t;
                }
                if (std::abs(temp_t) > FOC_FEATURE_MLS_T_THRESHOLD)
                    comb_mls[sign].at(i).possible = false;
                comb_mls[sign].at(i).sum_abs_tdoa += ((float)std::abs(temp_t))/FOC_MOX_DAQ_FREQ/FOC_MOX_INTERP_FACTOR;
            } while (std::prev_permutation(mask.begin(), mask.end()));
        }

        // calculate likelihood of maxlines in a combination
        for (int i = 0; i < comb_mls[sign].size(); i++) {
            if (!comb_mls[sign].at(i).possible)
                continue;
            std::fill(mask.begin(), mask.end(), false);
            std::fill(mask.begin(), mask.begin()+2, true); // FOC_NUM_SENSORS > 2
            comb_mls[sign].at(i).sum_llh_mls_t = 0;
            comb_mls[sign].at(i).sum_llh_mls_value = 0;
            comb_mls[sign].at(i).sum_llh_mls_levels = 0;
            do { // sort 2 mls
                iterator = 0;
                for (int j = 0; j < FOC_NUM_SENSORS; j++) { // sort 2 mls
                    if (mask[j])
                        temp_idx[iterator++] = j;
                } // temp_idx[0 ~ 1] contains index of sensors
                // calculate likelihood of t
                comb_mls[sign].at(i).sum_llh_mls_t += calculate_likelihood_t_of_two_mls(data_maxline[temp_idx[0]][sign].at(comb_mls[sign].at(i).idx[temp_idx[0]]), data_maxline[temp_idx[1]][sign].at(comb_mls[sign].at(i).idx[temp_idx[1]]));
                // calculate likelihood of value
                comb_mls[sign].at(i).sum_llh_mls_value += calculate_likelihood_value_of_two_mls(data_maxline[temp_idx[0]][sign].at(comb_mls[sign].at(i).idx[temp_idx[0]]), data_maxline[temp_idx[1]][sign].at(comb_mls[sign].at(i).idx[temp_idx[1]]));
                // calculate diff of levels
                comb_mls[sign].at(i).sum_llh_mls_levels += 1.0 - std::abs(data_maxline[temp_idx[0]][sign].at(comb_mls[sign].at(i).idx[temp_idx[0]]).levels - data_maxline[temp_idx[1]][sign].at(comb_mls[sign].at(i).idx[temp_idx[1]]).levels)/(float)FOC_WT_LEVELS;
            } while (std::prev_permutation(mask.begin(), mask.end()));
            // calculate overall likelihood
            comb_mls[sign].at(i).belief = (comb_mls[sign].at(i).sum_llh_mls_levels*0.45+comb_mls[sign].at(i).sum_llh_mls_value*0.45+comb_mls[sign].at(i).sum_llh_mls_t*0.1)/(float)num_permutation;
        }

#if 0
// DEBUG
for (int i = 0; i < comb_mls[sign].size(); i++) {
    if (comb_mls[sign].at(i).possible) {
        printf("comb_mls[%d].at(%d).t = ", sign, i);
        for (int j = 0; j < FOC_NUM_SENSORS; j++)
            printf("%d ", comb_mls[sign].at(i).idx[j]);
        printf("llh_t = %f, ", comb_mls[sign].at(i).sum_llh_mls_t);
        printf("llh_v = %f, ", comb_mls[sign].at(i).sum_llh_mls_value);
        printf("llh_l = %f, ", comb_mls[sign].at(i).sum_llh_mls_levels);
        printf("belief = %f\n", comb_mls[sign].at(i).belief);
    }
}
#endif

        // find optimum combination
        while (true) {
            temp_belief_comb = 0;
            for (int i = 0; i < comb_mls[sign].size(); i++) {// find max belief comb
                if (!comb_mls[sign].at(i).possible) continue;
                if (comb_mls[sign].at(i).belief > temp_belief_comb) {
                    temp_belief_comb = comb_mls[sign].at(i).belief;
                    temp_idx_comb = i;
                }
            } 
            if (temp_belief_comb == 0) // no valid combs found, end finding
                break;
            // save to data_feature
            memset(&new_feature, 0, sizeof(new_feature));
            new_feature.type = sign;
            memcpy(new_feature.idx_ml, comb_mls[sign].at(temp_idx_comb).idx, FOC_NUM_SENSORS*sizeof(int));
            for (int i = 0; i < FOC_NUM_SENSORS; i++)
                new_feature.toa[i] = (float)(data_maxline[i][sign].at(comb_mls[sign].at(temp_idx_comb).idx[i]).t[0]+FOC_LEN_WAVELET/2)/(float)(FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);
            new_feature.sum_abs_tdoa = comb_mls[sign].at(temp_idx_comb).sum_abs_tdoa;
            for (int i = 0; i < FOC_NUM_SENSORS; i++)
                new_feature.sum_abs_top_level_wt_value += std::abs(data_maxline[i][sign].at(comb_mls[sign].at(temp_idx_comb).idx[i]).value[data_maxline[i][sign].at(comb_mls[sign].at(temp_idx_comb).idx[i]).levels-1]);
            new_feature.sum_llh_mls_t = comb_mls[sign].at(temp_idx_comb).sum_llh_mls_t;
            new_feature.sum_llh_mls_value = comb_mls[sign].at(temp_idx_comb).sum_llh_mls_value;
            new_feature.sum_llh_mls_levels = comb_mls[sign].at(temp_idx_comb).sum_llh_mls_levels;
            data_feature.push_back(new_feature);
            // chop off other combs related to maxlines of this comb
            for (int i = 0; i < comb_mls[sign].size(); i++) {
                if (!comb_mls[sign].at(i).possible) continue;
                for (int j = 0; j < FOC_NUM_SENSORS; j++)
                    if (comb_mls[sign].at(i).idx[j] == comb_mls[sign].at(temp_idx_comb).idx[j]) {
                        comb_mls[sign].at(i).possible = false;
                        break;
                    }
            }
        }
    }

#if 0
// DEBUG
for (int sign = 0; sign < 2; sign++)
    printf("comb_mls[%d].size() = %d\n", sign, comb_mls[sign].size());

if (data_feature.size() > 0) {
    for (int i = 0; i < data_feature.size(); i++) {
        printf("data_feature.at(%d).type = %d, t = ", i, data_feature.at(i).type);
        for (int j = 0; j < FOC_NUM_SENSORS; j++)
            printf("%d ", data_maxline[j][data_feature.at(i).type].at(data_feature.at(i).idx_ml[j]).t[0]);
        printf(", llh_t = %f, ", data_feature.at(i).sum_llh_mls_t);
        printf(", llh_v = %f", data_feature.at(i).sum_llh_mls_value);
        printf(", llh_l = %f\n", data_feature.at(i).sum_llh_mls_levels);
    }
}
else
    printf("no feature\n");
#endif
  
    if (data_feature.size() > 0) {
        // calculate credit of every feature
        float temp_sum_credit = 0;
        for (int i = 0; i < data_feature.size(); i++) {
            data_feature.at(i).credit = (data_feature.at(i).sum_llh_mls_t*0.1 + data_feature.at(i).sum_llh_mls_value*0.45 + data_feature.at(i).sum_llh_mls_levels*0.45)*(data_feature.at(i).type == 0?0.3:0.7);
            temp_sum_credit += data_feature.at(i).credit;
        }
        for (int i = 0; i < data_feature.size(); i++) {
            data_feature.at(i).credit /= temp_sum_credit;
        }
        return true;
    }
    else
        return false;
}
