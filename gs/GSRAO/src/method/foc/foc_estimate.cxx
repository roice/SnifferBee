#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <algorithm> // std::max
#include <time.h> // for random seed
#include "method/foc/flying_odor_compass.h"
#include "method/foc/virtual_plume.h"
#include "ziggurat.h" // generate random numbers
#include "cblas.h" // linear algebra
#include "method/foc/vector_rotation.h"

// latest samples
#define POSSIBLE_ANG_RANGE      120.0*M_PI/180.0  // possible angle range to resample/init particles

unsigned int rand_seed; // seed to generate random numbers
float rand_fn[128];
unsigned int rand_kn[128];
float rand_wn[128];

std::vector<FOC_Particle_t> particles;

static void init_particles(unsigned int, int, int, float, float*, std::vector<FOC_Particle_t>&);
static void split_new_particles(unsigned int, float, int, float, float*, float, FOC_Estimation_t&);
static void CalculateRotationMatrix(float*, float*, float*);

void foc_estimate_source_init(std::vector<FOC_Estimation_t>& out)
{ 
    // generate seed for random numbers
    rand_seed = time(NULL);

    srand(rand_seed);

    // setup normal distributed random number generator
    r4_nor_setup ( rand_kn, rand_fn, rand_wn );

    // init particles vector
    FOC_Particle_t new_particle;
    particles.clear();
    for (int i = 0; i < FOC_MAX_PARTICLES; i++) {
        new_particle.plume = new std::vector<FOC_Puff_t>;
        new_particle.plume->reserve(N_PUFFS);
        particles.push_back(new_particle);
    }

    out.clear();
}

/* Estimate horizontal plume dispersion direction according to TOA
 * Args:
 *      feature     including toa
 *      out         direction, out = speed*e[2] = speed*{e_x, e_y}
 * Return:
 *      false       can't determine where the odor comes from
 *      true
 * Equations:
 *                                      1
 *      e_x = +/- --------------------------------------------------
 *                 sqrt(1 + 1/3*((dt_lf+dt_rf)/(dt_lf-dt_rf))^2)
 *                1      dt_lf+dt_rf
 *      e_y = --------- ------------- e_x
 *             sqrt(3)   dt_lf-dt_rf
 *      e_x^2 + e_y^2 = 1
 *      The sign of e_x & e_y is consist with dt_lf or dt_rf:
 *      sign(sqrt(3)e_x + 3e_y) = sign(dt_lf)
 *      sign(-sqrt(3)e_x + 3e_y) = sign(dt_rf)
 */
bool estimate_horizontal_plume_dispersion_direction_according_to_toa(FOC_Feature_t& feature, float* out)
{
    float e_x, e_y, dt_lf = feature.toa[1]-feature.toa[0], dt_rf = feature.toa[2]-feature.toa[0], speed;
    float sqrt_3 = sqrt(3);

    // check if dt is valid
    if (dt_lf == 0 and dt_rf == 0)
        return false;

    // calculate e_x & e_y
    if (dt_lf == dt_rf) {
        if (dt_lf > 0.) {
            e_x = 0.;
            e_y = -1.;
        }
        else {
            e_x = 0.;
            e_y = 1.;
        }
    }
    else {
        float dt_add = dt_lf + dt_rf;
        float dt_minus = dt_lf - dt_rf;
        e_x = 1.0 / sqrt(1 + 1.0/3.0*pow(dt_add/dt_minus, 2));
        e_y = 1.0/sqrt_3*dt_add/dt_minus*e_x;
        
        // determine sign(e_x) & sign(e_y)
        //if (absf(dt_lf) > absf(dt_rf)) { // math.h
        if (std::abs(dt_lf) > std::abs(dt_rf)) { // cmath
            if (std::signbit(sqrt_3*e_x+3*e_y)==std::signbit(dt_lf)) {
                e_x *= -1;
                e_y *= -1;
            }
        }
        else {
            if (std::signbit(-sqrt_3*e_x+3*e_y)==std::signbit(dt_rf)) {
                e_x *= -1;
                e_y *= -1;
            }
        }
    }

    // calculate wind speed
    //if (absf(dt_lf) > absf(dt_rf)) // math.h
    if (std::abs(dt_lf) > std::abs(dt_rf)) // cmath
        //speed = sqrt_3*FOC_RADIUS/2.0*absf(e_x+sqrt_3*e_y)/absf(dt_lf); // math.h
        speed = sqrt_3*FOC_RADIUS/2.0*std::abs(e_x+sqrt_3*e_y)/std::abs(dt_lf); // cmath
    else
        //speed = sqrt_3*FOC_RADIUS/2.0*absf(e_x-sqrt_3*e_y)/absf(dt_rf); // math.h
        speed = sqrt_3*FOC_RADIUS/2.0*std::abs(e_x-sqrt_3*e_y)/std::abs(dt_rf); // cmath

    // check if wind speed is valid
    if (speed > FOC_WIND_MAX or speed < FOC_WIND_MIN)
        return false;

    // save result
    out[0] = e_x; //*speed;
    out[1] = e_y; //*speed;

    return true;
}

int sign(float x) {
    if (x > 0) return 1;
    else if (x < 0) return -1;
    else
        return 0;
}

static float calculate_likelihood_of_two_std(float* x, float* y)
{
#if 0
    float nrm_x = cblas_snrm2(FOC_NUM_SENSORS, x, 1);
    float nrm_y = cblas_snrm2(FOC_NUM_SENSORS, y, 1);
    if (nrm_x == 0 and nrm_y == 0)
        return 1.0;
    else if (nrm_y == 0)
        return 0;
    else if (nrm_x == 0)
        return 0;

    float angle = std::acos(cblas_sdot(FOC_NUM_SENSORS, x, 1, y, 1) / (nrm_x*nrm_y));
    if (angle == 0)
        return 1.0;
    else
        return (1.0 - angle/M_PI)*((nrm_y)/(nrm_x));
#endif

    float dx[FOC_NUM_SENSORS-1], dy[FOC_NUM_SENSORS-1];
    for (int i = 0; i < FOC_NUM_SENSORS-1; i++) {
        dx[i] = x[i+1] - x[i];
        dy[i] = y[i+1] - y[i];
    }
    for (int i = 0; i < FOC_NUM_SENSORS-1; i++) {
        if (sign(dx[i]) != sign(dy[i]))
            return 0;
    }

    float ratio_x[FOC_NUM_SENSORS-1], ratio_y[FOC_NUM_SENSORS-1];
    for (int i = 0; i < FOC_NUM_SENSORS-1; i++) {
        ratio_x[i] = x[i+1] / x[i];
        ratio_y[i] = y[i+1] / y[i];
    }
    float nrm_rx = cblas_snrm2(FOC_NUM_SENSORS-1, ratio_x, 1);
    float nrm_ry = cblas_snrm2(FOC_NUM_SENSORS-1, ratio_y, 1);
    if (nrm_rx == 0 and nrm_ry == 0)
        return 0.0;
    else if (nrm_ry == 0)
        return 0;
    else if (nrm_rx == 0)
        return 0;

    float angle = std::acos(cblas_sdot(FOC_NUM_SENSORS-1, ratio_x, 1, ratio_y, 1) / (nrm_rx*nrm_ry));
    if (angle == 0)
        return 1.0;
    else
        return (1.0 - angle/M_PI);
        //return 1.0/std::exp(angle);
}

static float calculate_likelihood_of_two_tdoa(float* x, float* y)
{/**/

    float dx[FOC_NUM_SENSORS-1], dy[FOC_NUM_SENSORS-1];
    for (int i = 0; i < FOC_NUM_SENSORS-1; i++) {
        dx[i] = x[i+1] - x[i];
        dy[i] = y[i+1] - y[i];
    }
    for (int i = 0; i < FOC_NUM_SENSORS-1; i++) {
        if (sign(dx[i]) != sign(dy[i]))
            return 0;
    }

    float nrm_dx = cblas_snrm2(FOC_NUM_SENSORS-1, dx, 1);
    float nrm_dy = cblas_snrm2(FOC_NUM_SENSORS-1, dy, 1);
    if (nrm_dx == 0 and nrm_dy == 0)
        return 0.0;
    else if (nrm_dy == 0)
        return 0;
    else if (nrm_dx == 0)
        return 0;

    double ratio_nrm_dy_dx = nrm_dx/nrm_dy; 
    if (ratio_nrm_dy_dx > 2.)
        ratio_nrm_dy_dx = 2.; // 0. ~ 2.
    ratio_nrm_dy_dx -= 1.0; // -1. ~ 1.
    float likeli_speed = 1.0 - std::abs(ratio_nrm_dy_dx); // 0. ~ 1.
    likeli_speed = 1.0;

    float angle = std::acos(cblas_sdot(FOC_NUM_SENSORS-1, dx, 1, dy, 1) / (nrm_dx*nrm_dy));
    if (angle == 0)
        return 1.0*likeli_speed;
    else if (angle > M_PI/2.)
        return 0;
    else
        return (1.0 - angle/(M_PI/2.0))*likeli_speed;
        //return 1.0/std::exp(angle);
}

//#define FOC_2D

/* Estimate the 3D direction the odor comes from 
 * Args:
 *      feature     feature of combinations of sensor maxlines, containing TDOA etc. info
 *      est         direction estimation
 */
bool foc_estimate_source_update(std::vector<FOC_Feature_t>& feature, std::vector<FOC_Estimation_t>& data_est, std::vector<FOC_Input_t>& data_raw, int size_of_signal, std::vector<float> data_wt[FOC_NUM_SENSORS][FOC_WT_LEVELS])
{
    if (feature.size() == 0)
        return false;

    float current_time = (float)(size_of_signal+FOC_LEN_WAVELET/2)/(float)(FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR);

    FOC_Estimation_t new_est;

#ifdef FOC_2D
/* ===============  Calculate horizontal flow vector of odor parcel  ===================== */
    float est_horizontal_odor_trans_direction[3] = {0};   // estimated horizontal odor transport direction, e/n
    float est_horizontal_odor_std_deviation[FOC_NUM_SENSORS] = {0};   // standard deviations of odor sensors
    float temp_sum_sum_abs_top_level_wt_value[2] = {0};
    float temp_average_sum_abs_top_level_wt_value[2];
    int count_num_valid_features[2] = {0};
    float temp_sum_hd[2][3] = {0};
    for (int i = 0; i < feature.size(); i++) {
        if (feature.at(i).toa[0] < current_time - FOC_RECENT_TIME_TO_EST)
            continue;
        if(estimate_horizontal_plume_dispersion_direction_according_to_toa(feature.at(i), feature.at(i).direction_p)) {
            memset(feature.at(i).direction, 0, 3*sizeof(float));
            rotate_vector(feature.at(i).direction_p, feature.at(i).direction, data_raw.at(int(feature.at(i).toa[0]*FOC_MOX_DAQ_FREQ)).attitude[2], 0, 0); // vehicle coord to ENU coord

/*
            feature.at(i).direction[0] += data_raw.at(int(feature.at(i).toa[0]*FOC_MOX_DAQ_FREQ)).wind[0];
            feature.at(i).direction[1] += data_raw.at(int(feature.at(i).toa[0]*FOC_MOX_DAQ_FREQ)).wind[1];
*/

            feature.at(i).valid_to_infer_direction = true;
            temp_sum_sum_abs_top_level_wt_value[feature.at(i).type] += feature.at(i).sum_abs_top_level_wt_value;
            count_num_valid_features[feature.at(i).type] ++;
        }
        else
            feature.at(i).valid_to_infer_direction = false;
    }
    for (int sign = 0; sign < 2; sign++) {
        if (count_num_valid_features[sign] == 0)
            continue;
        temp_average_sum_abs_top_level_wt_value[sign] = temp_sum_sum_abs_top_level_wt_value[sign] / count_num_valid_features[sign];
        for (int i = 0; i < feature.size(); i++) {
            if (feature.at(i).toa[0] >= current_time - FOC_RECENT_TIME_TO_EST and feature.at(i).type == sign and feature.at(i).valid_to_infer_direction == true) {
                if (feature.at(i).sum_abs_top_level_wt_value < temp_average_sum_abs_top_level_wt_value[sign])
                    continue;
                for (int j = 0; j < 2; j++)
                    temp_sum_hd[sign][j] += feature.at(i).direction[j]*feature.at(i).credit*std::sqrt(feature.at(i).sum_abs_top_level_wt_value);
            }
        }
    }
    
    float temp_mod_sum_hd;
    for (int sign = 0; sign < 2; sign++) {
        temp_mod_sum_hd = std::sqrt(temp_sum_hd[sign][0]*temp_sum_hd[sign][0]+temp_sum_hd[sign][1]*temp_sum_hd[sign][1]);
        for (int i = 0; i < 2; i++)
            temp_sum_hd[sign][i] /= temp_mod_sum_hd;
    }

#if 0
if (temp_sum_hd[0][1] > 0 or temp_sum_hd[1][1] > 0) {
    printf("current time = %f\n", current_time);
    for (int sign = 0; sign < 2; sign++)
        printf("temp_sum_hd[%d] = {%f, %f}\n", temp_sum_hd[sign][0], temp_sum_hd[sign][1]);
    for (int i = 0; i < feature.size(); i++) {
        if (feature.at(i).toa[0] < current_time - FOC_RECENT_TIME_TO_EST)
            continue;
        if (feature.at(i).valid_to_infer_direction)
            printf("feature.at(%d): type = %d, toa = [ %f, %f, %f ], direction_p = [ %f, %f ], satlwv = %f, credit = %f\n", i, feature.at(i).type, feature.at(i).toa[0], feature.at(i).toa[1], feature.at(i).toa[2], feature.at(i).direction_p[0], feature.at(i).direction_p[1], feature.at(i).sum_abs_top_level_wt_value, feature.at(i).credit);
    }
}
#endif

    float hd[3] = {0};
    for (int i = 0; i < 2; i++)
        /* TODO:
         * Note: for ground robots, it can only consider odor arrival events, so 0.1:0.9 is OK
         *     But for flying robots, 0.3:0.7 is appropriate */
        //hd[i] = 0.3*temp_sum_hd[0][i] + 0.7*temp_sum_hd[1][i];
        hd[i] = 0.2*temp_sum_hd[0][i] + 0.8*temp_sum_hd[1][i];
    
    memcpy(new_est.direction, hd, 3*sizeof(float));
    data_est.push_back(new_est);

#else /* 3D, default */
    float est_horizontal_odor_trans_direction[3] = {0};   // estimated horizontal odor transport direction, e/n
    float est_horizontal_odor_std_deviation[FOC_NUM_SENSORS] = {0};   // standard deviations of odor sensors
/*======== Step 1: calculate horizontal direction ========*/ 
    float temp_sum_sum_abs_top_level_wt_value[2] = {0};
    float temp_average_sum_abs_top_level_wt_value[2];
    int count_num_valid_features[2] = {0};
    // Phase 1: calculate horizontal odor direction for every combination
    for (int i = 0; i < feature.size(); i++) {
        if (feature.at(i).toa[0] < current_time - FOC_RECENT_TIME_TO_EST)
            continue;
        if(estimate_horizontal_plume_dispersion_direction_according_to_toa(feature.at(i), feature.at(i).direction_p)) {
            memset(feature.at(i).direction, 0, 3*sizeof(float));
            rotate_vector(feature.at(i).direction_p, feature.at(i).direction, data_raw.at(int(feature.at(i).toa[0]*FOC_MOX_DAQ_FREQ)).attitude[2], 0, 0); // vehicle coord to ENU coord
            feature.at(i).valid_to_infer_direction = true;
            temp_sum_sum_abs_top_level_wt_value[feature.at(i).type] += feature.at(i).sum_abs_top_level_wt_value;
            count_num_valid_features[feature.at(i).type] ++;
        }
        else
            feature.at(i).valid_to_infer_direction = false;
    }
#if 0 // experiments show better results achieved without this part of routine. -_-!
    // select the important combinations
    for (int sign = 0; sign < 2; sign++) {
        if (count_num_valid_features[sign] == 0)
            continue;
        temp_average_sum_abs_top_level_wt_value[sign] = temp_sum_sum_abs_top_level_wt_value[sign] / count_num_valid_features[sign];
        for (int i = 0; i < feature.size(); i++) {
            if (feature.at(i).toa[0] >= current_time - FOC_RECENT_TIME_TO_EST and feature.at(i).type == sign and feature.at(i).valid_to_infer_direction == true) {
                if (feature.at(i).sum_abs_top_level_wt_value < temp_average_sum_abs_top_level_wt_value[sign])
                    feature.at(i).valid_to_infer_direction = false;
            }
        }
    }
#endif

    // Phase 2: odor & wind joint direction */
    double sigma_odor_hd = 30./180.*M_PI;
    double sigma_wind_hd = 20./180.*M_PI;
    double sigma_joint_hd = std::sqrt(sigma_odor_hd*sigma_odor_hd*sigma_wind_hd*sigma_wind_hd/(sigma_odor_hd*sigma_odor_hd+sigma_wind_hd*sigma_wind_hd));
    float e_odor_hd[2], e_wind_hd[2];
    float angle_odor_wind_hds; // angle between odor and wind directions
    for (int i = 0; i < feature.size(); i++) {
        if (feature.at(i).toa[0] >= current_time - FOC_RECENT_TIME_TO_EST and feature.at(i).valid_to_infer_direction == true) {
            if (cblas_snrm2(2, data_raw.at(int(feature.at(i).toa[0]*FOC_MOX_DAQ_FREQ)).wind, 1) < FOC_WIND_MIN) // no valid wind, use only odor direction
                continue;
            // retrieve odor & wind directions
            for (int j = 0; j < 2; j++) {
                e_odor_hd[j] = feature.at(i).direction[j]/cblas_snrm2(2, feature.at(i).direction, 1);
                e_wind_hd[j] = data_raw.at(int(feature.at(i).toa[0]*FOC_MOX_DAQ_FREQ)).wind[j]/cblas_snrm2(2, data_raw.at(int(feature.at(i).toa[0]*FOC_MOX_DAQ_FREQ)).wind, 1);
            } 
            angle_odor_wind_hds = std::acos(cblas_sdot(2, e_odor_hd, 1, e_wind_hd, 1) / cblas_snrm2(2, e_odor_hd, 1) / cblas_snrm2(2, e_wind_hd, 1));
            if (angle_odor_wind_hds > M_PI/2)
                feature.at(i).valid_to_infer_direction = false;
            else {
                for (int j = 0; j < 2; j++)
                    feature.at(i).direction[j] = (e_odor_hd[j]*sigma_wind_hd*sigma_wind_hd + e_wind_hd[j]*sigma_odor_hd*sigma_odor_hd)/(sigma_odor_hd*sigma_odor_hd+sigma_wind_hd*sigma_wind_hd);
            }
        }
    }

    // Phase 3: save result
    float joint_hd_odor_wind[3] = {0};
    for (int i = 0; i < feature.size(); i++) {
        if (feature.at(i).toa[0] <= current_time - FOC_RECENT_TIME_TO_EST or feature.at(i).valid_to_infer_direction == false)
            continue;
        for (int j = 0; j < 2; j++)
            joint_hd_odor_wind[j] += feature.at(i).direction[j];
    }
    for (int i = 0; i < 2; i++)
        new_est.direction[i] = joint_hd_odor_wind[i];

#if 0 // for debug
    memcpy(new_est.direction, hd, 3*sizeof(float));
    data_est.push_back(new_est);
#endif

/*======== Step 2: Estimate altitude ========*/ 
    float radius_particle_to_robot = 0.5; // m 
   
    // Phase 0: calculate average wind vector & attitude & position
    float average_wind[3] = {0};
    float average_att[3] = {0};
    float average_pos[3] = {0};
    int count_valid_features = 0;
    for (int idx_f = 0; idx_f < feature.size(); idx_f++) {
        if (feature.at(idx_f).toa[0] <= current_time - FOC_RECENT_TIME_TO_EST or feature.at(idx_f).valid_to_infer_direction == false)
            continue;
        for (int i = 0; i < 3; i++) {
            average_wind[i] += data_raw.at(int(feature.at(idx_f).toa[0]*FOC_MOX_DAQ_FREQ)).wind[i];
            average_att[i] += data_raw.at(int(feature.at(idx_f).toa[0]*FOC_MOX_DAQ_FREQ)).attitude[i];
            average_pos[i] += data_raw.at(int(feature.at(idx_f).toa[0]*FOC_MOX_DAQ_FREQ)).position[i];
        }
        count_valid_features++;
    }
    if (count_valid_features == 0) return false;
    for (int i = 0; i < 3; i++) {
        average_wind[i] /= (float)count_valid_features;
        average_att[i] /= (float)count_valid_features;
        average_pos[i] /= (float)count_valid_features;
    }
    // save to result
    memcpy(new_est.pos, average_pos, 3*sizeof(float));
    new_est.t = current_time;
#if 0
printf("average pos = [ %f, %f, %f ]\n", average_pos[0], average_pos[1], average_pos[2]);
#endif

    // Phase 1: add up fluctuations
# if 0    
    float fluct[FOC_NUM_SENSORS] = {0}; // sum of fluctuations (std)
    for (int idx_f = 0; idx_f < feature.size(); idx_f++) {
        if (feature.at(idx_f).toa[0] <= current_time - FOC_RECENT_TIME_TO_EST or feature.at(idx_f).valid_to_infer_direction == false)
            continue;
        for (int i = 0; i < FOC_NUM_SENSORS; i++)
            fluct[i] += feature.at(idx_f).abs_top_level_wt_value[i];
    }
#else
    double sum_fluct[FOC_NUM_SENSORS] = {0};
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        for (int j = 0; j < FOC_LEN_RECENT_INFO; j++)
            sum_fluct[i] += std::abs(data_wt[i][FOC_WT_LEVELS-1].at((int)data_wt[i][FOC_WT_LEVELS-1].size()-FOC_LEN_RECENT_INFO+j>0?(int)data_wt[i][FOC_WT_LEVELS-1].size()-FOC_LEN_RECENT_INFO+j:0));
    }
    float fluct[FOC_NUM_SENSORS] = {0};
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        fluct[i] = sum_fluct[i]/1000.;
#endif

//printf("fluct = [ %f, %f, %f ]\n", fluct[0], fluct[1], fluct[2]);
for (int i = 0; i < FOC_MAX_PARTICLES; i++) {
//    printf("particles.at(%d).std = [ %f, %f, %f ]\n", i, particles.at(i).std.std[0], particles.at(i).std.std[1], particles.at(i).std.std[2]);
}

    // Phase 2: spread particles
    float reverse_joint_hd[3] = {0}; // reverse of joint (odor & wind) horizontal direction
    float norm_reverse_joint_hd = 0;
    for (int i = 0; i < 2; i++)
        reverse_joint_hd[i] = -joint_hd_odor_wind[i];
    for (int i = 0; i < 2; i++)
        norm_reverse_joint_hd += reverse_joint_hd[i]*reverse_joint_hd[i];
    norm_reverse_joint_hd = std::sqrt(norm_reverse_joint_hd);
    for (int i = 0; i < 2; i++)
        reverse_joint_hd[i] /= norm_reverse_joint_hd;
    init_particles(rand_seed, 0, FOC_MAX_PARTICLES, radius_particle_to_robot, reverse_joint_hd, particles);

    // Phase 3: release virtual plumes and calculate weights
    release_virtual_plumes_and_calculate_weights(&particles, average_pos, average_att, average_wind, fluct);

    // Phase 3:
    float sum_weight = 0;
    for (int i = 0; i < FOC_MAX_PARTICLES; i++) {
        sum_weight += particles.at(i).weight;
    }
    if (sum_weight == 0) {
        printf("No weights\n");
        return false;
    }

    for (int i = 0; i < FOC_MAX_PARTICLES; i++) {
//        printf("particles.at(%d).weight = %f, alt = %f\n", i, particles.at(i).weight, particles.at(i).pos_r[2]);
    }

    //   find the max weight
    float temp_weight = 0;
    int idx_max_weight = FOC_MAX_PARTICLES;
    for (int i = 0; i < FOC_MAX_PARTICLES; i++) {
        if (particles.at(i).weight > temp_weight) {
            temp_weight = particles.at(i).weight;
            idx_max_weight = i;
        }
    }
    if (idx_max_weight == FOC_MAX_PARTICLES) {
        printf(" No max found\n");
        return false;
    }

    //printf("max alt = %f\n", std::atan2(particles.at(idx_max_weight).pos_r[2], radius_particle_to_robot)*180./M_PI);

#if 0 
    float hd[3] = {0};
    for (int i = 0; i < feature.size(); i++) {
        if (feature.at(i).toa[0] <= current_time - FOC_RECENT_TIME_TO_EST or feature.at(i).valid_to_infer_direction == false)
            continue;
        for (int j = 0; j < 3; j++)
            hd[j] += feature.at(i).direction[j];
    }
    memcpy(new_est.direction, hd, 3*sizeof(float));
#endif
    //printf("alt = %f\n", hd[2]);

//    if (alt_est_valid_count == 0)
//    return false;
    new_est.direction[2] = std::atan2(particles.at(idx_max_weight).pos_r[2], radius_particle_to_robot);
    if (new_est.direction[0] != new_est.direction[0] or new_est.direction[1] != new_est.direction[1])
        return false;
    new_est.particles = &particles;
    data_est.push_back(new_est);

#endif

#if 0
    
    float temp_ang, possibility_to_survive;
    int num_new_particles_to_split;
    float rot_m_split[9]; // rotation matrix to split important particles
    double temp_weight = 0.0;
    new_out.particles = new std::vector<FOC_Particle_t>;
    new_out.particles->reserve(FOC_MAX_PARTICLES);
    if (out.size() > 0 and out.back().particles->size() > 0) { // not first execution
        // determine whether to resample
        for (int i = 0; i < out.back().particles->size(); i++) {
            temp_weight += std::pow(out.back().particles->at(i).weight, 2);
        }
        if (int(1.0/temp_weight) < FOC_MAX_PARTICLES/2) { // need to resample
            // resample 
            for (int i = 0; i < out.back().particles->size(); i++) { 
                // check if the particle are important or not
                if (out.back().particles->at(i).weight < 1.0/FOC_MAX_PARTICLES) {
                    // not important
                    // check if the particles are outside the possible range
                    temp_ang = std::acos(cblas_sdot(3, out.back().particles->at(i).pos_r, 1, wind_est_reverse, 1) / cblas_snrm2(3, out.back().particles->at(i).pos_r, 1) / cblas_snrm2(3, wind_est_reverse, 1));
                    if (temp_ang > POSSIBLE_ANG_RANGE) // delete this particle
                        continue;
                    // in possibile range, survive at possibility max{N_s*w^i_k, 0.5}
                    possibility_to_survive = std::max(double(FOC_MAX_PARTICLES*out.back().particles->at(i).weight), 0.5);
                    //if (r4_uni(rand_seed) < possibility_to_survive)// goodluck
                    if (((float)rand()/(float)RAND_MAX) < possibility_to_survive)
                        new_out.particles->push_back(out.back().particles->at(i));
                }
                else {
                    // important, split to min{int(N_s*w^i_k),5} new particles around it in 3D normal distribution (std angle = 10 degrees)
                    num_new_particles_to_split = std::min(int(FOC_MAX_PARTICLES*out.back().particles->at(i).weight), 5);
                    CalculateRotationMatrix(unit_z, out.back().particles->at(i).pos_r, rot_m_split);
                    for (int idx = 0; idx < num_new_particles_to_split; idx++) {
                        if (new_out.particles->size() >= FOC_MAX_PARTICLES)
                            continue;
                        else { // generate new particles
                            split_new_particles(rand_seed, 10.0, 1, radius_particle_to_robot, rot_m_split, out.back().particles->at(i).weight, new_out);
                        }
                    }
                }
            }
        }
        if (new_out.particles->size() < FOC_MAX_PARTICLES) // fill up particles
            init_particles(rand_seed, FOC_MAX_PARTICLES-new_out.particles->size(), radius_particle_to_robot, rot_m_init, new_out);
    
        // normalize weights
        temp_weight = 0.0;
        for (int i = 0; i < new_out.particles->size(); i++)
            temp_weight += new_out.particles->at(i).weight;
        for (int i = 0; i < new_out.particles->size(); i++)
            new_out.particles->at(i).weight /= temp_weight;
    }
    else // first run
        // generate FOC_MAX_PARTICLES number of particles
        init_particles(rand_seed, FOC_MAX_PARTICLES, radius_particle_to_robot, rot_m_init, new_out);

/* =================  Step 3: Calculate particle weights ======================
 *  Step 3 Phase 1: Release virtual plumes and calculate virtual tdoa & std */
#ifdef GPU_COMPUTING   // GPU
    release_virtual_plumes_and_calculate_virtual_tdoa_std(new_out.particles, raw.back(), wind_est);
#else   // CPU
    // traverse every particle
    for (int i = 0; i < new_out.particles->size(); i++) {
        // release virtual plume
        release_virtual_plume(new_out.particles->at(i).pos_r, raw.back().position, raw.back().attitude, wind_est, new_out.particles->at(i).plume);
        // calculate tdoa and std
        calculate_virtual_tdoa_and_std(new_out.particles->at(i).plume, raw.back().position, raw.back().attitude, new_out.particles->at(i));
    }
#endif

/* Step 3 Phase 2: Compare tdoa and std to get likelihood */
    float temp_virtual_hd_p[3]; float temp_virtual_hd[3];
    float temp_angle_virtual_real_hd;
    for (int i = 0; i < new_out.particles->size(); i++) { // traverse every particle
        // calculate virtual horizontal source direction
        memset(temp_virtual_hd_p, 0, sizeof(temp_virtual_hd_p));
        memset(temp_virtual_hd, 0, sizeof(temp_virtual_hd));
        if (!estimate_horizontal_direction_according_to_tdoa(new_out.particles->at(i).tdoa, temp_virtual_hd_p)) {
            new_out.particles->at(i).weight = 0;
            continue;
        }
        rotate_vector(temp_virtual_hd_p, temp_virtual_hd, raw.back().attitude[2], 0, 0);
        // get angle between virtual horizontal odor direction and real one
        temp_angle_virtual_real_hd = std::acos((temp_virtual_hd[0]*est_horizontal_odor_trans_direction[0] + temp_virtual_hd[1]*est_horizontal_odor_trans_direction[1] + temp_virtual_hd[2]*est_horizontal_odor_trans_direction[2])/std::sqrt((temp_virtual_hd[0]*temp_virtual_hd[0]+temp_virtual_hd[1]*temp_virtual_hd[1]+temp_virtual_hd[2]*temp_virtual_hd[2])*(est_horizontal_odor_trans_direction[0]*est_horizontal_odor_trans_direction[0]+est_horizontal_odor_trans_direction[1]*est_horizontal_odor_trans_direction[1]+est_horizontal_odor_trans_direction[2]*est_horizontal_odor_trans_direction[2])));
        // get likelihood according to angle
        new_out.particles->at(i).weight = 1.0 - std::abs(temp_angle_virtual_real_hd)/M_PI;
    }

/*  Step 3 Phase 3: Update weights of particles */
    double sum_w = 0;
    for (int i = 0; i < new_out.particles->size(); i++)
        sum_w += new_out.particles->at(i).weight;
    if (sum_w == 0) { // no particles
        return false;
    }
    for (int i = 0; i < new_out.particles->size(); i++)
        new_out.particles->at(i).weight /= sum_w;

/* =======================  Step 4: Maintain main particles ============================== */ 
    // calculate main particle
    FOC_Particle_t  temp_main_particle;
    int temp_main_p_index = 0;
    for (int i = 0; i < new_out.particles->size(); i++) {
        if (new_out.particles->at(i).weight > new_out.particles->at(temp_main_p_index).weight) {
            temp_main_p_index = i;
        }
    }
    memcpy(&temp_main_particle, &(new_out.particles->at(temp_main_p_index)), sizeof(FOC_Particle_t));
    for (int i = 0; i < 3; i++)
        temp_main_particle.pos_r[i] += raw.back().position[i];
    // inherit history particles
    new_out.hist_particles = new std::vector<FOC_Particle_t>;
    new_out.hist_particles->reserve(FOC_MAX_HIST_PARTICLES);
    if (out.size() > 0 and out.back().hist_particles->size() > 0) {
        for (int i = out.back().hist_particles->size() < FOC_MAX_HIST_PARTICLES ? 0 : 1; i < out.back().hist_particles->size(); i++) {
            new_out.hist_particles->push_back(out.back().hist_particles->at(i));
        }
    }
    // add new particle
    new_out.hist_particles->push_back(temp_main_particle);

#if 0
/* Phase 7: Calculate direction of gas source */
    double temp_direction[3] = {0};
    double norm_direction;
    for (int i = 0; i < new_out.particles->size(); i++)
        for (int j = 0; j < 3; j++)
            temp_direction[j] += new_out.particles->at(i).pos_r[j]*new_out.particles->at(i).weight;
    norm_direction = std::sqrt(temp_direction[0]*temp_direction[0]+temp_direction[1]*temp_direction[1]+temp_direction[2]*temp_direction[2]);
    for (int i = 0; i < 3; i++)
        new_out.direction[i] = temp_direction[i] / norm_direction;
#endif

    /*
    memset(new_out.direction, 0, 3*sizeof(float));
    float teminit_particles(unsigned int seed, int num, float radius_particle_to_robot, float* rot_m, std::vector<FOC_Particle_t>& out)p_direct[10][3];
    memset(temp_direct, 0, sizeof(temp_direct));
    for (int i = (int)delta.size() - 10 >= 0 ? delta.size() - 10 : 0; i < delta.size(); i++)
        estimate_horizontal_direction_according_to_tdoa(delta.at(i), temp_direct[delta.size()-i-1]);
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 3; j++)
            new_out.direction[j] += temp_direct[i][j]/10.0;
    */

    out.push_back(new_out);

#endif

    return true;
}

/* init particles
 * Args:
 *      seed        seed to generate random numbers
 *      insert_idx  index of particle vector to begin to insert
 *      num         number of particles to init/generate, N > 0
 *      radius..    radius from robot to particle
 *      e           direction of joint
 *      new_out     the estimation data of this iteration
 */
static void init_particles(unsigned int seed, int insert_idx, int num, float radius_particle_to_robot, float* e, std::vector<FOC_Particle_t>& out)
{
    if (num <= 0) return;
    if (insert_idx < 0) return;
    if (insert_idx + num > FOC_MAX_PARTICLES) return;

    float temp_pos[3];
    float temp_e[3];
    for (int i = insert_idx; i < insert_idx+num; i++) {
        //float angle_z = r4_uni(seed)*POSSIBLE_ANG_RANGE; // particles are uniformly distributed
        //float angle_xy = r4_uni(seed)*2*M_PI;
#if 0
        float angle_z = (((float)rand()/(float)RAND_MAX)-0.5)*POSSIBLE_ANG_RANGE+M_PI/2.0; // particles are uniformly distributed
#else
        float angle_z = ((float)(i-insert_idx)/(float)num)*(M_PI/18.0+M_PI/2.0-M_PI/18.) + M_PI/18.0;
#endif

#if 1
        float angle_xy = std::atan2(e[1], e[0]);
#else
        memset(temp_e, 0, 3*sizeof(float));
        rotate_vector(e, temp_e, ((float)rand()/(float)RAND_MAX-0.5)*M_PI, 0, 0);
        float angle_xy = std::atan2(temp_e[1], temp_e[0]);
#endif

#if 0
        temp_pos[0] = std::cos(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[1] = std::sin(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[2] = std::cos(angle_z)*radius_particle_to_robot;
#else
        temp_pos[0] = e[0]*radius_particle_to_robot;
        temp_pos[1] = e[1]*radius_particle_to_robot;
        temp_pos[2] = radius_particle_to_robot/std::tan(angle_z);
#endif
        memcpy(out.at(i).pos_r, temp_pos, 3*sizeof(float));
        out.at(i).weight = 1.0/FOC_MAX_PARTICLES;
    }
}

/* split new particles
 * Args:
 *      seed        seed to generate random numbers
 *      nor_std     normal standard deviation
 *      num         number of particles to init/generate, N > 0
 *      radius..    radius from robot to particle
 *      rot_m       rotation matrix
 *      weight      weight of particles
 *      new_out     the estimation data of this iteration
 */
static void split_new_particles(unsigned int seed, float nor_std, int num, float radius_particle_to_robot, float* rot_m, float weight, FOC_Estimation_t& new_out)
{
    if (num <= 0) return;

    FOC_Particle_t new_particle;
    float temp_pos[3];
    for (int i = 0; i < num; i++) {
        float angle_z = std::abs(r4_nor(seed, rand_kn, rand_fn, rand_wn))*nor_std*M_PI/180.0; // particles are uniformly distributed
        //float angle_xy = r4_uni(seed)*2*M_PI;
        float angle_xy = ((float)rand()/(float)RAND_MAX)*2*M_PI;
        temp_pos[0] = std::cos(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[1] = std::sin(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[2] = std::cos(angle_z)*radius_particle_to_robot;
        memset(new_particle.pos_r, 0, 3*sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, rot_m, 3, temp_pos, 1, 1.0, new_particle.pos_r, 1);
        new_particle.weight = weight;
        new_particle.plume = new std::vector<FOC_Puff_t>;
        new_particle.plume->reserve(N_PUFFS);
        new_out.particles->push_back(new_particle);
    }
}

/* Vector Rotation related */
static void CrossProduct(float* a, float* b, float* c)
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

static float DotProduct(float* a, float* b)
{
    float result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    return result;
}

static float Normalize(float* v)
{
    float result = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    return result;
}

static void GetRotationMatrix(float angle, float* axis, float* rotationMatrix)
{
    float norm = Normalize(axis);
    float u[3];
    u[0] = axis[0] / norm;
    u[1] = axis[1] / norm;
    u[2] = axis[2] / norm;

    rotationMatrix[0] = std::cos(angle) + u[0] * u[0] * (1 - std::cos(angle));
    rotationMatrix[1] = u[0] * u[1] * (1 - std::cos(angle) - u[2] * std::sin(angle));
    rotationMatrix[2] = u[1] * std::sin(angle) + u[0] * u[2] * (1 - std::cos(angle));

    rotationMatrix[3] = u[2] * std::sin(angle) + u[0] * u[1] * (1 - std::cos(angle));
    rotationMatrix[4] = std::cos(angle) + u[1] * u[1] * (1 - std::cos(angle));
    rotationMatrix[5] = -u[0] * std::sin(angle) + u[1] * u[2] * (1 - std::cos(angle));
      
    rotationMatrix[6] = -u[1] * std::sin(angle) + u[0] * u[2] * (1 - std::cos(angle));
    rotationMatrix[7] = u[0] * std::sin(angle) + u[1] * u[2] * (1 - std::cos(angle));
    rotationMatrix[8] = std::cos(angle) + u[2] * u[2] * (1 - std::cos(angle));
}

static void CalculateRotationMatrix(float* vectorBefore, float* vectorAfter, float* rotationMatrix)
{
    float rotationAxis[3];
    float rotationAngle;
    CrossProduct(vectorBefore, vectorAfter, rotationAxis);
    rotationAngle = std::acos(DotProduct(vectorBefore, vectorAfter) / Normalize(vectorBefore) / Normalize(vectorAfter));
    GetRotationMatrix(rotationAngle, rotationAxis, rotationMatrix);
}


#if 0

static firfilt_rrrf f_ws[2]; // xy, filter for wind speed est

static bool estimate_horizontal_direction_according_to_tdoa(FOC_Delta_t&, FOC_Estimation_t&);

void foc_estimate_wind_speed_through_odor_tdoa_init(std::vector<FOC_Estimation_t>& out)
{
/* create FIR filter for Phase 0: wind estimation */
    for (int i = 0; i < 2; i++)
        f_ws[i] = firfilt_rrrf_create_kaiser(FOC_DELAY*FOC_MOX_DAQ_FREQ, 0.1f/FOC_MOX_DAQ_FREQ*2, 60.0, 0.0);

    out.clear();
}

/* Estimate the direction the odor comes from
 * Args:
 *      in      standard deviation & time of arrival of signals of different sensor
 *      out     horizontal direction & results of particle filter
 */
bool foc_estimate_wind_speed_through_odor_tdoa_update(std::vector<FOC_Delta_t>& in, std::vector<FOC_Estimation_t>& out)
{
    FOC_Estimation_t new_out = {0};

/* Phase 0: estimate horizontal direction according to TOA (time of arrival) */
    if (!estimate_horizontal_direction_according_to_toa(in.back(), new_out)) {
        new_out.valid = false;
        // signal holding
        for (int i = 0; i < 2; i++) {
            //new_out.wind_speed_xy[i] = out.back().wind_speed_xy[i];
            //firfilt_rrrf_push(f_ws[i], new_out.wind_speed_xy[i]);
        }
    }
    else {
        new_out.valid = true;
        // insert new valid data to filter
        for (int i = 0; i < 2; i++) {
            firfilt_rrrf_push(f_ws[i], new_out.wind_speed_xy[i]);
            firfilt_rrrf_execute(f_ws[i], &new_out.wind_speed_filtered_xy[i]);
        }
    }
    //for (int i = 0; i < 2; i++)
        //firfilt_rrrf_execute(f_ws[i], &new_out.wind_speed_filtered_xy[i]);
    if (!new_out.valid) {
        out.push_back(new_out);
        return false;
    }

/* Phase 1:  */
    
    out.push_back(new_out);
    return true;
}



#endif
