#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <algorithm> // std::max
#include <time.h> // for random seed
#include "flying_odor_compass.h"
#include "foc/virtual_plume.h"
#include "ziggurat.h" // generate random numbers
#include "cblas.h" // linear algebra
#include "foc/vector_rotation.h"

// latest samples
#define POSSIBLE_ANG_RANGE      90.0*M_PI/180.0  // possible angle range to resample/init particles

unsigned int rand_seed; // seed to generate random numbers
float rand_fn[128];
unsigned int rand_kn[128];
float rand_wn[128];

static void init_particles(unsigned int, int, float, float*, FOC_Estimation_t&);
static void split_new_particles(unsigned int, float, int, float, float*, float, FOC_Estimation_t&);
static void CalculateRotationMatrix(float*, float*, float*);

void foc_estimate_source_direction_init(std::vector<FOC_Estimation_t>& out)
{ 
    // generate seed for random numbers
    rand_seed = time(NULL);

    srand(rand_seed);

    // setup normal distributed random number generator
    r4_nor_setup ( rand_kn, rand_fn, rand_wn );

    out.clear();
}

/* Estimate the 3D direction the odor comes from 
 * Args:
 *      in      standard deviation & time of arrival of signals of different sensors
 *      out     direction estimation
 */
bool foc_estimate_source_direction_update(std::vector<FOC_Input_t>& raw, std::vector<FOC_STD_t>* std, std::vector<FOC_TDOA_t>* tdoa, std::vector<FOC_Wind_t>& wind, std::vector<FOC_Estimation_t>& out)
{
    FOC_Estimation_t new_out;

/* ===============  Step 1: Prepare to release particles  =====================
 *  Step 1 Phase 1: estimate horizontal direction where the odor comes from */
    float est_horizontal_odor_trans_direction[3] = {0};   // estimated horizontal odor transport direction, e/n
    float est_horizontal_odor_std_deviation[FOC_NUM_SENSORS] = {0};   // standard deviations of odor sensors
    float est_horizontal_odor_trans_direction_clustering = 0;   // clustering of odor trans direction
    int deep_traceback = FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR;
    int index_tdoa;
    float temp_hd[3], temp_hd_p[3];
    double temp_sum_hd[3] = {0}; double temp_norm_hd = 0;
    std::vector<FOC_Vector_t>* hds = new std::vector<FOC_Vector_t>;
    FOC_Vector_t new_hd_v = {0}; int temp_count = 0;
    for (int grp = 3; grp < FOC_DIFF_GROUPS; grp++)
    for (int lyr = 0; lyr < FOC_DIFF_LAYERS_PER_GROUP; lyr++) {
        memset(temp_sum_hd, 0, sizeof(temp_sum_hd));
        index_tdoa = tdoa[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].size() -1;
        while (index_tdoa >= 0) {
            if (tdoa[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(index_tdoa).index > tdoa[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].back().index - deep_traceback) {
                if (tdoa[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(index_tdoa).dt < FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR*FOC_RADIUS/FOC_WIND_MAX) {
                    index_tdoa --;
                    continue;
                }
                memset(temp_hd_p, 0, sizeof(temp_hd_p));
                memset(temp_hd, 0, sizeof(temp_hd));
                if (!estimate_horizontal_direction_according_to_tdoa(tdoa[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(index_tdoa), temp_hd_p)) {
                    index_tdoa --;
                    continue;
                }
                rotate_vector(temp_hd_p, temp_hd, raw.at(tdoa[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].at(index_tdoa).index/FOC_MOX_INTERP_FACTOR).attitude[2], 0, 0);
                for (int j = 0; j < 3; j++)
                    temp_sum_hd[j] += temp_hd[j];
                // save temp_hd to calculate belief later
                new_hd_v.x = temp_hd[0];
                new_hd_v.y = temp_hd[1];
                hds->push_back(new_hd_v);
            }
            index_tdoa --;
        }    
        temp_norm_hd = std::sqrt(temp_sum_hd[0]*temp_sum_hd[0] + temp_sum_hd[1]*temp_sum_hd[1] + temp_sum_hd[2]*temp_sum_hd[2]);
        for (int j = 0; j < 3; j++)
            est_horizontal_odor_trans_direction[j] += temp_sum_hd[j] / temp_norm_hd;
        for (int j = 0; j < FOC_NUM_SENSORS; j++)
            est_horizontal_odor_std_deviation[j] += std[grp*FOC_DIFF_LAYERS_PER_GROUP+lyr].back().std[j];
    }
    if (hds->size() > 0)
    {
        // calculate belief
        for (int i = 0; i < hds->size(); i++) {
            if ( std::abs(std::acos((hds->at(i).x*est_horizontal_odor_trans_direction[0]+hds->at(i).y*est_horizontal_odor_trans_direction[1])/std::sqrt((hds->at(i).x*hds->at(i).x+hds->at(i).y*hds->at(i).y)*(est_horizontal_odor_trans_direction[0]*est_horizontal_odor_trans_direction[0]+est_horizontal_odor_trans_direction[1]*est_horizontal_odor_trans_direction[1])))) < 60.0*M_PI/180.0 )
                temp_count ++;
        }
        est_horizontal_odor_trans_direction_clustering = (float)temp_count / (float)hds->size();
        delete hds;
    }
    else {
        delete hds;
        return false;
    }
    // save wind info into new_out
    new_out.wind[0] = wind.back().wind[0];
    new_out.wind[1] = wind.back().wind[1];

/*  Step 1 Phase 2: compute the center of the recent trajectory of the robot */
    float robot_traj_center[3] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = raw.size()-FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ; j < raw.size(); j++)
            robot_traj_center[i] += raw.at(j).position[i];
        robot_traj_center[i] /= FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ;
    }

/*  Step 1 Phase 3: compute the max deviation of the robot trajectory, and compute the radius of the particles to the robot */
    float robot_traj_deviation = 0; float temp_distance;
    for (int j = raw.size()-FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ; j < raw.size(); j++) {
        temp_distance = std::pow(raw.at(j).position[0]-robot_traj_center[0], 2) + std::pow(raw.at(j).position[1]-robot_traj_center[1], 2) + std::pow(raw.at(j).position[2]-robot_traj_center[2], 2);
        if (temp_distance > robot_traj_deviation)   // find max deviation
            robot_traj_deviation = temp_distance;
    }
    robot_traj_deviation = sqrt(robot_traj_deviation);
    float radius_particle_to_robot = robot_traj_deviation + 3*FOC_RADIUS; // 3 is empirical value
#ifdef FOC_ESTIMATE_DEBUG
    new_out.radius_particle_to_robot = radius_particle_to_robot;
#endif

/*  Step 1 Phase 4: calculate wind */
    float wind_est[3];
    for (int i = 0; i < 3; i++)
        wind_est[i] = wind.back().wind[i];

/* ====================  Step 2: init particles / resample particles =================== */ 
    // compute rotation matrix to rotate unit_z to reverse direction of wind
    float unit_z[3] = {0., 0., 1.};
    float wind_est_reverse[3] = {-wind.back().wind[0], -wind.back().wind[1], -wind.back().wind[2]};
    float rot_m_init[9]; // rotation matrix to generate new particles
    CalculateRotationMatrix(unit_z, wind_est_reverse, rot_m_init);
    // Resampling / init particles
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
#if 1   // GPU
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
    float temp_direct[10][3];
    memset(temp_direct, 0, sizeof(temp_direct));
    for (int i = (int)delta.size() - 10 >= 0 ? delta.size() - 10 : 0; i < delta.size(); i++)
        estimate_horizontal_direction_according_to_tdoa(delta.at(i), temp_direct[delta.size()-i-1]);
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 3; j++)
            new_out.direction[j] += temp_direct[i][j]/10.0;
    */

    out.push_back(new_out);

    return true;
}

/* init particles
 * Args:
 *      seed        seed to generate random numbers
 *      num         number of particles to init/generate, N > 0
 *      radius..    radius from robot to particle
 *      rot_m       rotation matrix
 *      new_out     the estimation data of this iteration
 */
static void init_particles(unsigned int seed, int num, float radius_particle_to_robot, float* rot_m, FOC_Estimation_t& new_out)
{
    if (num <= 0) return;

    FOC_Particle_t new_particle;
    float temp_pos[3];
    for (int i = 0; i < num; i++) {
        //float angle_z = r4_uni(seed)*POSSIBLE_ANG_RANGE; // particles are uniformly distributed
        //float angle_xy = r4_uni(seed)*2*M_PI;
        float angle_z = ((float)rand()/(float)RAND_MAX)*POSSIBLE_ANG_RANGE; // particles are uniformly distributed
        float angle_xy = ((float)rand()/(float)RAND_MAX)*2*M_PI;
        temp_pos[0] = std::cos(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[1] = std::sin(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[2] = std::cos(angle_z)*radius_particle_to_robot;
        memset(new_particle.pos_r, 0, 3*sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, rot_m, 3, temp_pos, 1, 1.0, new_particle.pos_r, 1);
        new_particle.weight = 1.0/FOC_MAX_PARTICLES;
        new_particle.plume = new std::vector<FOC_Puff_t>;
        new_particle.plume->reserve(N_PUFFS);
        new_out.particles->push_back(new_particle);
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
