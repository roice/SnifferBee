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
#define POSSIBLE_ANG_RANGE  180.0*M_PI/180.0  // possible angle range to resample/init particles
#define N_DELAY     (FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ)
#define N_SAMPLES   (FOC_TIME_RECENT_INFO*FOC_MOX_DAQ_FREQ) // number of samples to trace back

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

    // setup normal distributed random number generator
    r4_nor_setup ( rand_kn, rand_fn, rand_wn );

    out.clear();
}

/* Estimate the 3D direction the odor comes from 
 * Args:
 *      in      standard deviation & time of arrival of signals of different sensors
 *      out     direction estimation
 */
bool foc_estimate_source_direction_update(std::vector<FOC_Input_t>& raw, std::vector<FOC_Delta_t>& delta, std::vector<FOC_Wind_t>& wind, std::vector<FOC_Estimation_t>& out)
{
    // signal delay FOC_SIGNAL_DELAY s after interpolation, and delta delay FOC_TIME_RECENT_INFO s after delta computation
    if (wind.size() < N_SAMPLES + N_DELAY or raw.size() < N_SAMPLES + N_DELAY)
        return false;

    FOC_Estimation_t new_out;
    new_out.particles = new std::vector<FOC_Particle_t>;
    new_out.particles->reserve(FOC_MAX_PARTICLES);
 
/* Phase 0: get average wind (actually disturbance) vector */
    double temp[3] = {0};
    float temp_wind[3] = {0};
    float wind_est[3] = {0};
    // TODO: 3D
    for (int i = raw.size()-N_DELAY-N_SAMPLES; i < raw.size()-N_DELAY; i++) {
        memset(temp_wind, 0, 3*sizeof(float));
        rotate_vector(raw.at(i).wind, temp_wind, raw.at(i).attitude[2], raw.at(i).attitude[1], raw.at(i).attitude[0]);
        for (int j = 0; j < 2; j++)
            temp[j] += temp_wind[j];
    }
    for (int i = 0; i < 2; i++)
        wind_est[i] = temp[i]/N_SAMPLES;
    if (wind_est[0] == 0. and wind_est[1] == 0. and wind_est[2] == 0.) {
        new_out.valid = false;
        out.push_back(new_out);
        return false;
    }
    // convert wind disturbance info to wind info
    for (int i = 0; i < 2; i++) {
        wind_est[i] = wind_est[i]/100.0*0.5; // 100 -- 0.5 m/s
    }

/* Phase 1: compute the center of the recent trajectory of the robot */
    float robot_traj_center[3];
    for (int i = 0; i < 3; i++) {
        for (int j = raw.size()-N_DELAY-N_SAMPLES; j < raw.size()-N_DELAY; j++)
            temp[i] += raw.at(j).position[i];
        robot_traj_center[i] = temp[i]/N_SAMPLES;
    }

/* Phase 2: compute the max deviation of the robot trajectory, and compute the radius of the particles to the robot */
    float robot_traj_deviation = 0; float temp_distance;
    for (int j = raw.size()-N_DELAY-N_SAMPLES; j < raw.size()-N_DELAY; j++) {
        temp_distance = raw.at(j).position[0]*raw.at(j).position[0] + raw.at(j).position[1]*raw.at(j).position[1] + raw.at(j).position[2]*raw.at(j).position[2];
        if (temp_distance > robot_traj_deviation)
            robot_traj_deviation = temp_distance;
    }
    robot_traj_deviation = sqrt(temp_distance);
    float radius_particle_to_robot = robot_traj_deviation; //+ 3*FOC_RADIUS; // 3 is empirical value

/* Phase 3: init particles / resample particles */
    // compute rotation matrix to rotate unit_z to reverse direction of wind
    float unit_z[3] = {0., 0., 1.};
    float wind_est_reverse[3] = {-wind_est[0], -wind_est[1], -wind_est[2]};
    float rot_m_init[9]; // rotation matrix to generate new particles
    CalculateRotationMatrix(unit_z, wind_est_reverse, rot_m_init);
    // Resampling / init particles
    float temp_ang, possibility_to_survive;
    int num_new_particles_to_split;
    float rot_m_split[9]; // rotation matrix to split important particles
    double temp_weight = 0.0;
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
                    if (r4_uni(rand_seed) < possibility_to_survive)// goodluck
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
            new_out.particles->at(i).weight = new_out.particles->at(i).weight/temp_weight;
    }
    else // first run
        // generate FOC_MAX_PARTICLES number of particles
        init_particles(rand_seed, FOC_MAX_PARTICLES, radius_particle_to_robot, rot_m_init, new_out);

#if 0
/* Phase 4: Release virtual plumes and get virtual mox readings */
    // traverse every particle
    for (int i = 0; i < new_out.particles->size(); i++) {
        new_out.particles->at(i).reading->clear();
        // traverse recent samples
        for (int j = raw.size()-N_DELAY-N_SAMPLES; j < raw.size()-N_DELAY; j++) {
            // release virtual plume
            release_virtual_plume(new_out.particles->at(i).pos_r, raw.at(j).position, raw.at(j).attitude, wind_est, new_out.particles->at(i).plume);
            // calculate virtual mox readings
            calculate_virtual_mox_reading(new_out.particles->at(i).plume, new_out.particles->at(i).reading, raw.at(j).position, raw.at(j).attitude);
        }
    }
/* Phase 5: calculate delta for virtual mox readings, and evaluate them to calculate weight */
    for (int i = 0; i < new_out.particles->size(); i++) { // traverse every particle
        // calculate virtual delta
        calculate_virtual_delta(new_out.particles->at(i).reading, new_out.particles->at(i).delta);
    }
    // evaluate virtual delta, to get likelihood
    if (!calculate_likelihood_of_virtual_delta(delta.back(), new_out.particles)) {
        new_out.valid = false;
        out.push_back(new_out);
        return false;
    }

/* Phase 6: Update weights of particles */
    double sum_w = 0;
    for (int i = 0; i < new_out.particles->size(); i++)
        sum_w += new_out.particles->at(i).weight;
    if (sum_w == 0) { // no particles
        new_out.valid = false;
        out.push_back(new_out);
        return false;
    }
    for (int i = 0; i < new_out.particles->size(); i++)
        new_out.particles->at(i).weight /= sum_w;

/* Phase 7: Calculate direction of gas source */
    double temp_direction[3] = {0};
    double norm_direction;
    for (int i = 0; i < new_out.particles->size(); i++)
        for (int j = 0; j < 3; j++)
            temp_direction[j] += new_out.particles->at(i).pos_r[j]*new_out.particles->at(i).weight;
    norm_direction = std::sqrt(temp_direction[0]*temp_direction[0]+temp_direction[1]*temp_direction[1]+temp_direction[2]*temp_direction[2]);
    for (int i = 0; i < 3; i++)
        new_out.direction[i] = temp_direction[i] / norm_direction;
#else

    memset(new_out.direction, 0, 3*sizeof(float));
    float temp_direct[3] = {0};
    estimate_horizontal_direction_according_to_tdoa(delta.back(), temp_direct);
    rotate_vector(temp_direct, new_out.direction, raw.back().attitude[2], 0, 0);
#endif

    // save results
    new_out.valid = true;
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
        float angle_z = r4_uni(seed)*POSSIBLE_ANG_RANGE; // particles are uniformly distributed
        //float angle_z = M_PI/2.0;
        float angle_xy = r4_uni(seed)*2*M_PI;
        temp_pos[0] = std::cos(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[1] = std::sin(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[2] = std::cos(angle_z)*radius_particle_to_robot;
        memset(new_particle.pos_r, 0, 3*sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, rot_m, 3, temp_pos, 1, 1.0, new_particle.pos_r, 1);
        new_particle.weight = 1.0/FOC_MAX_PARTICLES;
        new_particle.plume = new std::vector<FOC_Puff_t>;
        new_particle.plume->reserve(N_PUFFS);
        new_particle.reading = new std::vector<FOC_Reading_t>;
        new_particle.reading->reserve(N_SAMPLES);
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
        //float angle_z = M_PI/2.0;
        float angle_xy = r4_uni(seed)*2*M_PI;
        temp_pos[0] = std::cos(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[1] = std::sin(angle_xy)*std::sin(angle_z)*radius_particle_to_robot;
        temp_pos[2] = std::cos(angle_z)*radius_particle_to_robot;
        memset(new_particle.pos_r, 0, 3*sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, rot_m, 3, temp_pos, 1, 1.0, new_particle.pos_r, 1);
        new_particle.weight = weight;
        new_particle.plume = new std::vector<FOC_Puff_t>;
        new_particle.plume->reserve(N_PUFFS);
        new_particle.reading = new std::vector<FOC_Reading_t>;
        new_particle.reading->reserve(N_SAMPLES);
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
