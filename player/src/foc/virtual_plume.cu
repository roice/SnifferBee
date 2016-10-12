#include <stdlib.h>
#include <cmath>
#include <vector>
#include "flying_odor_compass.h"
#include "foc/wake_qr.h"
#include "foc/virtual_plume.h"
#include "foc/vector_rotation.h"
#include "foc/error_cuda.h"

#define     VIRTUAL_PLUME_DT    0.01   // second

/* update puff pos & r
 * Args:
 *      pos_qr      position of quad-rotor
 *      att_qr      attitude of quad-rotor
 *      puff        puff to be updated
 *      dt          delta time
 */
static void update_puff_info(float* pos_qr, float* att_qr, float* wind, FOC_Puff_t& puff, float dt)
{
    float vel[3];
    wake_qr_calculate_velocity(pos_qr, att_qr, puff.pos, wind, vel);
    // update puff position
    for (int i = 0; i < 3; i++) {
        puff.pos[i] += (vel[i]+wind[i])*dt;
    }
    // update puff radius
}

/* release virtual plume
 * Args:
 *      pos_r       position of virtual source relative to quad-rotor
 *      pos_qr      position of quad-rotor
 *      att_qr      attitude of quad-rotor
 *      plume       puff vector to be filled up
 */
void release_virtual_plume(float* pos_r, float* pos_qr, float* att_qr, float* wind, std::vector<FOC_Puff_t>* plume)
{
    // clear plume info
    plume->clear();
    // init puff info
    FOC_Puff_t puff;
    for (int i = 0; i < 3; i++)
        puff.pos[i] = pos_r[i] + pos_qr[i];
    for (int i = 0; i < N_PUFFS; i++) {
        // calculate puff pos and radius
        update_puff_info(pos_qr, att_qr, wind, puff, VIRTUAL_PLUME_DT);
        // save puff info
        plume->push_back(puff);
    }
}

/* calculate virtual mox readings
 * Args:
 *      plume           virtual plume
 *      reading         virtual mox readings
 *      pos             position of quad-rotor
 *      att             attitude of quad-rotor
 * TODO: multiple sensors, FOC_NUM_SENSORS > 3
 */
void calculate_virtual_tdoa_and_std(std::vector<FOC_Puff_t>* plume, float* pos, float* att, FOC_Particle_t& particle)
{
    if (!plume or plume->size() < 1)
        return;
 
    // calculate position of sensors
    float pos_s[3][3] = {{0}, {0}, {0}};
    float temp_s[3][3] = { {0, FOC_RADIUS, 0},
        {FOC_RADIUS*(-0.8660254), FOC_RADIUS*(-0.5), 0},
        {FOC_RADIUS*0.8660254, FOC_RADIUS*(-0.5), 0} };
    for (int i = 0; i < 3; i++) // relative position of sensors
        rotate_vector(temp_s[i], pos_s[i], att[2], att[1], att[0]);
    for (int i = 0; i < 3; i++) // absolute position of sensors
        for (int j = 0; j < 3; j++)
            pos_s[i][j] += pos[j];
    
    // calculate tdoa
    double temp_dis[3] = {
        std::sqrt(std::pow(plume->front().pos[0]-pos_s[0][0], 2)+std::pow(plume->front().pos[1]-pos_s[0][1], 2)+std::pow(plume->front().pos[2]-pos_s[0][2], 2)), 
        std::sqrt(std::pow(plume->front().pos[0]-pos_s[1][0], 2)+std::pow(plume->front().pos[1]-pos_s[1][1], 2)+std::pow(plume->front().pos[2]-pos_s[1][2], 2)),
        std::sqrt(std::pow(plume->front().pos[0]-pos_s[2][0], 2)+std::pow(plume->front().pos[1]-pos_s[2][1], 2)+std::pow(plume->front().pos[2]-pos_s[2][2], 2)) }; // FOC_NUM_SENSORS = 3
    double temp_distance;
    int temp_idx[3] = {0};
    for (int i = 0; i < N_PUFFS; i++) {
        for (int j = 0; j < 3; j++) { // FOC_NUM_SENSORS = 3
            temp_distance = std::sqrt(std::pow(plume->at(i).pos[0]-pos_s[j][0], 2)+std::pow(plume->at(i).pos[1]-pos_s[j][1], 2)+std::pow(plume->at(i).pos[2]-pos_s[j][2], 2));
            if (temp_distance < temp_dis[j]) {
                temp_dis[j] = temp_distance;
                temp_idx[j] = i;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        particle.tdoa.toa[i] = temp_idx[0] - temp_idx[i];
    }
    
    // calculate standard deviation
    for (int i = 0; i < 3; i++)
        particle.std.std[i] = std::exp(temp_dis[i]);
}

#if 0
/* Calculate likelihood of particles according to virtual delta
 * Args:
 *      delta       measurement delta
 *      particles   particles containing virtual delta
 * Out:
 *      false       measurement delta doesn't contain enough info
 */
bool calculate_likelihood_of_virtual_delta(FOC_Delta_t& delta, std::vector<FOC_Particle_t>* particles)
{
    float u[2];
    if (!estimate_horizontal_direction_according_to_tdoa(delta, u))
        return false;

    if (particles->size() < 1)
        return false;

    float u_v[2];
    float angle, cos_angle;
    for (int i = 0; i < particles->size(); i++) {
        if (!estimate_horizontal_direction_according_to_tdoa(particles->at(i).delta, u_v)) {
            particles->at(i).weight = 0;
            continue;
        }

        cos_angle = (u[0]*u_v[0]+u[1]*u_v[1])/std::sqrt((u[0]*u[0]+u[1]*u[1])*(u_v[0]*u_v[0]+u_v[1]*u_v[1]));
        if (cos_angle >= 1.0)
            angle = 0;
        else if (cos_angle <= -1.0)
            angle = M_PI;
        else
            angle = std::acos(cos_angle);
        particles->at(i).weight = 1 - std::abs(angle)/M_PI;
        if (particles->at(i).weight < 0)
            particles->at(i).weight = 0;
    }

    return true;
}
#endif

/* Estimate horizontal direction according to TOA
 * Args:
 *      delta       std & toa
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
bool estimate_horizontal_direction_according_to_tdoa(FOC_TDOA_t& delta, float* out)
{
    float e_x, e_y, dt_lf = delta.toa[1], dt_rf = delta.toa[2], speed;
    float sqrt_3 = sqrt(3);

    // check if dt is valid
    if (dt_lf == 0 and dt_rf == 0)
        return false;

    // calculate e_x & e_y
    if (dt_lf == dt_rf) {
        e_x = 0;
        e_y = 1;
    }
    else {
        float dt_add = dt_lf + dt_rf;
        float dt_minus = dt_lf - dt_rf;
        e_x = 1.0 / sqrt(1 + 1.0/3.0*pow(dt_add/dt_minus, 2));
        e_y = 1.0/sqrt_3*dt_add/dt_minus*e_x;
    }

    // determine sign(e_x) & sign(e_y)
    //if (absf(dt_lf) > absf(dt_rf)) { // math.h
    if (std::abs(dt_lf) > std::abs(dt_rf)) { // cmath
        if (std::signbit(sqrt_3*e_x+3*e_y)!=std::signbit(dt_lf)) {
            e_x *= -1;
            e_y *= -1;
        }
    }
    else {
        if (std::signbit(-sqrt_3*e_x+3*e_y)!=std::signbit(dt_rf)) {
            e_x *= -1;
            e_y *= -1;
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

#if 0
void calculate_virtual_delta(std::vector<FOC_Reading_t>* mox_reading, FOC_Delta_t& delta)
{
    if (mox_reading->size() < 2)
        return;

    int N = mox_reading->size();

    memset(&delta, 0, sizeof(delta));

    // standard deviation
    double sum[FOC_NUM_SENSORS] = {0};
    float mean[FOC_NUM_SENSORS] = {0};
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
    {
        for (int i = 0; i < N; i++)
            sum[idx] += mox_reading->at(i).reading[idx];
        mean[idx] = sum[idx]/N;
        sum[idx] = 0;
        for (int i = 0; i < N; i++)
            sum[idx] += std::pow((mox_reading->at(i).reading[idx] - mean[idx]), 2);
        delta.std[idx] = std::sqrt(sum[idx]/N);
    }

   // time of arrival
   //   interpolation
   std::vector<FOC_Reading_t> reading;
   reading.reserve(N*FOC_MOX_INTERP_FACTOR);
   FOC_Reading_t new_reading;
   float y1[FOC_NUM_SENSORS];
   float y2[FOC_NUM_SENSORS];
   for (int i = 0; i < N-1; i++) {
       for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) { // normalize
           y1[idx] = (mox_reading->at(i).reading[idx] - mean[idx]) / delta.std[idx];
           y2[idx] = (mox_reading->at(i+1).reading[idx] - mean[idx]) / delta.std[idx];
       }
       for (int j = 0; j < FOC_MOX_INTERP_FACTOR; j++) {
           for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
               new_reading.reading[idx] = (y2[idx]-y1[idx])*j/FOC_MOX_INTERP_FACTOR + y1[idx];
           reading.push_back(new_reading);
       }
   }
   //   correlation
   N = reading.size();
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
                   temp += reading.at(i).reading[idx]*reading.at(i+t).reading[0];
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
       delta.toa[idx] = time[index];
    }
}
#endif

/* GPU version */
__device__ void RotateVector(float* vector, float* out, float yaw, float pitch, float roll)
{
    // calculate rotation matrix
    float sin_yaw = __sinf(yaw);
    float cos_yaw = __cosf(yaw);
    float sin_pitch = __sinf(pitch);
    float cos_pitch = __cosf(pitch);
    float sin_roll = __sinf(roll);
    float cos_roll = __cosf(roll);
    float R_zyx[9];
    R_zyx[0] = cos_yaw;
    R_zyx[1] = -sin_yaw;
    R_zyx[2] = 0.0;
    R_zyx[3] = sin_yaw;
    R_zyx[4] = cos_yaw;
    R_zyx[5] = 0.0;
    R_zyx[6] = 0.0;
    R_zyx[7] = 0.0;
    R_zyx[8] = 1.0;
    // rotate
    out[0] = R_zyx[0]*vector[0]+R_zyx[1]*vector[1]+R_zyx[2]*vector[2];
    out[1] = R_zyx[3]*vector[0]+R_zyx[4]*vector[1]+R_zyx[5]*vector[2];
    out[2] = R_zyx[6]*vector[0]+R_zyx[7]*vector[1]+R_zyx[8]*vector[2];
}

__device__ void InducedVelocityVortexRing(float* center_ring, float radius_ring, float Gamma_ring, float core_radius_ring, float* att_ring, float*pos, float* vel)
{
    float op[3], op_z, op_r, db_op_z, db_op_r;
    float u_z, u_r, m, a, b;

    /* Step 1: convert absolute position to relative position */
    // relative position, center_ring as origin
    for (int i = 0; i < 3; i++)
        op[i] = pos[i] - center_ring[i];

    /* Step 2: calculate relative velocity */
    float norm_vel; // norm of velocity
    float unit_vector[3] = {0}; // unit vector of the ring axis
    float db_radius = radius_ring*radius_ring;          // radius^2
    float db_delta = core_radius_ring*core_radius_ring; // core^2
   
    // calculate unit vector
    float temp_v[3] = {0.0, 0.0, 1.0};
    RotateVector(temp_v, unit_vector, att_ring[2], att_ring[1], att_ring[0]);

    if (op[0]==0.0f && op[1]==0.0f && op[2]==0.0f) { // P is at center
        norm_vel = Gamma_ring/(2*radius_ring); // norm of velocity
        for (int i = 0; i < 3; i++)
            vel[i] += -unit_vector[i]*norm_vel;
    }
    else {
        // op_z, cylindrical coord
        op_z = cblas_sdot(3, op, 1, unit_vector, 1);
        db_op_z = op_z*op_z;
        // op_r, cylindrical coord
        db_op_r = op[0]*op[0]+op[1]*op[1]+op[2]*op[2]-db_op_z;
        if (db_op_r < 0) {
            db_op_r = 0;
            op_r = 0;
        }
        else
            op_r = __frsqrt_rn(db_op_r);
        // a, A, m
        a = __frsqrt_rn((op_r+radius_ring)*(op_r+radius_ring)+db_op_z+db_delta);
        b = (op_r-radius_ring)*(op_r-radius_ring)+db_op_z+db_delta;
        m = 4*op_r*radius_ring/(a*a);
        // u_z, cylindrical coord 
        u_z = Gamma_ring/(2*M_PI*a) * ((-(db_op_r-db_radius+db_op_z+db_delta)/b)
            *complete_elliptic_int_second(m) + complete_elliptic_int_first(m));
        // u_r, cylindrical coord
        u_r = Gamma_ring*op_z/(2*M_PI*op_r*a) * (((db_op_r+db_radius+db_op_z+db_delta)/b)
            *complete_elliptic_int_second(m) - complete_elliptic_int_first(m));
        // map u_z, u_r to cartesian coord
        norm_vel = __frsqrt_rn(u_z*u_z + u_r*u_r);
        for (int i = 0; i < 3; i++)
            vel[i] += -unit_vector[i]*norm_vel;
    }
    /* Step 3: convert relative velocity to absolute velocity */
    // as the velocity of quad-rotor is ommited, this step is skipped
}

__device__ void WakeQRCalculateVelocity(float* pos_qr, float* att_qr, float* pos, float* wind, float* vel)
{
    /* vortex ring method */
    float c_ring[4][3] = {{0}, {0}, {0}, {0}}; // center of 4 motor vortex ring
    // calculate centers of motors
    float temp_c[4][3] = {{QR_MOTOR_DISTANCE/2.0/1.414, QR_MOTOR_DISTANCE/2.0/1.414, 0.0},
        {-QR_MOTOR_DISTANCE/2.0/1.414, QR_MOTOR_DISTANCE/2.0/1.414, 0.0}, 
        {-QR_MOTOR_DISTANCE/2.0/1.414, -QR_MOTOR_DISTANCE/2.0/1.414, 0.0}, 
        {QR_MOTOR_DISTANCE/2.0/1.414, -QR_MOTOR_DISTANCE/2.0/1.414, 0.0}};
    for (int i = 0; i < 4; i++) {
        RotateVector(temp_c[i], c_ring[i], att_qr[2], att_qr[1], att_qr[0]);
        for (int j = 0; j < 3; j++)
            c_ring[i][j] += pos_qr[j];
    }

    Wake_QR_ring_t wake_qr_rings[4];
    float temp_v[3] = {0.0, 0.0, -1.0};
    float unit[3] = {0};
    rotate_vector(temp_v, unit, att_qr[2], att_qr[1], att_qr[0]);
    for (int i = 0; i < QR_WAKE_RINGS; i++) { // num of rings
        for (int j = 0; j < 4; j++) { // 4 rotors
            for (int k = 0; k < 3; k++) // xyz
                wake_qr_rings[i].pos[j][k] = c_ring[j][k] + i*unit[k]*0.01; // 0.01 m gap
        }
        HANDLE_ERROR( cudaMemcpy(wake_qr_rings[i].att, att_qr, 
                3*sizeof(float), cudaMemcpyDeviceToDevice) );
    }

    // calculate induced velocity
    for (int i = 0; i < 4; i++) // quadrotor
        for (int j = 0; j < 4; j++)
            InducedVelocityVortexRing(wake_qr_rings[i].pos[j], QR_PROP_RADIUS, 0.134125, 0.005,
                    wake_qr_rings[i].att, pos, vel);
}

__global__ void CalculateVirtualPlumes(FOC_Puff_t* puffs, FOC_Input_t* raw, float* wind, int num_plumes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float vel[3], pos[3];
    if (tid < num_plumes) {
        HANDLE_ERROR( cudaMemcpy(pos, raw->pos, 
                3*sizeof(float), cudaMemcpyDeviceToDevice) );
        for (int i = 0; i < N_PUFFS; i++) {
            // calculate puff pos and radius
            WakeQRCalculateVelocity(raw->position, raw->attitude, pos, wind, vel);
            // update puff position
            for (int i = 0; i < 3; i++)
                pos[i] += (vel[i]+wind[i])*VIRTUAL_PLUME_DT;
            HANDLE_ERROR( cudaMemcpy(puffs[tid*N_PUFFS+i].pos, pos,
                3*sizeof(float), cudaMemcpyDeviceToDevice) );
        }
    }
}

__global__ void CalculateVirtualTDOAandSTD(FOC_Puff_t* puffs, FOC_Input_t& raw, FOC_TDOA_t* tdoa, FOC_STD_t* std, int num_plumes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_plumes)   return;

    float* pos = raw.position;
    float* att = raw.attitude;

    // calculate position of sensors
    float pos_s[3][3] = {{0}, {0}, {0}};
    float temp_s[3][3] = { {0, FOC_RADIUS, 0},
        {FOC_RADIUS*(-0.8660254), FOC_RADIUS*(-0.5), 0},
        {FOC_RADIUS*0.8660254, FOC_RADIUS*(-0.5), 0} };
    for (int i = 0; i < 3; i++) // relative position of sensors
        RotateVector(temp_s[i], pos_s[i], att[2], att[1], att[0]);
    for (int i = 0; i < 3; i++) // absolute position of sensors
        for (int j = 0; j < 3; j++)
            pos_s[i][j] += pos[j];
    
    // calculate tdoa
    double temp_dis[3] = {
        __frsqrt_rn(__powf(puffs[tid*N_PUFFS].pos[0]-pos_s[0][0], 2)+__powf(puffs[tid*N_PUFFS].pos[1]-pos_s[0][1], 2)+__powf(puffs[tid*N_PUFFS].pos[2]-pos_s[0][2], 2)), 
        __frsqrt_rn(__powf(puffs[tid*N_PUFFS].pos[0]-pos_s[1][0], 2)+__powf(puffs[tid*N_PUFFS].pos[1]-pos_s[1][1], 2)+__powf(puffs[tid*N_PUFFS].pos[2]-pos_s[1][2], 2)),
        __frsqrt_rn(__powf(puffs[tid*N_PUFFS].pos[0]-pos_s[2][0], 2)+__powf(puffs[tid*N_PUFFS].pos[1]-pos_s[2][1], 2)+__powf(puffs[tid*N_PUFFS].pos[2]-pos_s[2][2], 2)) }; // FOC_NUM_SENSORS = 3
    double temp_distance;
    int temp_idx[3] = {0};
    for (int i = 0; i < N_PUFFS; i++) {
        for (int j = 0; j < 3; j++) { // FOC_NUM_SENSORS = 3
            temp_distance = __frsqrt_rn(__powf(puffs[tid*N_PUFFS+i].pos[0]-pos_s[j][0], 2)+__powf(puffs[tid*N_PUFFS+i].pos[1]-pos_s[j][1], 2)+__powf(puffs[tid*N_PUFFS+i].pos[2]-pos_s[j][2], 2));
            if (temp_distance < temp_dis[j]) {
                temp_dis[j] = temp_distance;
                temp_idx[j] = i;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        tdoa[tid].toa[i] = temp_idx[0] - temp_idx[i];
    }
    
    // calculate standard deviation
    for (int i = 0; i < 3; i++)
        std[tid].std[i] = std::exp(temp_dis[i]);
}

void release_virtual_plumes_and_calculate_virtual_tdoa_std(std::vector<FOC_Particle_t>* particles, FOC_Input_t& raw, float* est_wind)
{
    /* get the properties of all the graphic cards this machine has */ 
    int count; // number of devices
    HANDLE_ERROR( cudaGetDeviceCount(&count) );
    
    if (count <= 0) {// no graphic card found
        return;
    }

    // puffs
    FOC_Puff_t*     puffs;
    FOC_Puff_t*     dev_puffs;
    FOC_Input_t*    dev_raw;    // contains pos, att...
    float*          dev_wind;
    FOC_TDOA_t*     tdoa;
    FOC_STD_t*      std;
    FOC_TDOA_t*     dev_tdoa;
    FOC_STD_t*      dev_std;
    
    int num_plumes = particles->size();

    // Phase 1: allocate memory
    // allocate a page-locked host memory containing all of the marker states
    HANDLE_ERROR( cudaHostAlloc((void**)&puffs, 
        num_plumes*N_PUFFS*sizeof(*puffs), cudaHostAllocDefault) );
    // allocate device memory as big as the host's
    HANDLE_ERROR( cudaMalloc((void**)&dev_puffs, 
        num_plumes*N_PUFFS*sizeof(*dev_puffs)) );
    // allocate device memory for dev_raw
    HANDLE_ERROR( cudaMalloc((void**)&dev_raw, sizeof(*dev_raw)) );
    // allocate device memory for dev_wind
    HANDLE_ERROR( cudaMalloc((void**)&dev_wind, 3*sizeof(*dev_wind)) );
    // allocate device memory for dev_tdoa
    HANDLE_ERROR( cudaMalloc((void**)&dev_tdoa, 
        num_plumes*sizeof(*dev_tdoa)) );
    // allocate device memory for dev_std
    HANDLE_ERROR( cudaMalloc((void**)&dev_std, 
        num_plumes*sizeof(*dev_std)) );

    // Phase 2: copy info to GPU's memory
    HANDLE_ERROR( cudaMemcpy(dev_raw, &raw, 
                sizeof(FOC_Input_t), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_wind, est_wind, 
                3*sizeof(float), cudaMemcpyHostToDevice) );

    // Phase 3: release virtual plumes & calculate tdoa/std
    CalculateVirtualPlumes(dev_puffs, dev_raw, dev_wind, num_plumes);
    CalculateVirtualTDOAandSTD(dev_puffs, dev_raw, dev_tdoa, dev_std, num_plumes)

    // Phase 4: copy cuda memory to CPU memory
    HANDLE_ERROR( cudaMemcpy(puffs, dev_puffs, 
                num_plumes*N_PUFFS*sizeof(*puffs), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(tdoa, dev_tdoa, 
                num_plumes*sizeof(*tdoa), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(std, dev_std, 
                num_plumes*sizeof(*std), cudaMemcpyDeviceToHost) );

    // Phase 5: save puffs and tdoa/std info
    for (int i = 0; i < num_plumes; i++) {
        particles->at(i).plume->clear();
        for (int j = 0; j < N_PUFFS; j++)
            particles->at(i).plume->push_back(puffs[i*N_PUFFS+j]);
        memcpy(&particles->at(i).tdoa, &(tdoa[i]), sizeof(*tdoa));
        memcpy(&particles->at(i).std, &(std[i]), sizeof(*std));
    }

    // Phase 6: free memory
    // free device memory
    HANDLE_ERROR( cudaFree(dev_puffs) );
    HANDLE_ERROR( cudaFree(dev_raw) );
    HANDLE_ERROR( cudaFree(dev_wind) );
    HANDLE_ERROR( cudaFree(dev_tdoa) );
    HANDLE_ERROR( cudaFree(dev_std) );
    // free host memory
    HANDLE_ERROR( cudaFreeHost(puffs) );
    HANDLE_ERROR( cudaFreeHost(tdoa) );
    HANDLE_ERROR( cudaFreeHost(std) );
}
