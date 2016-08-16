#include <stdlib.h>
#include <cmath>
#include <vector>
#include "flying_odor_compass.h"
#include "foc/wake_qr.h"
#include "foc/virtual_plume.h"
#include "foc/vector_rotation.h"

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
        update_puff_info(pos_qr, att_qr, wind, puff, 0.1);
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
 */
void calculate_virtual_mox_reading(std::vector<FOC_Puff_t>* plume, std::vector<FOC_Reading_t>* reading, float* pos, float* att)
{
    if (!plume or plume->size() < 1)
        return;

    // TODO: multiple sensors, FOC_NUM_SENSORS > 3
    // calculate position of sensors
    float pos_s[3][3] = {{0}, {0}, {0}};
    float temp_s[3][3] = { {0, FOC_RADIUS, 0},
        {FOC_RADIUS*(-0.8660254), FOC_RADIUS*(-0.5), 0},
        {FOC_RADIUS*0.8660254, FOC_RADIUS*(-0.5), 0} };
    for (int i = 0; i < 3; i++)
        rotate_vector(temp_s[i], pos_s[i], att[2], att[1], att[0]);
    
    // calculate mox reading
    //float inv_sigma[3] = {1.42857, 9.34579, 9.34579};
    float inv_sigma[3] = {5.0, 5.0, 5.0};
    float delta[3];
    float power;
    float conc;
    FOC_Reading_t new_reading;
    memset(&new_reading, 0, sizeof(FOC_Reading_t));
    
    // TODO: multiple sensors, FOC_NUM_SENSORS > 3
    for (int idx = 0; idx < 3; idx++) {
        conc = 0.0;
        for (int i = 0; i < plume->size(); i++) {
            power = 0.0;
            // get delta
            for (int k = 0; k < 3; k++)
                delta[k] = pos_s[idx][k] - plume->at(i).pos[k];
            for (int k = 0; k < 3; k++)
                power += delta[k]*inv_sigma[k]*delta[k];
            // calculate delta^T*sigma*delta
            conc += 1.0/1.4076*std::exp(-power);
        }
        new_reading.reading[idx] = conc; 
    }
    reading->push_back(new_reading);
}

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
bool estimate_horizontal_direction_according_to_tdoa(FOC_Delta_t& delta, float* out)
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
