#include <stdlib.h>
#include <cmath>
#include <vector>
#include "flying_odor_compass.h"
#include "foc/wake_qr.h"
#include "foc/virtual_plume.h"
#include "foc/vector_rotation.h"

/* CPU version */
#ifndef GPU_COMPUTING
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
        std::sqrt((plume->front().pos[0]-pos_s[0][0])*(plume->front().pos[0]-pos_s[0][0]) + (plume->front().pos[1]-pos_s[0][1])*(plume->front().pos[1]-pos_s[0][1]) + (plume->front().pos[2]-pos_s[0][2])*(plume->front().pos[2]-pos_s[0][2])), 
        std::sqrt((plume->front().pos[0]-pos_s[1][0])*(plume->front().pos[0]-pos_s[1][0]) + (plume->front().pos[1]-pos_s[1][1])*(plume->front().pos[1]-pos_s[1][1]) + (plume->front().pos[2]-pos_s[1][2])*(plume->front().pos[2]-pos_s[1][2])),
        std::sqrt((plume->front().pos[0]-pos_s[2][0])*(plume->front().pos[0]-pos_s[2][0]) + (plume->front().pos[1]-pos_s[2][1])*(plume->front().pos[1]-pos_s[2][1]) + (plume->front().pos[2]-pos_s[2][2])*(plume->front().pos[2]-pos_s[2][2])) }; // FOC_NUM_SENSORS = 3
    double temp_distance;
    int temp_idx[3] = {0};
    for (int i = 0; i < N_PUFFS; i++) {
        for (int j = 0; j < 3; j++) { // FOC_NUM_SENSORS = 3
            temp_distance = std::sqrt((plume->at(i).pos[0]-pos_s[j][0])*(plume->at(i).pos[0]-pos_s[j][0]) + (plume->at(i).pos[1]-pos_s[j][1])*(plume->at(i).pos[1]-pos_s[j][1]) + (plume->at(i).pos[2]-pos_s[j][2])*(plume->at(i).pos[2]-pos_s[j][2]));
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
#endif  // ifndef GPU_COMPUTING
