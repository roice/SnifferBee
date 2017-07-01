/*
 * Odor finding
 *
 * Author:
 *      Roice   Luo
 * Date:
 *      2017.06.22
 */

#include <cmath>
#include "method/plume_finding.h"

Plume_Finding::Plume_Finding(float pos_x, float pos_y, float pos_z, float alpha = 0.25, float scaler=0.1, float vel=0.1, float dt=0.04)
{
    initial_position[0] = pos_x;
    initial_position[1] = pos_y;
    initial_position[2] = pos_z;
    type_alpha = alpha;
    cast_scaler = scaler;
    cast_velocity = vel;
    time_marching_interval = dt;
    time_marching_idx = 0;
}

void Plume_Finding::update(void)
{
    float vel = cast_velocity;
    float dt = time_marching_interval;
    float idx = time_marching_idx;
    float alpha = type_alpha;

    if (idx > 0) {
        // calculate angle
        float R = std::sqrt(std::pow(current_position[0]-initial_position[0], 2)
            + std::pow(current_position[1]-initial_position[1], 2)
            + std::pow(current_position[2]-initial_position[2], 2));
        angle += std::asin(vel*dt/R);
    }
    else {
        angle += 5.0/180.0*M_PI;
    }

    // calculate position
    current_position[0] = cast_scaler*angle*std::cos(angle)*std::cos(alpha*angle) + initial_position[0];
    current_position[1] = cast_scaler*angle*std::cos(angle)*std::sin(alpha*angle) + initial_position[1];
    current_position[2] = -cast_scaler*angle*std::sin(angle) + initial_position[2];

    time_marching_idx ++;
}

void Plume_Finding::reinit(float pos_x, float pos_y, float pos_z)
{
    initial_position[0] = pos_x;
    initial_position[1] = pos_y;
    initial_position[2] = pos_z;
    angle = 0;
    time_marching_idx = 0;
}
