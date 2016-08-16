#include <stdlib.h>
#include <string>
#include <cmath>
#include <vector>
#include "cblas.h"
#include "wake_qr.h"
#include "foc/vector_rotation.h"

#define WAKE_QR_VRM     // vortex ring method
//#define WAKE_QR_VFM   // vortex filament method

static std::vector<Wake_QR_ring_t> wake_qr_rings;

static void induced_velocity_vortex_ring(float* center_ring, float radius_ring, float Gamma_ring, float core_radius_ring, float* att_ring, float*pos, float* vel);

// TODO: wind unused
void wake_qr_calculate_velocity(float* pos_qr, float* att_qr, float* pos, float* vel_wind, float* vel)
{
#if defined(WAKE_QR_VRM)
    /* vortex ring method */
    
    float c_ring[4][3] = {{0}, {0}, {0}, {0}}; // center of 4 motor vortex ring
    // calculate centers of motors
    float temp_c[4][3] = {{QR_MOTOR_DISTANCE/2.0/1.414, QR_MOTOR_DISTANCE/2.0/1.414, 0.0},
        {-QR_MOTOR_DISTANCE/2.0/1.414, QR_MOTOR_DISTANCE/2.0/1.414, 0.0}, 
        {-QR_MOTOR_DISTANCE/2.0/1.414, -QR_MOTOR_DISTANCE/2.0/1.414, 0.0}, 
        {QR_MOTOR_DISTANCE/2.0/1.414, -QR_MOTOR_DISTANCE/2.0/1.414, 0.0}};
    for (int i = 0; i < 4; i++) {
        rotate_vector(temp_c[i], c_ring[i], att_qr[2], att_qr[1], att_qr[0]);
        for (int j = 0; j < 3; j++)
            c_ring[i][j] += pos_qr[j];
    }

    // calculate centers of rings
    Wake_QR_ring_t new_wake_qr_ring;
    wake_qr_rings.clear();
    float temp_v[3] = {0.0, 0.0, -1.0};
    float unit[3] = {0};
    rotate_vector(temp_v, unit, att_qr[2], att_qr[1], att_qr[0]);
    
    for (int i = 0; i < QR_WAKE_RINGS; i++) { // num of rings
        for (int j = 0; j < 4; j++) { // 4 rotors
            for (int k = 0; k < 3; k++) // xyz
                new_wake_qr_ring.pos[j][k] = c_ring[j][k] + i*unit[k]*0.01; // 0.01 m gap
        }
        memcpy(new_wake_qr_ring.att, att_qr, 3*sizeof(float));
        wake_qr_rings.push_back(new_wake_qr_ring);
    }

    // calculate induced velocity
    memset(vel, 0, 3*sizeof(float)); // clear velocity
    for (int i = 0; i < wake_qr_rings.size(); i++)
        for (int j = 0; j < 4; j++)
            induced_velocity_vortex_ring(wake_qr_rings.at(i).pos[j], QR_PROP_RADIUS, 0.134125, 0.005,
                    wake_qr_rings.at(i).att, pos, vel);

#elif defined(WAKE_QR_VFM)
    /* vortex filament method */
#endif
}

std::vector<Wake_QR_ring_t>* wake_qr_get_info_vortex_rings(void)
{
    return &wake_qr_rings;
}

static float complete_elliptic_int_first(float k)
{
    return M_PI/2.0*(1.0 + 0.5*0.5*k*k + 0.5*0.5*0.75*0.75*std::pow(k, 4));
}
static float complete_elliptic_int_second(float k)
{
    return M_PI/2.0*(1.0 - 0.5*0.5*k*k - 0.5*0.5*0.75*0.75*std::pow(k, 4)/3.0);
}
/* vel += vel + result */
static void induced_velocity_vortex_ring(float* center_ring, float radius_ring, float Gamma_ring, float core_radius_ring, float* att_ring, float*pos, float* vel)
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
    rotate_vector(temp_v, unit_vector, att_ring[2], att_ring[1], att_ring[0]);

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
            op_r = std::sqrt(db_op_r);
        // a, A, m
        a = std::sqrt((op_r+radius_ring)*(op_r+radius_ring)+db_op_z+db_delta);
        b = (op_r-radius_ring)*(op_r-radius_ring)+db_op_z+db_delta;
        m = 4*op_r*radius_ring/(a*a);
        // u_z, cylindrical coord 
        u_z = Gamma_ring/(2*M_PI*a) * ((-(db_op_r-db_radius+db_op_z+db_delta)/b)
            *complete_elliptic_int_second(m) + complete_elliptic_int_first(m));
        // u_r, cylindrical coord
        u_r = Gamma_ring*op_z/(2*M_PI*op_r*a) * (((db_op_r+db_radius+db_op_z+db_delta)/b)
            *complete_elliptic_int_second(m) - complete_elliptic_int_first(m));
        // map u_z, u_r to cartesian coord
        norm_vel = std::sqrt(u_z*u_z + u_r*u_r);
        for (int i = 0; i < 3; i++)
            vel[i] += -unit_vector[i]*norm_vel;
    }
    /* Step 3: convert relative velocity to absolute velocity */
    // as the velocity of quad-rotor is ommited, this step is skipped
}
