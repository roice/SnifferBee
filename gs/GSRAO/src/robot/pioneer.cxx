/*
 * Pioneer Robot
 *         
 *
 * Author: Roice (LUO Bing)
 * Date: 2017-02-26 create this file
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <cmath>
/* GSRAO */
#include "mocap/packet_client.h"
#include "robot/microbee.h"
#include "robot/robot.h"
#include "io/net_send_bear_cmd.h"
#include "common/vector_rotation.h"
#include "GSRAO_Config.h"
#include "GSRAO_thread_comm.h"
/* CBLAS */
#include "cblas.h"
/* Liquid */
#include "liquid.h"

#define PIONEER_POS_DEST_THRESHOLD  0.35 // m

#define PIO_IDX 0 // only one Pioneer

bool pioneer_control_init(void)
{
    if (net_send_bear_cmd_init(""))
        return true;
    else
        return false;
}

void pioneer_control_close(void)
{
    net_send_bear_cmd_terminate();
}

#if 0
static float pioneer_calculate_rot_angle(float current_heading, float dest_heading)
{
    // convert (-pi~pi) to (0~2pi)
    float a = current_heading;
    float b = dest_heading;
    a = a >= 0 ? a : (2*M_PI + a);
    b = b >= 0 ? b : (2*M_PI + b);
    // calculate rotation angle from a to b
    float angle = b - a;
    if (angle > M_PI)
        angle = angle - 2*M_PI;
    else if (angle < -M_PI)
        angle = 2*M_PI + angle;

    return -angle*180.0/M_PI;
}

static float constrain(float amt, float low, float high)
{
    if (amt < low)
        return low;
    else if (amt > high)
        return high;
    else
        return amt;
}

static float applyDeadband(float value, float deadband)
{
    if (fabs(value) < deadband) {
        value = 0;
    } else if (value > 0) {
        value -= deadband;
    } else if (value < 0) {
        value += deadband;
    }
    return value;
}

float pioneer_calculate_v(float *pos_ref, float *pos)
{
    float dist = std::sqrt((pos_ref[0]-pos[0])*(pos_ref[0]-pos[0]) + (pos_ref[1]-pos[1])*(pos_ref[1]-pos[1]));
    float vel = constrain(4*dist, 0, 0.5); // limit velocity to 0-0.5 m/s

    return vel*1000.;
}

float pioneer_calculate_w_pid(float rot_angle, float current_rot_vel)
{
    static float errorVelocityI = 0;
    static float previous_rot_vel = 0;

    float dt = 0.1; // 0.1 s, 10 Hz
    float kp_ang = 1.0;
    float kp_vel = 0.5;
    float ki_vel = 0.001;
    float kd_vel = 0.1;

    float rot_acc = (current_rot_vel - previous_rot_vel)/0.1;
    previous_rot_vel = current_rot_vel;

    float err = applyDeadband(rot_angle, 5); // degree
    float setVel = constrain(kp_ang*err, -180, 180); // limit velocity to

    // velocity PID-controller
    err = setVel - current_rot_vel;
    float result = constrain(kp_vel*err, -100, 100);
    errorVelocityI += ki_vel*err;
    errorVelocityI = constrain(errorVelocityI, -100, 100);
    result += errorVelocityI;
    result -= constrain(kd_vel*rot_acc, -100, 100);

//printf("rot_angle = %f, current_rot_vel = %f, result = %f\n", rot_angle, current_rot_vel, result);

    return -result;
}

float pioneer_calculate_heading_velocity(float heading, float previous_heading, float dt)
{
    if (dt <= 0.)
        return 0.;

    float angle = heading - previous_heading;
    if (angle > M_PI)
        angle = -2*M_PI + angle;
    else if (angle < -M_PI)
        angle = 2*M_PI + angle;

    return (angle/dt)*180./M_PI;
}

static void* pioneer_control_loop(void* exit)
{
    struct timespec req, rem;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 100000000L; // 100 ms

    /* Get position, velocity, acceleration, attitude ... */
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state

    /* reference position and actual position */
    float pos_ref[3], pos[3];
    /* reference heading and actual heading */
    float heading_ref, heading;
    float previous_heading = data->robot[PIO_IDX].att[2];
    float heading_vel;

    while (!*((bool*)exit))
    {   // send velocity/rot_velocity to control the pioneer
        //   according to the reference position

        // get position and heading
        memcpy(pos_ref, robot_ref[PIO_IDX].enu, 3*sizeof(float)); // ref x/y/z
        memcpy(pos, data->robot[PIO_IDX].enu, 3*sizeof(float)); // actual x/y/z
        heading_ref = robot_ref[PIO_IDX].heading; // ref heading
        heading = data->robot[PIO_IDX].att[2]; // actual heading
        heading_vel = pioneer_calculate_heading_velocity(heading, previous_heading, 0.1);
        previous_heading = heading;

        // position error
        float err_pos[2] = {pos_ref[0]-pos[0], pos_ref[1]-pos[1]};
        // dest heading
        float dest_heading;

        if (std::sqrt((pos_ref[0]-pos[0])*(pos_ref[0]-pos[0]) + (pos_ref[1]-pos[1])*(pos_ref[1]-pos[1])) < PIONEER_POS_DEST_THRESHOLD)
        { // already arrive the pos_ref, adjusting heading to the ref heading
            pioneer_send_vw(0, 0);// stop the robot
            //pioneer_send_vw(0, pioneer_calculate_w_pid(pioneer_calculate_rot_angle(heading, heading_ref), heading_vel));
        }
        else
        { // move robot forward to the reference position
            dest_heading = std::atan2(-err_pos[0], err_pos[1]);
            pioneer_send_vw(pioneer_calculate_v(pos_ref, pos), pioneer_calculate_w_pid(pioneer_calculate_rot_angle(heading, dest_heading), heading_vel));

//printf("pos_ref = [%f, %f], pos = [%f, %f]\n", pos_ref[0], pos_ref[1], pos[0], pos[1]);
        }
        
        // 10 Hz
        nanosleep(&req, &rem); // 100 ms
    }

    net_send_bear_cmd_terminate();
}
#endif
