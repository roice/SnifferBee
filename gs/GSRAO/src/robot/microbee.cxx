/*
 * MicroBee Robot
 *         
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-05-23 create this file (robot_control.cxx)
 *       2016-05-25 change this file to microbee.cxx
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
/* thread */
#include <pthread.h>
/* GSRAO */
#include "mocap/packet_client.h"
#include "robot/microbee.h"
#include "robot/robot.h"
#include "io/serial.h"
#include "GSRAO_Config.h"

static pthread_t microbee_control_thread_handle;
static pthread_t microbee_state_thread_handle;
static bool exit_microbee_control_thread = false;
static bool exit_microbee_state_thread = false;
static MicroBee_t microbee[4]; // 4 robots max

static void* microbee_control_loop(void*);
static void* microbee_state_loop(void*);
static void microbee_pos_control(float, char);

/*-------- MicroBee State Refresh --------*/

/* microbee state refresh init */
bool microbee_state_init(void)
{
    // clear microbee states
    for (int i = 0; i < 4; i++) // 4 robots max
        memset(&(microbee[i].state), 0, sizeof(microbee[i].state));

    /* create state refresh loop */
    exit_microbee_state_thread = false;
    if (pthread_create(&microbee_state_thread_handle, NULL, &microbee_state_loop, (void*)&exit_microbee_state_thread) != 0)
        return false;

    return true;
}

/* close microbee state refresh loop */
void microbee_state_close(void)
{
    if (!exit_microbee_state_thread) // to avoid close twice
    {
        // exit microbee state refresh thread
        exit_microbee_state_thread = true;
        pthread_join(microbee_state_thread_handle, NULL);
        printf("microbee state thread terminated\n");
    }
}

MicroBee_t* microbee_get_states(void)
{
    return microbee;
}

static void* microbee_state_loop(void* exit)
{
    struct timespec req, rem, time;
    double current_time;
    int link_check_count = 0; // counter for link state check

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 100000000L; // 100 ms

    while (!*((bool*)exit))
    {
        // check link state
        if (++link_check_count >= 5) // 0.5 sec check once
        {
            clock_gettime(CLOCK_REALTIME, &time);
            current_time = time.tv_sec + time.tv_nsec/1.0e9;
            for (int i = 0; i < 4; i++) // 4 robots max
            {
                if (current_time - microbee[i].time > 0.5)
                    microbee[i].state.linked = false;
                else
                    microbee[i].state.linked = true;
            }
            link_check_count = 0;
        }

        // 10 Hz
        nanosleep(&req, &rem); // 100 ms
    }
}

/*-------- MicroBee Control ---------*/

/* microbee control init */
bool microbee_control_init(void)
{
    /* create trajectory control loop */
    exit_microbee_control_thread = false;
    if (pthread_create(&microbee_control_thread_handle, NULL, &microbee_control_loop, (void*)&exit_microbee_control_thread) != 0)
        return false;

    return true;
}

/* close microbee control loop */
void microbee_control_close(void)
{
    if (!exit_microbee_control_thread) // to avoid close twice
    {
        // exit microbee control thread
        exit_microbee_control_thread = true;
        pthread_join(microbee_control_thread_handle, NULL);
        printf("microbee control thread terminated\n");
    }
}

static void* microbee_control_loop(void* exit)
{
    struct timespec req, rem, time;
    double previous_time, current_time;
    float dtime;
    MocapData_t* data; // motion capture data

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 20000000L; // 20 ms

    // init previous_time, current_time, dtime
    clock_gettime(CLOCK_REALTIME, &time);
    current_time = time.tv_sec + time.tv_nsec/1.0e9;
    previous_time = current_time;
    dtime = 0;

    Robot_Ref_State_t* robot_ref = robot_get_ref_state();
robot_ref[0].enu[0] = 1.8;
robot_ref[0].enu[1] = -1.2;
robot_ref[0].enu[2] = 1.0;

    while (!*((bool*)exit))
    {
        clock_gettime(CLOCK_REALTIME, &time);
        current_time = time.tv_sec + time.tv_nsec/1.0e9;
        dtime = current_time - previous_time;
        previous_time = current_time;

        // position control, PID
        microbee_pos_control(dtime, 0);

        // 50 Hz
        nanosleep(&req, &rem); // 20 ms
    }
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

static void microbee_throttle_control(float dt, char robot_index)
{
    static float errorVelocityI[4] = {0}; // 4 robots max

    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile;

    // get alt
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    float EstAlt = data->robot[robot_index].enu[2]; // z axis, robot's real-time z axis pos
    float AltHold = robot_ref[robot_index].enu[2]; // reference z axis pos
    float vel_temp = data->robot[robot_index].vel[2]; // current alt velocity
    float accZ_temp = data->robot[robot_index].acc[2]; // current alt acc

    //printf("EstAlt %f, AltHold %f, vel_temp %f, accZ_temp %f\n", EstAlt, AltHold, vel_temp, accZ_temp);

    /* altitude control, throttle */
    // Altitude P-Controller
    float error = constrain(AltHold - EstAlt, -0.5, 0.5); // -0.5 - 0.5 m boundary
    error = applyDeadband(error, 0.01); // 1 cm deadband, remove small P parameter to reduce noise near zero position
    float setVel = constrain((pidProfile[robot_index].P[PIDALT]*error), -1.0, 1.0); // limit velocity to +/- 1.0 m/s

    // Velocity PID-Controller
    // P
    error = setVel - vel_temp;
    float result = constrain((pidProfile[robot_index].P[PIDVEL]*error), -250, 250); // limit
    // I
    errorVelocityI[robot_index] += (pidProfile[robot_index].I[PIDVEL]*error);
    errorVelocityI[robot_index] = constrain(errorVelocityI[robot_index], -200.0, 200.0); // limit
    result += errorVelocityI[robot_index];
    // D
    result -= constrain(pidProfile[robot_index].D[PIDVEL]*accZ_temp, -0.5, 0.5); // limit

    //printf("Alt adj = %f\n", result);

    // update throttle value
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
    rc_data[robot_index].throttle = constrain(1250 + result, 1000, 2000);
}

static void microbee_roll_pitch_control(float dt, char robot_index)
{
    static float errorPositionI[4][2] = {0};
    float error, result;

    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile;

    // get east/north
    float pos[2], pos_ref[2], vel[2]; // e/n
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref states
    for (char i = 0; i < 2; i++) // e/n
    {
        pos[i] = data->robot[robot_index].enu[i]; // e/n axis, robot's real-time x y axis pos
        pos_ref[i] = robot_ref[robot_index].enu[i]; // reference x y axis pos
        vel[i] = data->robot[robot_index].vel[i]; // current e/n velocity
    }

    // Position PID-Controller for east(x)/north(y) axis
    for (char i = 0; i < 2; i++) // 0 for roll, 1 for pitch
    {
        error = constrain(pos_ref[i] - pos[i], -2.0, 2.0); // limit error to -2.0~2.0 m
        error = applyDeadband(error, 0.01); // 1 cm deadband, remove small P parameter to reduce noise near zero position

        // P
        result = constrain((pidProfile[robot_index].P[PIDPOS]*error), -250, 250); // limit
        // I
        errorPositionI[robot_index][i] += (pidProfile[robot_index].I[PIDPOS]*error);
        errorPositionI[robot_index][i] = constrain(errorPositionI[robot_index][i], -200.0, 200.0); // limit
        result += errorPositionI[robot_index][i];
        // D
        result -= constrain(pidProfile[robot_index].D[PIDPOS]*vel[i], -0.5, 0.5); // limit

        // update roll/pitch value
        SPP_RC_DATA_t* rc_data = spp_get_rc_data();
        if (i == 0)
            rc_data[robot_index].roll = constrain(1500 + result, 1000, 2000);
        else if (i == 1)
            rc_data[robot_index].pitch = constrain(1500 + result, 1000, 2000);
    }
}

static void microbee_yaw_control(float dt, char robot_index)
{
    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile;

    // get yaw
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    float yaw = data->robot[robot_index].att[2]; // real-time yaw angle
    float yaw_ref = robot_ref[robot_index].heading; // reference yaw angle

    //error
}

static void microbee_pos_control(float dt, char robot_index)
{
    microbee_throttle_control(dt, robot_index);
    microbee_roll_pitch_control(dt, robot_index);
    microbee_yaw_control(dt, robot_index);
}