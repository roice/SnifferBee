/*
 * Robot control
 *         
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-05-23 create this file
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
#include "robot/robot_control.h"

static pthread_t robot_control_thread_handle;
static bool exit_robot_control_thread = false;
static RobotState_t robot_state_ref; // reference state

static void* robot_control_loop(void* exit);

/* robot control init */
bool robot_control_init(void)
{
    /* create trajectory control loop */
    exit_robot_control_thread = false;
    if (pthread_create(&robot_control_thread_handle, NULL, &robot_control_loop, (void*)&exit_robot_control_thread) != 0)
        return false;

    return true;
}

/* close robot control loop */
void robot_control_close(void)
{
    // exit robot control thread
    exit_robot_control_thread = true;
    pthread_join(robot_control_thread_handle, NULL);
}

RobotState_t* robot_get_ref_state(void)
{
    return &robot_state_ref;
}

static void robot_pos_control(float);

static void* robot_control_loop(void* exit)
{
    struct timespec req, rem, time;
    double previous_time, current_time;
    float dtime;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 20000000L; // 20 ms

    // init previous_time, current_time, dtime
    clock_gettime(CLOCK_REALTIME, &time);
    current_time = time.tv_sec + time.tv_nsec/1.0e9;
    previous_time = current_time;
    dtime = 0;

    while (!*((bool*)exit))
    {
        clock_gettime(CLOCK_REALTIME, &time);
        current_time = time.tv_sec + time.tv_nsec/1.0e9;
        dtime = current_time - previous_time;
        previous_time = current_time;

        // position control, PID
        robot_pos_control(dtime);

        // 50 Hz
        nanosleep(&req, &rem); // 20 ms
    }
}

static void robot_pos_control(float dt)
{}
