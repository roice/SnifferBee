/* 
 * Measure odor concentration while robot flying along a circle
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2017.02.06
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <cmath>
/* thread */
#include <pthread.h>
/* GSRAO */
#include "mocap/packet_client.h"
#include "robot/robot.h"
#include "GSRAO_Config.h"

static pthread_t circle_measure_thread_handle;
static bool exit_circle_measure_thread = false;

static void* circle_measure_loop(void*);

bool circle_measure_init(void)
{
    /* create circle_measure_loop */
    exit_circle_measure_thread = false;
    if(pthread_create(&circle_measure_thread_handle, NULL, &circle_measure_loop, (void*)&exit_circle_measure_thread) != 0)
        return false;

    return true;
}

void circle_measure_stop(void)
{
    if(!exit_circle_measure_thread) // to avoid close twice
    {
        // exit back-forth measure thread
        exit_circle_measure_thread = true;
        pthread_join(circle_measure_thread_handle, NULL);
        printf("circle measure thread terminated\n");
    }
}

static void* circle_measure_loop(void* exit)
{
    struct timespec req, rem;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 100000000; // 0.1 s

    // velocity setting
    float moving_vel = 10; // degree
    // circle location setting
    float pos_center_circle[3] = {-0., -2.8, 1.30}; // m
    float R = 2.0; // m
    // angle_A positive, angle_B negative
    float angle_A = 45.0; // degree
    float angle_B = -45.0; // degree

    // current angle
    float angle_current = angle_A;

    // init robot
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    // robot 1
    robot_ref[0].enu[0] = -std::sin(angle_A*M_PI/180.0)*R+pos_center_circle[0];
    robot_ref[0].enu[1] = std::cos(angle_A*M_PI/180.0)*R+pos_center_circle[1];
    robot_ref[0].enu[2] = pos_center_circle[2];
    //robot_ref[0].heading = -M_PI/2; // heading to east
    robot_ref[0].heading = 0; // heading to north
    
    bool direction = true; // true for A-B, false for B-A

    //for (int i = 0; i < 150; i++)
    //    nanosleep(&req, &rem); // 0.1 s

    while (!*((bool*)exit))
    {
        if (direction) { // if A-B
            if (angle_current < angle_B) {
                direction = !direction;
            }
            else {
                angle_current -= moving_vel*0.1; // 0.1 s
            }
        }
        else {
            if (angle_current > angle_A) {
                direction = !direction;
            }
            else {
                angle_current += moving_vel*0.1; // 0.1 s
            }
        }

        robot_ref[0].enu[0] = -std::sin(angle_current*M_PI/180.0)*R+pos_center_circle[0];
        robot_ref[0].enu[1] = std::cos(angle_current*M_PI/180.0)*R+pos_center_circle[1];
        
        nanosleep(&req, &rem); // 0.1 s
    }
}
