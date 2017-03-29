/* 
 * Measure odor concentration while robot hovering
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.06.02
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
#include "robot/robot.h"
#include "GSRAO_Config.h"

static pthread_t hover_measure_thread_handle;
static bool exit_hover_measure_thread = false;

static void* hover_measure_loop(void*);

bool hover_measure_init(void)
{
    /* create hover_measure_loop */
    exit_hover_measure_thread = false;
    if(pthread_create(&hover_measure_thread_handle, NULL, &hover_measure_loop, (void*)&exit_hover_measure_thread) != 0)
        return false;

    return true;
}

void hover_measure_stop(void)
{
    if(!exit_hover_measure_thread) // to avoid close twice
    {
        // exit hover measure thread
        exit_hover_measure_thread = true;
        pthread_join(hover_measure_thread_handle, NULL);
        printf("hover measure thread terminated\n");
    }
}

static void* hover_measure_loop(void* exit)
{
    struct timespec req, rem;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 500000000; // 0.5 s

    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    // robot 1
    robot_ref[0].enu[0] = 0.;
    robot_ref[0].enu[1] = 0.;
    robot_ref[0].enu[2] = 1.5;
    //robot_ref[0].heading = -M_PI/2; // heading to east
    robot_ref[0].heading = 0; // heading to north
    //robot_ref[0].heading = M_PI/4; // heading to north-west
    //robot_ref[0].heading = M_PI; // heading to south

    // robot 2
    robot_ref[1].enu[0] = 0;
    robot_ref[1].enu[1] = 0.6;
    robot_ref[1].enu[2] = 1.2;
    robot_ref[1].heading = -M_PI/2; // heading to east

    while (!*((bool*)exit))
    {
        // never change, 0.5s to avoid dead
        nanosleep(&req, &rem); // 0.5 s
    }
}
