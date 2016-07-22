/* 
 * Measure odor concentration while robot move back and forth
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.07.07
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
#include "robot/robot.h"
#include "GSRAO_Config.h"

static pthread_t back_forth_measure_thread_handle;
static bool exit_back_forth_measure_thread = false;

static void* back_forth_measure_loop(void*);

bool back_forth_measure_init(void)
{
    /* create back_forth_measure_loop */
    exit_back_forth_measure_thread = false;
    if(pthread_create(&back_forth_measure_thread_handle, NULL, &back_forth_measure_loop, (void*)&exit_back_forth_measure_thread) != 0)
        return false;

    return true;
}

void back_forth_measure_stop(void)
{
    if(!exit_back_forth_measure_thread) // to avoid close twice
    {
        // exit back-forth measure thread
        exit_back_forth_measure_thread = true;
        pthread_join(back_forth_measure_thread_handle, NULL);
        printf("back-forth measure thread terminated\n");
    }
}

static void* back_forth_measure_loop(void* exit)
{
    struct timespec req, rem;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 100000000; // 0.1 s

    // velocity setting
    float moving_vel = 0.3; // 0.3 m/s
    // back-forth location setting
    float moving_pos_A[3] = {0.3, 0, 1.7};
    float moving_pos_B[3] = {0.3, -2.4, 1.7};

    // init robot
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    // robot 1
    robot_ref[0].enu[0] = moving_pos_A[0];
    robot_ref[0].enu[1] = moving_pos_A[1];
    robot_ref[0].enu[2] = moving_pos_A[2];
    robot_ref[0].heading = -M_PI/2; // heading to east
    //robot_ref[0].heading = 0; // heading to north
    MocapData_t* data = mocap_get_data(); // get mocap data

    float pos_err_A[3], pos_err_B[3], norm_pos_err;
    float e[3];
    bool direction = false; // true for A-B, false for B-A

    for (int i = 0; i < 150; i++)
        nanosleep(&req, &rem); // 0.1 s

    while (!*((bool*)exit))
    {
        for (int i = 0; i < 3; i++) {
            pos_err_A[i] = moving_pos_A[i] - data->robot[0].enu[i];
            pos_err_B[i] = moving_pos_B[i] - data->robot[0].enu[i];
        }
        // if robot reach A, then return to B
        if (fabs(pos_err_A[0]) < 0.3 && fabs(pos_err_A[1]) < 0.3 && fabs(pos_err_A[2]) < 0.3) {
            direction = true;
        }
        // if robot reach B, then return to A
        else if (fabs(pos_err_B[0]) < 0.3 && fabs(pos_err_B[1]) < 0.3 && fabs(pos_err_B[2]) < 0.3) {
            direction = false;
        }

        if (direction) { // destination B
            norm_pos_err = sqrt(pos_err_B[0]*pos_err_B[0] + pos_err_B[1]*pos_err_B[1]);
            for (int i = 0; i < 2; i++) {
                e[i] = pos_err_B[i] / norm_pos_err;
                robot_ref[0].enu[i] += e[i] * moving_vel * 0.1; // 0.1 s
            }
        }
        else { // destination A
            norm_pos_err = sqrt(pos_err_A[0]*pos_err_A[0] + pos_err_A[1]*pos_err_A[1]);
            for (int i = 0; i < 2; i++) {
                e[i] = pos_err_A[i] / norm_pos_err;
                robot_ref[0].enu[i] += e[i] * moving_vel * 0.1; // 0.1 s
            }
        }
        

        nanosleep(&req, &rem); // 0.1 s
    }
}
