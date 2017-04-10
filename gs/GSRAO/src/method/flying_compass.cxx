/* 
 * Odor tracing using a flying odor compass
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2017.02.07
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
/* Flying Odor Compass */
#include "method/foc/flying_odor_compass.h"

static pthread_t flying_compass_thread_handle;
static bool exit_flying_compass_thread = false;

static void* flying_compass_loop(void*);

bool flying_compass_init(void)
{
    /* create flying_compass_loop */
    exit_flying_compass_thread = false;
    if(pthread_create(&flying_compass_thread_handle, NULL, &flying_compass_loop, (void*)&exit_flying_compass_thread) != 0)
        return false;

    return true;
}

void flying_compass_stop(void)
{
    if(!exit_flying_compass_thread) // to avoid close twice
    {
        // exit back-forth measure thread
        exit_flying_compass_thread = true;
        pthread_join(flying_compass_thread_handle, NULL);
        printf("circle measure thread terminated\n");
    }
}

static void* flying_compass_loop(void* exit)
{
    struct timespec req, rem;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 50000000; // 0.05 s

    // moving step
    float moving_step = 0.01; // 1 cm

    // azimuth obtained from FOC routine
    float azimuth;

    // int position of robot
    float pos_robot[3] = {1.8, 0., 1.3};

    int mb_idx = 0; // microbee 1

    // init robot
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    // robot 1
    robot_ref[mb_idx].enu[0] = pos_robot[0];
    robot_ref[mb_idx].enu[1] = pos_robot[1];
    robot_ref[mb_idx].enu[2] = pos_robot[2];
    robot_ref[mb_idx].heading = 0; // heading to north

    /* init flying odor compass */
    Flying_Odor_Compass foc;
    FOC_Input_t input;

    /* get robot state */
    std::vector<Robot_Record_t>* robot_rec = robot_get_record(); 

    for (int i = 0; i < 150; i++)
        nanosleep(&req, &rem); // 0.1 s

    /* filtering of gas source direction */
    int recent_num_est = 10;
    double sum_direction[3] = {0};
    int size_of_foc_est;
    float planar_e[2] = {0};

    while (!*((bool*)exit))
    {
        if (robot_rec[mb_idx].size() > 0) {
            
            // retrieve sensor value and push to FOC
            // and get azimuth angle estimated via FOC routine
            memcpy(&input.position[0], robot_rec[mb_idx].back().enu, 3*sizeof(float));
            memcpy(&input.attitude[0], robot_rec[mb_idx].back().att, 3*sizeof(float));
            memcpy(&input.mox_reading[0], robot_rec[mb_idx].back().sensor, 3*sizeof(float));
            memcpy(&input.wind[0], robot_rec[mb_idx].back().wind, 3*sizeof(float));
            if (foc.update(input)) {
 
                if (foc.data_est.size() >= recent_num_est) {
                    size_of_foc_est = foc.data_est.size();
                    // average this azimuth angle to obtain a stable angle
                    memset(sum_direction, 0, 3*sizeof(double));
                    for (int i = 0; i < recent_num_est; i++) {
                        for (int j = 0; j < 3; j++)
                            sum_direction[j] += foc.data_est.at(size_of_foc_est-recent_num_est+i).direction[j];
                    }
                    for (int j = 0; j < 2; j++)
                        planar_e[j] = -sum_direction[j]/(std::sqrt(sum_direction[0]*sum_direction[0]+sum_direction[1]*sum_direction[1]));
                    // TODO: elevation angle
        
                    // calculate the next waypoint
                    for (int i = 0; i < 2; i++)
                        pos_robot[i] += planar_e[i]*moving_step;
                        memcpy(robot_ref[mb_idx].enu, pos_robot, 3*sizeof(float));

printf("The ref pos of robot is [ %.2f %.2f %.2f ]\n", pos_robot[0], pos_robot[1], pos_robot[2]);

                }
            }
        }
        
        nanosleep(&req, &rem); // 0.05 s
    }
}
