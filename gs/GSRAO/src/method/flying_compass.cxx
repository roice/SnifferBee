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
/* Experiment Scene */
#include "scene/scene.h"
/* Plume Finding */
#include "method/plume_finding.h"
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

    srand(time(NULL));

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
    req.tv_nsec = 40000000; // 0.04 s

    // moving step
    float moving_step = 0.2; // m

    // azimuth obtained from FOC routine
    float azimuth;

    // int position of robot
    //float pos_robot[3] = {0, 2.0, 1.2}; // north
    //float pos_robot[3] = {2.4, 1.5, 1.8}; // north-east
    float pos_robot[3] = {-1.2, 1.8, 0.8}; // north-west

    int mb_idx = 0; // microbee 1

    // init robot
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    // robot 1
    robot_ref[mb_idx].enu[0] = pos_robot[0];
    robot_ref[mb_idx].enu[1] = pos_robot[1];
    robot_ref[mb_idx].enu[2] = pos_robot[2];
    robot_ref[mb_idx].heading = 0; // heading to north

    /* init plume finding */
    Plume_Finding   pfinding(pos_robot[0], pos_robot[1], pos_robot[2], 0.5, 0.05, 0.05, 0.04);

    /* init flying odor compass */
    Flying_Odor_Compass foc;
    FOC_Input_t input;

    /* get robot state */
    std::vector<Robot_Record_t>* robot_rec = robot_get_record(); 

    for (int i = 0; i < 150; i++)
        nanosleep(&req, &rem); // 0.1 s

    /* filtering of gas source direction */
    int recent_num_est = 5;
    double sum_direction[3] = {0};
    int size_of_foc_est;
    float planar_e[2] = {0};

    /* count continuous gas missing events */
    int count_gas_missing = 0;

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
            /* detected the gas flow direction */
                count_gas_missing = 0; 
                if (foc.data_est.size() >= recent_num_est) {
                    size_of_foc_est = foc.data_est.size();
                    // average this azimuth angle to obtain a stable angle
                    memset(sum_direction, 0, 3*sizeof(double));
                    for (int i = 0; i < recent_num_est; i++) {
                        if (foc.data_est.at(size_of_foc_est-recent_num_est+i).t > foc.data_est.at(size_of_foc_est-1).t-5.0)
                            for (int j = 0; j < 3; j++)
                                sum_direction[j] += foc.data_est.at(size_of_foc_est-recent_num_est+i).direction[j];
                    }
                    for (int j = 0; j < 2; j++)
                        planar_e[j] = -sum_direction[j]/(std::sqrt(sum_direction[0]*sum_direction[0]+sum_direction[1]*sum_direction[1]));
                    // TODO: elevation angle
        
                    // calculate the next waypoint
                    for (int i = 0; i < 2; i++)
                        pos_robot[i] += planar_e[i]*moving_step;
                    if (!scene_change_ref_pos(mb_idx, pos_robot)) {
                        for (int i = 0; i < 2; i++)
                            pos_robot[i] -= planar_e[i]*moving_step;
                    }
                    printf("Odor Tracing, the ref pos of robot is [ %.2f %.2f %.2f ]\n", pos_robot[0], pos_robot[1], pos_robot[2]);

                }
                printf("Reinit plume finding.\n");
                pfinding.reinit(pos_robot[0], pos_robot[1], pos_robot[2]);
            }
            else {
            /* cannot figure out the gas flow direction */
                if ((count_gas_missing++)*0.04 > 5.0) {
                    // 5 s not detect gas
                    pfinding.update();
                    if(!scene_change_ref_pos(mb_idx, pfinding.current_position)) {
                        // step forward by a random step length
                        float distance2orig = std::sqrt(std::pow(pfinding.current_position[0],2)+std::pow(pfinding.current_position[1],2)+std::pow(pfinding.current_position[2],2));
                        for (int i = 0; i < 3; i++)
                            pfinding.current_position[i] += 0.2*((float)rand()-RAND_MAX/2)/(float)RAND_MAX;
                        if(!scene_change_ref_pos(mb_idx, pfinding.current_position)) {
                            pfinding.reinit(pos_robot[0], pos_robot[1], pos_robot[2]);
                        }
                        else
                            memcpy(pos_robot, pfinding.current_position, 3*sizeof(float));
                    }
                    else
                        memcpy(pos_robot, pfinding.current_position, 3*sizeof(float));
                    printf("Odor Finding, the ref pos of robot is [ %.2f %.2f %.2f ]\n", pos_robot[0], pos_robot[1], pos_robot[2]);
                }
            }
        }
        
        nanosleep(&req, &rem); // 0.05 s
    }
}
