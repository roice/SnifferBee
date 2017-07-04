/* 
 * Odor tracing using a odor compass
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2017.02.26
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
#include "GSRAO_thread_comm.h"
#include "io/udp_ocdev.h"
/* Odor Compass */
#include "method/foc/flying_odor_compass.h"

static pthread_t odor_compass_thread_handle;
static bool exit_odor_compass_thread = false;

Flying_Odor_Compass* method_odor_compass_foc = NULL;

static void* odor_compass_loop(void*);

bool odor_compass_init(void)
{
    /* create odor_compass_loop */
    exit_odor_compass_thread = false;
    if(pthread_create(&odor_compass_thread_handle, NULL, &odor_compass_loop, (void*)&exit_odor_compass_thread) != 0)
        return false;

    return true;
}

void odor_compass_stop(void)
{
    if(!exit_odor_compass_thread) // to avoid close twice
    {
        // exit odor compass thread
        exit_odor_compass_thread = true;
        pthread_join(odor_compass_thread_handle, NULL);
        printf("Odor compass thread terminated\n");
    }
}

static void* odor_compass_loop(void* exit)
{
    struct timespec req, rem;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 40000000; // 0.04 s, 25 Hz

    // moving step
    float moving_step = 1.0; // m

    int robo_idx = 0; // only one pioneer

    /* init odor compass */
    Flying_Odor_Compass* foc = new Flying_Odor_Compass;
    foc->type_of_robot = 0; // ground robot
    FOC_Input_t input;
    // for external access of this class
    method_odor_compass_foc = foc;

    /* get robot state */
    std::vector<Robot_Record_t>* robot_rec = robot_get_record();

    GSRAO_thread_comm_t* tc = GSRAO_get_thread_comm();

    /* filtering of gas source direction */
    int recent_num_est = 10;
    int size_of_foc_est;
    double sum_direction[3] = {0};
    float planar_e[2] = {0};

    bool foc_updated = false;

    int foc_update_count = 0;

    while (!*((bool*)exit))
    {
        if (robot_rec[robo_idx].size() > 0) {
            
            // retrieve sensor value and push to FOC
            // and get azimuth angle estimated via FOC routine
            //memcpy(&input.position[0], robot_rec[robo_idx].back().enu, 3*sizeof(float));
            //memcpy(&input.attitude[0], robot_rec[robo_idx].back().att, 3*sizeof(float));
            memcpy(&input.mox_reading[0], robot_rec[robo_idx].back().sensor, 3*sizeof(float));
            //memcpy(&input.wind[0], robot_rec[robo_idx].back().wind, 3*sizeof(float));
            // wait for the complete of signal sampling
            //  and reload mox readings to foc
            foc_updated = foc->update(input);

#if 0
            if (foc_updated) {

printf("back().direction = [%f, %f]\n", foc->data_est.back().direction[0], foc->data_est.back().direction[1]);

                // add this azimuth angle to the sum_direction
                for (int j = 0; j < 3; j++)
                    sum_direction[j] += foc->data_est.back().direction[j];
            }
            recent_num_est_count++;
            if (recent_num_est_count >= recent_num_est) {
                recent_num_est_count = 0;
                if (sum_direction[0] != 0. or sum_direction[1] != 0.) {

printf("sum_direction = [%f, %f]\n", sum_direction[0], sum_direction[1]);

                    // calculate the next waypoint
                    for (int j = 0; j < 2; j++)
                        planar_e[j] = -sum_direction[j]/(std::sqrt(sum_direction[0]*sum_direction[0]+sum_direction[1]*sum_direction[1]));
                    for (int i = 0; i < 2; i++)
                        pos_robot[i] += planar_e[i]*moving_step;
                    memcpy(robot_ref[robo_idx].enu, pos_robot, 3*sizeof(float)); 
                    printf("The ref pos of robot is [ %.2f %.2f %.2f ]\n", pos_robot[0], pos_robot[1], pos_robot[2]);
                }
                memset(sum_direction, 0, sizeof(sum_direction));
            }
#endif


            if (foc_updated) {
                if (foc->data_est.size() >= recent_num_est) {
                    size_of_foc_est = foc->data_est.size();
                    // average this azimuth angle to obtain a stable angle
                    memset(sum_direction, 0, 3*sizeof(double));
                    for (int i = 0; i < recent_num_est; i++) {
                        for (int j = 0; j < 3; j++)
                            sum_direction[j] += foc->data_est.at(size_of_foc_est-recent_num_est+i).direction[j];
                    }
                    for (int j = 0; j < 2; j++)
                        planar_e[j] = -sum_direction[j]/(std::sqrt(sum_direction[0]*sum_direction[0]+sum_direction[1]*sum_direction[1]));
                    // TODO: elevation angle
        
                    // calculate the next waypoint
                    for (int i = 0; i < 2; i++)
                        pos_robot[i] += planar_e[i]*moving_step;
                    memcpy(robot_ref[robo_idx].enu, pos_robot, 3*sizeof(float));

printf("The ref pos of robot is [ %.2f %.2f %.2f ]\n", pos_robot[0], pos_robot[1], pos_robot[2]);

                }
            }


        }
        
        //nanosleep(&req, &rem); // 0.1 s
    }
}
