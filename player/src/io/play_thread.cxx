/* 
 * Playing thread
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.08.10
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <hdf5.h>
/* thread */
#include <pthread.h>
/* FOC */
#include "foc/flying_odor_compass.h"
#include "play_thread.h"
#include "robot/robot.h"

#define MAX_LEN_READ_BUFFER (60*60*20)    // 60 min, 20 Hz
#define PLAY_SPEED  8

std::string file_to_play = "/Users/roice/workspace/ExPlat/SnifferBee/player/data/Record_2016-09-25_16-01-18.h5";

Flying_Odor_Compass* foc = NULL;

float sensor_reading[MAX_LEN_READ_BUFFER][3] = {0};
float position[MAX_LEN_READ_BUFFER][3] = {0};
float attitude[MAX_LEN_READ_BUFFER][3] = {0};
float wind[MAX_LEN_READ_BUFFER][3] = {0};
float count[MAX_LEN_READ_BUFFER] = {0};

static pthread_t play_thread_handle;
static bool exit_play_thread = false;

static void* player_loop(void*);

bool play_thread_init(void)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    if (strcmp(file_to_play.c_str(), "") == 0)
        return false;

    // display which file to open
    printf("Reading file: %s\n", file_to_play.c_str());

    file_id = H5Fopen(file_to_play.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, "robot1/mox", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sensor_reading);
    status = H5Dclose(dataset_id);
    dataset_id = H5Dopen2(file_id, "robot1/enu", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, position);
    status = H5Dclose(dataset_id);
    dataset_id = H5Dopen2(file_id, "robot1/att", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, attitude);
    status = H5Dclose(dataset_id);
    dataset_id = H5Dopen2(file_id, "robot1/count", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, count);
    status = H5Dclose(dataset_id);

    file_id = H5Fopen("/Users/roice/workspace/ExPlat/SnifferBee/player/data/wind.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, "wind", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, wind);
    status = H5Dclose(dataset_id);

    /* create play_thread_loop */
    exit_play_thread = false;
    if(pthread_create(&play_thread_handle, NULL, &player_loop, (void*)&exit_play_thread) != 0)
        return false;

    return true;
}

void play_thread_stop(void)
{
    if(!exit_play_thread and foc != NULL) // to avoid close twice
    { 
        // exit play thread
        exit_play_thread = true;
        pthread_join(play_thread_handle, NULL);
        delete foc;
        foc = NULL;
        printf("Play thread terminated.\n");
    }
}

static void* player_loop(void* exit)
{
    struct timespec req, rem;
    struct timeval  tv;
    unsigned long useconds;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 50000000/PLAY_SPEED; // 0.05 s, 20 Hz for 1x speed

    FOC_Input_t input;

    foc = new Flying_Odor_Compass();

    printf("Begin reading file.\n");

    for (int i = 0; i < 4*60*20; i++)   // 20 Hz
    {
        // get time
        gettimeofday(&tv, NULL);
        useconds = tv.tv_sec*1000000 + tv.tv_usec;

        // read position
        memcpy(&input.position[0], &position[i][0], 3*sizeof(float));
        // read attitude
        memcpy(&input.attitude[0], &attitude[i][0], 3*sizeof(float));
        // read wind (disturbance)
        memcpy(&input.wind[0], &wind[i][0], 3*sizeof(float));
        // read count
        memcpy(&input.count, &count[i], sizeof(int));

        input.mox_reading[0] = sensor_reading[i][0];
        input.mox_reading[1] = sensor_reading[i][1];
        input.mox_reading[2] = sensor_reading[i][2];
        foc->update(input);

        // robot state update
        robot_state_t* robot_state = robot_get_state();
        memcpy(robot_state->position, input.position, 3*sizeof(float));
        memcpy(robot_state->attitude, input.attitude, 3*sizeof(float));
        memcpy(robot_state->wind, input.wind, 3*sizeof(float));

        gettimeofday(&tv, NULL);
        if (tv.tv_sec*1000000 + tv.tv_usec - useconds < 50000/PLAY_SPEED) {    // 20 Hz
            // loop interval
            req.tv_sec = 0;
            req.tv_nsec = (50000/PLAY_SPEED + useconds - tv.tv_sec*1000000 - tv.tv_usec)*1000;
            nanosleep(&req, &rem);
        }

        if (*((bool*)exit)) break;
    }

    printf("End reading file.\n");

    return 0;
}

void play_thread_set_file_path(const char* file_path)
{
    if (file_path != NULL)
        file_to_play = file_path;
}

void* play_thread_get_data()
{
    return foc;
}
