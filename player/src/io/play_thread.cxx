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
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <hdf5.h>
/* thread */
#include <pthread.h>
/* FOC */
#include "foc/flying_odor_compass.h"
#include "play_thread.h"
#include "robot/robot.h"

std::string file_to_play = "/Users/roice/workspace/ExPlat/SnifferBee/player/data/Record_2016-08-03_17-30-06.h5";

Flying_Odor_Compass* foc = NULL;

float sensor_reading[2000][3] = {0};
float position[2000][3] = {0};
float attitude[2000][3] = {0};
float wind[2000][3] = {0};
int count[2000] = {0};

static pthread_t play_thread_handle;
static bool exit_play_thread = false;

static void* player_loop(void*);

bool play_thread_init(void)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    if (std::strcmp(file_to_play.c_str(), "") == 0)
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
    dataset_id = H5Dopen2(file_id, "robot1/wind", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, wind);
    status = H5Dclose(dataset_id);
    dataset_id = H5Dopen2(file_id, "robot1/count", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, count);
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
        delete foc;
        foc = NULL;
        // exit play thread
        exit_play_thread = true;
        pthread_join(play_thread_handle, NULL);
        printf("Play thread terminated.\n");
    }
}

static void* player_loop(void* exit)
{
    struct timespec req, rem;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 100000000; // 0.1 s

    FOC_Input_t input;

    foc = new Flying_Odor_Compass();

    printf("Begin reading file.\n");

    for (int i = 0; i < 600; i++)
    {
        // read position
        memcpy(&input.position[0], &position[i][0], 3*sizeof(float));
        // read attitude
        memcpy(&input.attitude[0], &attitude[i][0], 3*sizeof(float));
        // read wind (disturbance)
        memcpy(&input.wind[0], &wind[i][0], 3*sizeof(float));

#if 1
        // check if there are data missing problem
        if (i > 0 and count[i]-count[i-1]>1) {
            for (int j = 1; j <= count[i]-count[i-1]; j++) {
                input.mox_reading[0] = (sensor_reading[i][0]-sensor_reading[i-1][0])/(count[i]-count[i-1])*j+sensor_reading[i][0];
                input.mox_reading[1] = (sensor_reading[i][1]-sensor_reading[i-1][1])/(count[i]-count[i-1])*j+sensor_reading[i][1];
                input.mox_reading[2] = (sensor_reading[i][2]-sensor_reading[i-1][2])/(count[i]-count[i-1])*j+sensor_reading[i][2];
                // convert 3.3-0.8 to 0.8-3.3
                input.mox_reading[0] = 3.3 - input.mox_reading[0];
                input.mox_reading[1] = 3.3 - input.mox_reading[1];
                input.mox_reading[2] = 3.3 - input.mox_reading[2];
                input.count ++;
                foc->update(input);
            }
        }
        else {
            input.mox_reading[0] = 3.3 - sensor_reading[i][0];
            input.mox_reading[1] = 3.3 - sensor_reading[i][1];
            input.mox_reading[2] = 3.3 - sensor_reading[i][2];
            input.count = count[i];
            foc->update(input);
        }
#else
        input.mox_reading[0] = 3.3 - sensor_reading[i][0];
        input.mox_reading[1] = 3.3 - sensor_reading[i][1];
        input.mox_reading[2] = 3.3 - sensor_reading[i][2];
        foc->update(input);
#endif
        // robot state update
        robot_state_t* robot_state = robot_get_state();
        memcpy(robot_state->position, input.position, 3*sizeof(float));
        memcpy(robot_state->attitude, input.attitude, 3*sizeof(float));
        memcpy(robot_state->wind, input.wind, 3*sizeof(float));

        nanosleep(&req, &rem); // 0.1 s
        if (*((bool*)exit)) break;
    }

#if 0
    while (!*((bool*)exit))
    {
        // never change, 0.5s to avoid dead
        nanosleep(&req, &rem); // 0.1 s
    }
#endif

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
