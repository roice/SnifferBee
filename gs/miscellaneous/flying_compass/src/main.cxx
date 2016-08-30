/*
 * Test file for flying odor compass
 *
 * Author:
 *      Roice, Luo
 * Date:
 *      2016.06.17
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <hdf5.h>
#include <vector>
#include "foc/flying_odor_compass.h"
#include "record.h"

//#define FILE "../data/Record_2016-08-03_17-30-06.h5"
//#define FILE "../data/Record_2016-08-19_14-32-35.h5"
//#define FILE "../data/Record_2016-08-19_16-36-49.h5"
//#define FILE "../data/Record_2016-08-19_16-45-13.h5"
//#define FILE "../data/Record_2016-08-19_16-52-27.h5"
//#define FILE "../data/Record_2016-08-19_16-56-45.h5"
//#define FILE "../data/Record_2016-08-23_15-28-25.h5"
//#define FILE "../data/Record_2016-08-25_18-26-08.h5"
//#define FILE "../data/Record_2016-08-25_21-10-17.h5"
//#define FILE "../data/Record_2016-08-25_21-17-16.h5"
//#define FILE "../data/Record_2016-08-29_14-59-40.h5"
//#define FILE "../data/Record_2016-08-29_17-09-52.h5"
//#define FILE "../data/Record_2016-08-29_17-35-28.h5"
//#define FILE "../data/Record_2016-08-29_18-29-01.h5"
//#define FILE "../data/Record_2016-08-30_08-58-21.h5"
//#define FILE "../data/Record_2016-08-30_09-08-46.h5"
#define FILE "../data/Record_2016-08-30_10-50-54.h5"

int main(int argc, char* argv[])
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    float sensor_reading[100000][3] = {0};
    float position[100000][3] = {0};
    float attitude[100000][3] = {0};
    float wind[100000][3] = {0};
    int count[100000] = {0};

    file_id = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
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

    FOC_Input_t input;   
    Flying_Odor_Compass foc;
    //for (int i = 25*30; i < 25*40; i++)
    for (int i = 25*60*0; i < 25*60*2; i++)
    //for (int i = 25*240; i < 25*250; i++)
    //for (int i = -200+550; i < -200+800; i++)
    {
        // read position
        memcpy(&input.position[0], &position[i][0], 3*sizeof(float));
        // read attitude
        memcpy(&input.attitude[0], &attitude[i][0], 3*sizeof(float));
        // read wind (disturbance)
        memcpy(&input.wind[0], &wind[i][0], 3*sizeof(float));

#if 0
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
                foc.update(input);
            }
        }
        else {
            input.mox_reading[0] = 3.3 - sensor_reading[i][0];
            input.mox_reading[1] = 3.3 - sensor_reading[i][1];
            input.mox_reading[2] = 3.3 - sensor_reading[i][2];
            foc.update(input);
        }
#else
        input.mox_reading[0] = 3.3 - sensor_reading[i][0];
        input.mox_reading[1] = 3.3 - sensor_reading[i][1];
        input.mox_reading[2] = 3.3 - sensor_reading[i][2];
        foc.update(input);
#endif
    }

    Record_Data(foc);

    return 0;
}
