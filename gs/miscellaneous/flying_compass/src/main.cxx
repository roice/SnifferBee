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
//#define FILE "../data/Record_2016-08-19_15-27-58.h5" // illustrating method for paper
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
//#define FILE "../data/Record_2016-09-25_16-01-18.h5"
//#define FILE "../data/Record_2016-10-27_20-16-05.h5"
//#define FILE "../data/Record_2016-12-15_15-48-00.h5"
//#define FILE "../data/Record_2016-12-15_16-14-30.h5"
//#define FILE "../data/Record_2016-12-16_10-13-13.h5"
//#define FILE "../data/Record_2016-12-26_19-51-55.h5" // no alcohol
//#define FILE "../data/Record_2016-12-26_20-09-14.h5" // no alcohol, wind
//#define FILE "../data/Record_2016-12-26_20-14-25.h5" // alcohol blow ahead to red
//#define FILE "../data/Record_2016-12-26_20-42-10.h5"
//#define FILE "../data/Record_2016-12-26_22-16-19.h5" // alcohol blow ahead to red, hovering
//#define FILE "../data/Record_2016-12-27_20-42-37.h5"
//#define FILE "../data/Record_2016-12-27_21-07-35.h5"
//#define FILE "../data/Record_2016-12-27_21-18-44.h5"
//#define FILE "../data/Record_2016-12-28_21-36-30.h5"
//#define FILE "../data/Record_2017-01-02_22-58-37.h5"
//#define FILE "../data/Record_2017-01-02_22-58-37.h5"
//#define FILE "../data/Record_2017-01-03_17-55-39.h5" // static, alcohol, blow ahead to red
//#define FILE "../data/Record_2017-01-05_16-56-00.h5"
//#define FILE "../data/Record_2017-01-05_16-59-54.h5" // experiment for paper
//#define FILE "../data/Record_2017-01-05_17-47-48.h5"
//#define FILE "../data/Record_2017-01-05_17-50-39.h5"
//#define FILE "../data/Record_2017-01-05_19-38-45.h5" // alcohol, below 15cm, blow ahead to red, static
//#define FILE "../data/Record_2017-01-05_19-42-01.h5" // alcohol, below 15cm, blow ahead to red, hovering
//#define FILE "../data/Record_2017-01-05_20-18-15.h5" // alcohol, below 20cm, blow ahead to red, static
//#define FILE "../data/Record_2017-01-05_20-21-41.h5" // alcohol, below 20cm, blow ahead to red, hovering
//#define FILE "../data/Record_2017-01-06_10-05-02.h5" // alcohol, flying
//#define FILE "../data/Record_2017-01-19_17-28-01.h5" // alcohol, flying at (-0.6,-1.2,1.3)
//#define FILE "../data/Record_2017-01-19_18-54-20.h5" // alcohol, hovering at (-0.6,-1.3,1.2)
//#define FILE "../data/Record_2017-01-20_19-57-10.h5" // alcohol, flying at (-0.6,-1.2,1.4), experiment for paper
//#define FILE "../data/Record_2017-01-20_20-25-29.h5" // alcohol, flying at (-0.6,-1.2,1.25), experiment for paper, fan 25 Hz
//#define FILE "../data/Record_2017-01-20_20-56-55.h5" // alcohol, flying at (-0.6,-1.2,1.25), experiment for paper, fan not shake, fan 25 Hz
//#define FILE "../data/Record_2017-01-23_17-43-13.h5" // alcohol, flying at (-0.6, -1.2, 1.4), experiment for paper, fan not shake, fan 25 Hz
//#define FILE "../data/Record_2017-01-23_17-53-24.h5" // alcohol, flying at (-0.6, -1.2, 1.55), experiment for paper, fan not shake, fan 25 Hz
//#define FILE "../data/Record_2017-01-23_18-09-16.h5" // alcohol, flying at (-0.6, -1.8, 1.20), experiment for paper, fan not shake, fan 25 Hz
//#define FILE "../data/Record_2017-01-23_18-02-43.h5" // alcohol, flying at (-0.6, -1.8, 1.40), experiment for paper, fan not shake, fan 25 Hz
//#define FILE "../data/Record_2017-01-23_22-55-12.h5" // alcohol, flying at (-0.6, -1.8, 1.40), experiment for paper, fan not shake, fan 27 Hz
//#define FILE "../data/Record_2017-01-23_22-15-01.h5" // alcohol, flying at (-0.6, -1.8, 1.20), experiment for paper, fan not shake, fan 27 Hz
//#define FILE "../data/Record_2017-01-23_22-19-43.h5" // alcohol, flying at (-0.6, -1.8, 1.10), experiment for paper, fan not shake, fan 27 Hz

//#define FILE "../data/Record_2017-02-02_05-26-21.h5" // 1.4
//#define FILE "../data/Record_2017-02-02_05-36-54.h5" // 1.22

#define FILE "../data/Record_2017-02-06_15-34-59.h5" // alcohol, flying from (-1.6-0.6, -1.2, 1.25) to (1.6-0.6, -1.2, 1.25), fan shake, fan 25 Hz
//#define FILE "../data/Record_2017-02-05_20-13-00.h5" // alcohol, flying from (-1.6-0.6, -1.2, 1.25) to (1.6-0.6, -1.2, 1.25), fan shake, fan 25 Hz

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
    //for (int i = 20*60*1; i < 20*60*3; i++)
    for (int i = 20*(15); i < 20*(15+180); i++)
    //for (int i = 6000; i < 7000; i++)
    {
        // read position
        memcpy(&input.position[0], &position[i][0], 3*sizeof(float));
        // read attitude
        memcpy(&input.attitude[0], &attitude[i][0], 3*sizeof(float));
        // read wind (disturbance)
        //memcpy(&input.wind[0], &wind[i][0], 3*sizeof(float));
        for (int j = 0; j < 3; j++)
            input.wind[j] = 3.0*wind[i][j];

        input.mox_reading[0] = sensor_reading[i][0];
        input.mox_reading[1] = sensor_reading[i][1];
        input.mox_reading[2] = sensor_reading[i][2];
        foc.update(input);
    }

    Record_Data(foc);

    return 0;
}
