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

#define FILE "../data/Record_2016-06-07_22-30-51.h5"

int main(int argc, char* argv[])
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    float sensor_reading[1000][3] = {0};
    float attitude[1000][3] = {0};
    
    file_id = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, "/sensors_of_robot_0", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sensor_reading);
    status = H5Dclose(dataset_id);
    dataset_id = H5Dopen2(file_id, "/att_of_robot_0", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, attitude);
    status = H5Dclose(dataset_id);

    FOC_Input_t input;   
    Flying_Odor_Compass foc;
    for (int i = 0; i < 300; i++)
    {
        // convert 3.3-0.8 to 0.8-3.3
        input.mox_reading[0] = 3.3 - sensor_reading[i][0];
        input.mox_reading[1] = 3.3 - sensor_reading[i][1];
        input.mox_reading[2] = 3.3 - sensor_reading[i][2];

        // read attitude
        memcpy(&input.attitude[0], &attitude[i][0], 3*sizeof(float));

        foc.update(input);
    }

    Record_Data(foc.data_wind, foc.data_raw, foc.data_denoise, foc.data_interp, foc.data_smooth, foc.data_diff, foc.data_delta, foc.data_est);

    return 0;
}
