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

#define FILE "../data/Record_2016-07-08_15-24-37.h5"

int main(int argc, char* argv[])
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    float sensor_reading[1000][3] = {0};
    
    file_id = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, "/sensors_of_robot_0", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sensor_reading);
    status = H5Dclose(dataset_id);

    printf("Sensor readings has been loaded.\n");

    FOC_Input_t input;   
    Flying_Odor_Compass foc;
    for (int i = 0; i < 1000; i++)
    {
        // convert 3.3-0.8 to 0.8-3.3
        input.mox_reading[0] = 3.3 - sensor_reading[i][0];
        input.mox_reading[1] = 3.3 - sensor_reading[i][1];
        input.mox_reading[2] = 3.3 - sensor_reading[i][2];

        foc.update(input);
    }

    Record_Data(foc.foc_input, foc.foc_ukf_out, foc.foc_interp_out, foc.foc_diff_out, foc.foc_peak_time);

    return 0;
}
