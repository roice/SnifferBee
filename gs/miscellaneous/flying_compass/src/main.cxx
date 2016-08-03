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

#define FILE "../data/Record_2016-08-03_17-30-06.h5"

int main(int argc, char* argv[])
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    float sensor_reading[2000][3] = {0};
    float attitude[2000][3] = {0};
    int count[2000] = {0};

    file_id = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, "robot1/mox", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sensor_reading);
    status = H5Dclose(dataset_id);
    dataset_id = H5Dopen2(file_id, "robot1/att", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, attitude);
    status = H5Dclose(dataset_id);
    dataset_id = H5Dopen2(file_id, "robot1/count", H5P_DEFAULT);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, count);
    status = H5Dclose(dataset_id);

    FOC_Input_t input;   
    Flying_Odor_Compass foc;
    for (int i = 0; i < 600; i++)
    {
        // read attitude
        memcpy(&input.attitude[0], &attitude[i][0], 3*sizeof(float));

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
#endif
        input.mox_reading[0] = 3.3 - sensor_reading[i][0];
        input.mox_reading[1] = 3.3 - sensor_reading[i][1];
        input.mox_reading[2] = 3.3 - sensor_reading[i][2];
        foc.update(input);
    }

    Record_Data(foc.data_wind, foc.data_raw, foc.data_denoise, foc.data_interp, foc.data_smooth, foc.data_diff, foc.data_delta, foc.data_est);

    return 0;
}
