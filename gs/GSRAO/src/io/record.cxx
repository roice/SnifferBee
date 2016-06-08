/*
 * Record experiment data of GSRAO
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.06.07
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <vector>
#include <time.h>
#include "robot/robot.h"
#include "io/serial.h" // macro MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE

void GSRAO_Save_Data(void)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;
    hsize_t data_dims[2];

    char ds_name[32];
    
    // create file, if the file already exists, the current contents will be 
    // deleted so that the application can rewrite the file with new data.
    time_t t;
    struct tm* t_lo;
    char filename[60];
    t = time(NULL);
    t_lo = localtime(&t);
    strftime(filename, sizeof(filename), "Record_%Y-%m-%d_%H-%M-%S.h5", t_lo);
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* get records */
    std::vector <Robot_Record_t>* robot_rec = robot_get_record();

    // save robot time info
    double* time_seq;
    for (int i = 0; i < 4; i++) // 4 robots max
    {
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 1; // time
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        snprintf(ds_name, 32, "time_of_robot_%d", i);
        dataset_id = H5Dcreate2(file_id, ds_name, H5T_NATIVE_DOUBLE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        time_seq = (double*)malloc(data_dims[0]*data_dims[1]*sizeof(*time_seq));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(time_seq[idx]), &(robot_rec[i].at(idx).time), sizeof(double)); // time
        status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, time_seq);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(time_seq);
    }

    // save robot enu info
    float* data;
    for (int i = 0; i < 4; i++) // 4 robots max
    {
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 3; // enu[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        snprintf(ds_name, 32, "enu_of_robot_%d", i);
        dataset_id = H5Dcreate2(file_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_rec[i].at(idx).enu[0]), 3*sizeof(float)); // enu
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);
    }

    // save robot att info
    for (int i = 0; i < 4; i++) // 4 robots max
    {
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 3; // att[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        snprintf(ds_name, 32, "att_of_robot_%d", i);
        dataset_id = H5Dcreate2(file_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_rec[i].at(idx).att[0]), 3*sizeof(float)); // att
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);
    }

    // save robot sensor info
    for (int i = 0; i < 4; i++) // 4 robots max
    {
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 3; // sensor[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        snprintf(ds_name, 32, "sensors_of_robot_%d", i);
        dataset_id = H5Dcreate2(file_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_rec[i].at(idx).sensor[0]), 3*sizeof(float)); // sensor
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);
    }

#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
    // save robot motor values
    int* motor_data;
    for (int i = 0; i < 4; i++) // 4 robots max
    {
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 4; // motor[4]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        snprintf(ds_name, 32, "motors_of_robot_%d", i);
        dataset_id = H5Dcreate2(file_id, ds_name, H5T_NATIVE_INT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        motor_data = (int*)malloc(data_dims[0]*data_dims[1]*sizeof(*motor_data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(motor_data[idx*4]), &(robot_rec[i].at(idx).motor[0]), 4*sizeof(int)); // motor
        status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, motor_data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(motor_data);
    }
#endif

    // save robot bat voltage
    for (int i = 0; i < 4; i++) // 4 robots max
    {
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 1; // bat volt
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        snprintf(ds_name, 32, "bat_volt_of_robot_%d", i);
        dataset_id = H5Dcreate2(file_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx]), &(robot_rec[i].at(idx).bat_volt), sizeof(float)); // bat volt
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);
    }

    /* Terminate access to the file. */
    status = H5Fclose(file_id);
}
