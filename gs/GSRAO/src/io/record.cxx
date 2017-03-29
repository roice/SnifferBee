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
#include "io/record.h" // macro RECORD_ROBOT_DEBUG_INFO
#include "method/foc/flying_odor_compass.h"

extern Flying_Odor_Compass* method_odor_compass_foc;

void GSRAO_Save_Data(void)
{
    hid_t file_id, group_id, dataset_id, dataspace_id;
    herr_t status;
    hsize_t data_dims[2];

    char gr_name[32];
    char ds_name[128];
    
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
    std::vector<Robot_Record_t>* robot_rec = robot_get_record();
    std::vector<Robot_Debug_Record_t>* robot_debug_rec = robot_get_debug_record();

    double* time_seq;
    float* data;
    int* count_data;
#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
    int* motor_data;
#endif
 
    for (int i = 0; i < 4; i++) // 4 robots max
    {
        /* Create a group named "robot1~robot4" in the file */
        snprintf(gr_name, 32, "robot%d", i+1);
        group_id = H5Gcreate2(file_id, gr_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // robot time
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 1; // time
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "time", H5T_NATIVE_DOUBLE, dataspace_id,
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

        // robot enu info
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 3; // enu[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "enu", H5T_NATIVE_FLOAT, dataspace_id,
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

        // robot att info
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 3; // att[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "att", H5T_NATIVE_FLOAT, dataspace_id,
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

        // robot sensor info
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 3; // sensor[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "mox", H5T_NATIVE_FLOAT, dataspace_id,
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

        // robot measurement count number
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 1; // count
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "count", H5T_NATIVE_INT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        count_data = (int*)malloc(data_dims[0]*data_dims[1]*sizeof(*count_data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(count_data[idx]), &(robot_rec[i].at(idx).count), sizeof(int));
        status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, count_data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(count_data);

#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
        // robot motor values
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 4; // motor[4]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "motors", H5T_NATIVE_INT, dataspace_id,
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
#endif

        // robot bat voltage
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 1; // bat volt
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "bat_volt", H5T_NATIVE_FLOAT, dataspace_id,
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

        // robot wind measurement/estimation
        data_dims[0] = robot_rec[i].size();
        data_dims[1] = 3; // wind[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "wind", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_rec[i].at(idx).wind[0]), 3*sizeof(float)); // att
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "wind_p", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_rec[i].at(idx).wind_p[0]), 3*sizeof(float)); // att
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        /* close group "robot0~robot4" */
        status = H5Gclose(group_id);

#ifdef RECORD_ROBOT_DEBUG_INFO

        /* Create a group named "robot1/debug~robot4/debug" in the file */
        snprintf(gr_name, 32, "robot%d/debug", i+1);
        group_id = H5Gcreate2(file_id, gr_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


        // robot enu
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "enu", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).enu[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot att
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "att", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).att[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot vel
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "vel", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).vel[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot acc
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "acc", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).acc[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

#if 0
        // robot vel_p
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "vel_p", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).vel_p[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot acc_p
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "acc_p", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).acc_p[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);
#endif

        // robot leso_z1
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "leso_z1", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).leso_z1[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot leso_z2
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "leso_z2", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).leso_z2[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot leso_z3
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3; // leso_z3[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "leso_z3", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).leso_z3[0]), 3*sizeof(float)); // leso_z3
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot throttle
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 1;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "throttle", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx]), &(robot_debug_rec[i].at(idx).throttle), sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot roll
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 1;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "roll", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx]), &(robot_debug_rec[i].at(idx).roll), sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot pitch
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 1;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "pitch", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx]), &(robot_debug_rec[i].at(idx).pitch), sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot yaw
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 1;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "yaw", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx]), &(robot_debug_rec[i].at(idx).yaw), sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot wind_estimated
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "wind_estimated", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).wind_estimated), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        // robot wind_resist coef
        data_dims[0] = robot_debug_rec[i].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "wind_resist_coef", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(robot_debug_rec[i].at(idx).wind_resist_coef), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        for (int anemo_idx = 0; anemo_idx < 3; anemo_idx++) {
            // anemometer
            data_dims[0] = robot_debug_rec[i].size();
            data_dims[1] = 3;
            dataspace_id = H5Screate_simple(2, data_dims, NULL);
            // create data set
            snprintf(ds_name, 128, "anemometer_%d", anemo_idx+1);
            dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            // write data
            data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
            for (int idx = 0; idx < data_dims[0]; idx++)
                memcpy(&(data[idx*3]), robot_debug_rec[i].at(idx).anemometer[anemo_idx], 3*sizeof(float));
            status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, data);
            /* End access to the dataset and release resources used by it. */
            status = H5Dclose(dataset_id);
            /* Terminate access to the data space. */ 
            status = H5Sclose(dataspace_id);
            // free space
            free(data);
        }

        /* close group "robot1/debug ~ robot4/debug" */
        status = H5Gclose(group_id);

#endif
    }

    /* Create a group named "anemometers" in the file */
    group_id = H5Gcreate2(file_id, "anemometers", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<Anemometer_Data_t>* anemo_record = sonic_anemometer_get_wind_record();
 
    for (int anemo_idx = 0; anemo_idx < 4; anemo_idx++) {
        // anemometer
        data_dims[0] = anemo_record[anemo_idx].size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        snprintf(ds_name, 128, "%d", anemo_idx+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(data[idx*3]), &(anemo_record[anemo_idx].at(idx).speed[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);
    }
    /* close group "anemometers" */
    status = H5Gclose(group_id);

    if (method_odor_compass_foc != NULL)
    { 
        /* Create a group named "foc" in the file */
        group_id = H5Gcreate2(file_id, "foc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        printf("data_interp[0].size() = %d\n", method_odor_compass_foc->data_interp[0].size());
        // robot sensor info
        data_dims[0] = method_odor_compass_foc->data_interp[0].size();
        data_dims[1] = 3; // sensor[3]
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        dataset_id = H5Dcreate2(group_id, "data_interp", H5T_NATIVE_FLOAT, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            for (int idx_s = 0; idx_s < data_dims[1]; idx_s++)
                data[idx*3+idx_s] = method_odor_compass_foc->data_interp[idx_s].at(idx);
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(data);

        float* data_pointer;
 
        printf("data_est.size() = %d\n", method_odor_compass_foc->data_est.size());
        // save data_est
        data_dims[0] = method_odor_compass_foc->data_est.size();
        data_dims[1] = 3;
        // save data_est.t 
        dataspace_id = H5Screate_simple(1, data_dims, NULL); 
        dataset_id = H5Dcreate2(group_id, "est_t", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            data_pointer[idx] = method_odor_compass_foc->data_est.at(idx).t;
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
        
        // save data_est.pos
        dataspace_id = H5Screate_simple(2, data_dims, NULL); 
        dataset_id = H5Dcreate2(group_id, "est_pos", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*3]), &(method_odor_compass_foc->data_est.at(idx).pos[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
        // save data_est.wind_p
        dataspace_id = H5Screate_simple(2, data_dims, NULL); 
        dataset_id = H5Dcreate2(group_id, "est_wind_p", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*3]), &(method_odor_compass_foc->data_est.at(idx).wind_p[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
        // save data_est.wind
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        dataset_id = H5Dcreate2(group_id, "est_wind", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*3]), &(method_odor_compass_foc->data_est.at(idx).wind[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
        // save data_est.direction
        data_dims[0] = method_odor_compass_foc->data_est.size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        dataset_id = H5Dcreate2(group_id, "est_direction", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*3]), &(method_odor_compass_foc->data_est.at(idx).direction[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
        // save data_est.std
        data_dims[0] = method_odor_compass_foc->data_est.size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        dataset_id = H5Dcreate2(group_id, "est_std", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(method_odor_compass_foc->data_est.at(idx).std[0]), FOC_NUM_SENSORS*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
        // save data_est.clustering
        dataspace_id = H5Screate_simple(1, data_dims, NULL);
        dataset_id = H5Dcreate2(group_id, "est_clustering", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx]), &(method_odor_compass_foc->data_est.at(idx).clustering), sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space 
        // save data_est.dt
        dataspace_id = H5Screate_simple(1, data_dims, NULL);
        dataset_id = H5Dcreate2(group_id, "est_dt", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx]), &(method_odor_compass_foc->data_est.at(idx).dt), sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
        // save data_est.valid
        char* char_pointer;
        dataspace_id = H5Screate_simple(1, data_dims, NULL);
        dataset_id = H5Dcreate2(group_id, "est_valid", H5T_NATIVE_CHAR, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        char_pointer = (char*)malloc(data_dims[0]*sizeof(*char_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            char_pointer[idx] = method_odor_compass_foc->data_est.at(idx).valid ? 1 : 0;
        status = H5Dwrite(dataset_id, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, char_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(char_pointer); // free space

        /* close group "foc" */
        status = H5Gclose(group_id);
    }
    delete method_odor_compass_foc;
    method_odor_compass_foc = NULL;


    /* Terminate access to the file. */
    status = H5Fclose(file_id);
}
