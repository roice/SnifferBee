#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <hdf5.h>
#include <vector>
#include <time.h>
#include "foc/flying_odor_compass.h"

void Record_Data(Flying_Odor_Compass& foc)
{
    hid_t file_id, group_id, dataset_id, dataspace_id; 
    herr_t status;
    hsize_t data_dims[2];   // dataset dimensions
    float* data_pointer;
    
    // create file, if the file already exists, the current contents will be 
    // deleted so that the application can rewrite the file with new data.
    time_t t;
    struct tm* t_lo;
    char filename[60];
    t = time(NULL);
    t_lo = localtime(&t);
    strftime(filename, sizeof(filename), "FOC_Record_%Y-%m-%d_%H-%M-%S.h5", t_lo);
    //file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); 
    file_id = H5Fcreate("FOC_Record.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create a group named "/FOC" in the file */
    group_id = H5Gcreate2(file_id, "/FOC", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
    // save data_wind
    data_dims[0] = foc.data_wind.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "wind", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_wind.at(idx).wind[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer);   // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it.
    status = H5Sclose(dataspace_id); // Terminate access to the data space.
    free(data_pointer); // free space
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "wind_filtered", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_wind.at(idx).wind_filtered[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer);   // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it.
    status = H5Sclose(dataspace_id); // Terminate access to the data space.
    free(data_pointer); // free space

    // save data_raw
    data_dims[0] = foc.data_raw.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "mox_reading", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_raw.at(idx).mox_reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer);   // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it.
    status = H5Sclose(dataspace_id); // Terminate access to the data space.
    free(data_pointer); // free space
   
    // save data_denoise
    data_dims[0] = foc.data_denoise.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_denoise", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_denoise.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data 
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_interp
    data_dims[0] = foc.data_interp.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_interp", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_interp.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_smooth
    data_dims[0] = foc.data_smooth.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_smooth", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_smooth.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

#if defined(FOC_DELTA_METHOD_CROSS_CORRELATION)
    // save data_diff
    data_dims[0] = foc.data_diff.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_diff", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_diff.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

#elif defined(FOC_DELTA_METHOD_EDGE_DETECTION)
    // save data_gradient
    data_dims[0] = foc.data_gradient.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_gradient", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_gradient.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_edge
    data_dims[0] = foc.data_edge_max.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_edge_max", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_edge_max.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    
    data_dims[0] = foc.data_edge_min.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_edge_min", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_edge_min.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_cp
    int* int_pointer;
    data_dims[0] = foc.data_cp_max.size();
    data_dims[1] = FOC_NUM_SENSORS;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_cp_max", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    int_pointer = (int*)malloc(data_dims[0]*data_dims[1]*sizeof(*int_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(int_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_cp_max.at(idx).index[0]), FOC_NUM_SENSORS*sizeof(int));
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, int_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(int_pointer); // free space

    data_dims[0] = foc.data_cp_min.size();
    data_dims[1] = FOC_NUM_SENSORS;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_cp_min", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    int_pointer = (int*)malloc(data_dims[0]*data_dims[1]*sizeof(*int_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(int_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_cp_min.at(idx).index[0]), FOC_NUM_SENSORS*sizeof(int));
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, int_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(int_pointer); // free space
#endif

    // save data_delta
    data_dims[0] = foc.data_delta.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_std", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_delta.at(idx).std[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_toa", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_delta.at(idx).toa[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_est.wind_speed_xy
    data_dims[0] = foc.data_est.size();
    data_dims[1] = 2;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "est_wind_speed_xy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*2]), &(foc.data_est.at(idx).wind_speed_xy[0]), 2*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    // save data_est.wind_speed_filtered_xy
    data_dims[0] = foc.data_est.size();
    data_dims[1] = 2;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "est_wind_speed_filtered_xy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*2]), &(foc.data_est.at(idx).wind_speed_filtered_xy[0]), 2*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    // save data_est.wind_speed_en
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "est_wind_speed_en", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*2]), &(foc.data_est.at(idx).wind_speed_en[0]), 2*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    // save data_est.direction
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "direction", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*2]), &(foc.data_est.at(idx).direction[0]), 2*sizeof(float));
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
        char_pointer[idx] = foc.data_est.at(idx).valid ? 1 : 0;
    status = H5Dwrite(dataset_id, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, char_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(char_pointer); // free space

    /* Close group "/FOC" */
    status = H5Gclose(group_id);

    /* Terminate access to the file. */
    status = H5Fclose(file_id);
}
