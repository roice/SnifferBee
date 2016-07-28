#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <hdf5.h>
#include <vector>
#include <time.h>
#include "foc/flying_odor_compass.h"

void Record_Data(std::vector<FOC_Wind_t>& data_wind, std::vector<FOC_Input_t>& data_raw, std::vector<FOC_Reading_t>& data_denoise,
        std::vector<FOC_Reading_t>& data_interp, std::vector<FOC_Reading_t>& data_smooth, std::vector<FOC_Reading_t>& data_diff, std::vector<FOC_Delta_t>& data_delta,
        std::vector<double>* data_peak_time)
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
    data_dims[0] = data_wind.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "wind_direction", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(data_wind.at(idx).dir[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer);   // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it.
    status = H5Sclose(dataspace_id); // Terminate access to the data space.
    free(data_pointer); // free space

    // save data_raw
    data_dims[0] = data_raw.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "mox_reading", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(data_raw.at(idx).mox_reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer);   // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it.
    status = H5Sclose(dataspace_id); // Terminate access to the data space.
    free(data_pointer); // free space
   
    // save data_denoise
    data_dims[0] = data_denoise.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_denoise", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(data_denoise.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data 
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_interp
    data_dims[0] = data_interp.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_interp", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(data_interp.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_smooth
    data_dims[0] = data_smooth.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_smooth", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(data_smooth.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_diff
    data_dims[0] = data_diff.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_diff", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(data_diff.at(idx).reading[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_delta
    data_dims[0] = data_delta.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_std", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(data_delta.at(idx).std[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_toa", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(data_delta.at(idx).toa[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space


    /* Close group "/FOC" */
    status = H5Gclose(group_id);

    /* Terminate access to the file. */
    status = H5Fclose(file_id);
}
