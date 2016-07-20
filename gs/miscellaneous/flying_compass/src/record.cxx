#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <hdf5.h>
#include <vector>
#include <time.h>
#include "foc/flying_odor_compass.h"

void Record_Data(std::vector<FOC_Input_t>& foc_input, std::vector<FOC_Reading_t>& foc_ukf_out,
        std::vector<FOC_Reading_t>& foc_interp_out, std::vector<FOC_Reading_t>& foc_diff_out,
        std::vector<double>* foc_peak_time)
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
    strftime(filename, sizeof(filename), "FOC_Record_%Y-%m-%d_%H-%M-%S.h5", t_lo);
    //file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); 
    file_id = H5Fcreate("FOC_Record.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    float* data;
    
    // save foc input
    data_dims[0] = foc_input.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    // create data set
    dataset_id = H5Dcreate2(file_id, "mox_reading", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write data
    data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
    for (int idx = 0; idx < data_dims[0]; idx++)
        memcpy(&(data[idx*3]), &(foc_input.at(idx).mox_reading[0]), 3*sizeof(float)); // sensor
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
    /* End access to the dataset and release resources used by it. */
    status = H5Dclose(dataset_id);
    /* Terminate access to the data space. */ 
    status = H5Sclose(dataspace_id);
    // free space
    free(data);

    // save foc ukf output
    data_dims[0] = foc_ukf_out.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    // create data set
    dataset_id = H5Dcreate2(file_id, "mox_ukf_output", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write data
    data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
    for (int idx = 0; idx < data_dims[0]; idx++)
        memcpy(&(data[idx*3]), &(foc_ukf_out.at(idx).reading[0]), 3*sizeof(float)); // sensor
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
    /* End access to the dataset and release resources used by it. */
    status = H5Dclose(dataset_id);
    /* Terminate access to the data space. */ 
    status = H5Sclose(dataspace_id);
    // free space
    free(data);

    // save foc interp output
    data_dims[0] = foc_interp_out.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    // create data set
    dataset_id = H5Dcreate2(file_id, "mox_interp_output", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write data
    data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
    for (int idx = 0; idx < data_dims[0]; idx++)
        memcpy(&(data[idx*3]), &(foc_interp_out.at(idx).reading[0]), 3*sizeof(float)); // sensor
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
    /* End access to the dataset and release resources used by it. */
    status = H5Dclose(dataset_id);
    /* Terminate access to the data space. */ 
    status = H5Sclose(dataspace_id);
    // free space
    free(data);

    // save foc diff output
    data_dims[0] = foc_diff_out.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    // create data set
    dataset_id = H5Dcreate2(file_id, "mox_diff_output", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write data
    data = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data));
    for (int idx = 0; idx < data_dims[0]; idx++)
        memcpy(&(data[idx*3]), &(foc_diff_out.at(idx).reading[0]), 3*sizeof(float)); // sensor
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
    /* End access to the dataset and release resources used by it. */
    status = H5Dclose(dataset_id);
    /* Terminate access to the data space. */ 
    status = H5Sclose(dataspace_id);
    // free space
    free(data);

    // save foc peak output
    char dset_name[100];
    double* time_data;
    for (int sensor_idx = 0; sensor_idx < FOC_NUM_SENSORS; sensor_idx++) {
        data_dims[0] = foc_peak_time[sensor_idx].size();
        data_dims[1] = 1;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        // create data set
        snprintf(dset_name, 50, "mox_peak_time_%d", sensor_idx);
        dataset_id = H5Dcreate2(file_id, dset_name, H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // write data
        time_data = (double*)malloc(data_dims[0]*data_dims[1]*sizeof(*time_data));
        for (int idx = 0; idx < data_dims[0]; idx++)
            memcpy(&(time_data[idx]), &(foc_peak_time[sensor_idx].at(idx)), sizeof(double));
        status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, time_data);
        /* End access to the dataset and release resources used by it. */
        status = H5Dclose(dataset_id);
        /* Terminate access to the data space. */ 
        status = H5Sclose(dataspace_id);
        // free space
        free(time_data);
    }

    /* Terminate access to the file. */
    status = H5Fclose(file_id);
}
