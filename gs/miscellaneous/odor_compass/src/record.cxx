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
    hsize_t data_dims[3];   // dataset dimensions
    float* data_pointer;
    int* int_pointer;
    
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

    char ds_name[100];

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
    dataset_id = H5Dcreate2(group_id, "wind_p", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_wind.at(idx).wind_p[0]), 3*sizeof(float));
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
    data_dims[1] = FOC_NUM_SENSORS;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "mox_reading", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_raw.at(idx).mox_reading[0]), FOC_NUM_SENSORS*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer);   // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it.
    status = H5Sclose(dataspace_id); // Terminate access to the data space.
    free(data_pointer); // free space
    
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "position", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_raw.at(idx).position[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer);   // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it.
    status = H5Sclose(dataspace_id); // Terminate access to the data space.
    free(data_pointer); // free space

    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "attitude", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_raw.at(idx).attitude[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer);   // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it.
    status = H5Sclose(dataspace_id); // Terminate access to the data space.
    free(data_pointer); // free space

#if 0
    // save data_denoise
    data_dims[0] = foc.data_denoise.size();
    data_dims[1] = FOC_NUM_SENSORS;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_denoise", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_denoise.at(idx).reading[0]), FOC_NUM_SENSORS*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data 
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
#endif

    // save data_interp
    data_dims[0] = foc.data_interp[0].size(); // data_interp[0 ~ FOC_NUM_SENSORS-1] have the same size
    data_dims[1] = FOC_NUM_SENSORS;
    dataspace_id = H5Screate_simple(2, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "mox_interp", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int i = 0; i < data_dims[0]; i++)    // prepare data
        for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
            data_pointer[i*FOC_NUM_SENSORS+idx] = foc.data_interp[idx].at(i);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_wvs
    data_dims[0] = foc.data_wvs_idx.at(1)*FOC_WT_LEVELS; // data_wvs[0 ~ FOC_NUM_SENSORS-1] have the same size
    dataspace_id = H5Screate_simple(1, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "wvs", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*sizeof(*data_pointer));
    for (int i = 0; i < data_dims[0]; i++) {    // prepare data
        data_pointer[i] = foc.data_wvs[i];
    }
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    // save data_wvs_idx
    data_dims[0] = foc.data_wvs_idx.size();
    dataspace_id = H5Screate_simple(1, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "wvs_index", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    int_pointer = (int*)malloc(data_dims[0]*sizeof(*int_pointer));
    for (int i = 0; i < data_dims[0]; i++)    // prepare data
        int_pointer[i] = foc.data_wvs_idx.at(i);
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, int_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(int_pointer); // free space

    // save data_wt_out
    data_dims[0] = foc.data_wt_out[0][0].size(); // data_wt_out[0 ~ FOC_NUM_SENSORS-1][0 ~ FOC_WT_LEVELS-1] have the same size
    data_dims[1] = FOC_WT_LEVELS;
    data_dims[2] = FOC_NUM_SENSORS;
    dataspace_id = H5Screate_simple(3, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "wt_out", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*data_dims[2]*sizeof(*data_pointer));
    for (int i = 0; i < data_dims[0]; i++)    // prepare data
        for (int j = 0; j < data_dims[1]; j++)
            for (int idx = 0; idx < data_dims[2]; idx++)
                data_pointer[i*FOC_WT_LEVELS*FOC_NUM_SENSORS+j*FOC_NUM_SENSORS+idx] = foc.data_wt_out[idx][j].at(i);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space 

    // save data_modmax
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        for (int level = 0; level < FOC_WT_LEVELS; level++) {
            data_dims[0] = foc.data_modmax[idx][level][0].size();
            data_dims[1] = 3; // t, value, level
            dataspace_id = H5Screate_simple(2, data_dims, NULL);
            snprintf(ds_name, sizeof(ds_name), "wt_maxima_s%d_l%d", idx, level);
            dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
            data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
            for (int i = 0; i < data_dims[0]; i++) {    // prepare data
                data_pointer[i*3+0] = foc.data_modmax[idx][level][0].at(i).t;
                data_pointer[i*3+1] = foc.data_modmax[idx][level][0].at(i).value;
                data_pointer[i*3+2] = foc.data_modmax[idx][level][0].at(i).level;
            }
            status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
            status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
            status = H5Sclose(dataspace_id); // Terminate access to the data space. 
            free(data_pointer); // free space

            data_dims[0] = foc.data_modmax[idx][level][1].size();
            data_dims[1] = 3; // t, value, level
            dataspace_id = H5Screate_simple(2, data_dims, NULL);
            snprintf(ds_name, sizeof(ds_name), "wt_minima_s%d_l%d", idx, level);
            dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
            data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
            for (int i = 0; i < data_dims[0]; i++) {    // prepare data
                data_pointer[i*3+0] = foc.data_modmax[idx][level][1].at(i).t;
                data_pointer[i*3+1] = foc.data_modmax[idx][level][1].at(i).value;
                data_pointer[i*3+2] = foc.data_modmax[idx][level][1].at(i).level;
            }
            status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
            status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
            status = H5Sclose(dataspace_id); // Terminate access to the data space. 
            free(data_pointer); // free space
        }
    }



    // save data_maxline
    for (int idx_s = 0; idx_s < FOC_NUM_SENSORS; idx_s++) {
        for (int sign = 0; sign < 2; sign++) {
            data_dims[0] = foc.data_maxline[idx_s][sign].size();
            data_dims[1] = 1; // levels
            dataspace_id = H5Screate_simple(1, data_dims, NULL);
            if (sign == 0)
                snprintf(ds_name, sizeof(ds_name), "wt_maxline_levels_s%d", idx_s);
            else
                snprintf(ds_name, sizeof(ds_name), "wt_minline_levels_s%d", idx_s);
            dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
            int_pointer = (int*)malloc(data_dims[0]*sizeof(*int_pointer));
            for (int i = 0; i < data_dims[0]; i++) {    // prepare data
                int_pointer[i] = foc.data_maxline[idx_s][sign].at(i).levels;
            }
            status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, int_pointer); // write data
            status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
            status = H5Sclose(dataspace_id); // Terminate access to the data space. 
            free(int_pointer); // free space

            data_dims[1] = FOC_WT_LEVELS; // t[FOC_WT_LEVELS]
            dataspace_id = H5Screate_simple(2, data_dims, NULL);
            if (sign == 0)
                snprintf(ds_name, sizeof(ds_name), "wt_maxline_t_s%d", idx_s);
            else
                snprintf(ds_name, sizeof(ds_name), "wt_minline_t_s%d", idx_s);
            dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
            data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
            for (int i = 0; i < data_dims[0]; i++) // prepare data
                for (int j = 0; j < data_dims[1]; j++)
                    data_pointer[i*data_dims[1]+j] = foc.data_maxline[idx_s][sign].at(i).t[j];
            status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
            status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
            status = H5Sclose(dataspace_id); // Terminate access to the data space. 
            free(data_pointer); // free space

            data_dims[1] = FOC_WT_LEVELS; // t[FOC_WT_LEVELS]
            dataspace_id = H5Screate_simple(2, data_dims, NULL);
            if (sign == 0)
                snprintf(ds_name, sizeof(ds_name), "wt_maxline_value_s%d", idx_s);
            else
                snprintf(ds_name, sizeof(ds_name), "wt_minline_value_s%d", idx_s);
            dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
            data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
            for (int i = 0; i < data_dims[0]; i++) {    // prepare data
                memcpy(&data_pointer[i*data_dims[1]], foc.data_maxline[idx_s][sign].at(i).value, data_dims[1]*sizeof(float));
            }
            status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
            status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
            status = H5Sclose(dataspace_id); // Terminate access to the data space. 
            free(data_pointer); // free space
        }
    }

    // data_feature
    data_dims[0] = foc.data_feature.size();
    data_dims[1] = 1;
    dataspace_id = H5Screate_simple(1, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "feature_type", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    int_pointer = (int*)malloc(data_dims[0]*sizeof(*int_pointer));
    for (int i = 0; i < data_dims[0]; i++) {    // prepare data
        int_pointer[i] = foc.data_feature.at(i).type;
    }
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, int_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(int_pointer); // free space

    data_dims[0] = foc.data_feature.size();
    data_dims[1] = FOC_NUM_SENSORS;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "feature_idx_ml", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
    int_pointer = (int*)malloc(data_dims[0]*data_dims[1]*sizeof(*int_pointer));
    for (int i = 0; i < data_dims[0]; i++) {    // prepare data
        memcpy(&int_pointer[i*FOC_NUM_SENSORS], foc.data_feature.at(i).idx_ml, FOC_NUM_SENSORS*sizeof(int));
    }
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, int_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(int_pointer); // free space


    // save data_est
    data_dims[0] = foc.data_est.size();
    data_dims[1] = 3;
    // save data_est.t 
    dataspace_id = H5Screate_simple(1, data_dims, NULL); 
    dataset_id = H5Dcreate2(group_id, "est_t", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        data_pointer[idx] = foc.data_est.at(idx).t;
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
        memcpy(&(data_pointer[idx*3]), &(foc.data_est.at(idx).pos[0]), 3*sizeof(float));
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
        memcpy(&(data_pointer[idx*3]), &(foc.data_est.at(idx).wind_p[0]), 3*sizeof(float));
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
        memcpy(&(data_pointer[idx*3]), &(foc.data_est.at(idx).wind[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    // save data_est.direction
    data_dims[0] = foc.data_est.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "est_direction", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(foc.data_est.at(idx).direction[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    // save data_est.std
    data_dims[0] = foc.data_est.size();
    data_dims[1] = FOC_NUM_SENSORS;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "est_std", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_est.at(idx).std[0]), FOC_NUM_SENSORS*sizeof(float));
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
        memcpy(&(data_pointer[idx]), &(foc.data_est.at(idx).clustering), sizeof(float));
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
        memcpy(&(data_pointer[idx]), &(foc.data_est.at(idx).dt), sizeof(float));
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

// Debug info
#if 1
    /* Create a group named "/Debug" in the file */
    group_id = H5Gcreate2(file_id, "/Debug", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // save particels
    extern std::vector<FOC_Particle_t> particles;
    data_dims[0] = particles.size();
    data_dims[1] = 3;
    dataspace_id = H5Screate_simple(2, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "particles_pos_r", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        memcpy(&(data_pointer[idx*3]), &(particles.at(idx).pos_r[0]), 3*sizeof(float));
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space
    
    dataspace_id = H5Screate_simple(1, data_dims, NULL);
    dataset_id = H5Dcreate2(group_id, "particles_weight", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
    data_pointer = (float*)malloc(data_dims[0]*sizeof(*data_pointer));
    for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
        data_pointer[idx] = particles.at(idx).weight;
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
    status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
    status = H5Sclose(dataspace_id); // Terminate access to the data space. 
    free(data_pointer); // free space

    for (int i = 0; i < particles.size(); i++) {
        data_dims[0] = particles.at(i).plume->size();
        data_dims[1] = 3;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "particle_%d_plume_pos", i);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*3]), &(particles.at(i).plume->at(idx).pos[0]), 3*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
    }
#endif

#if 0
    // save data_smooth
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
    for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP+1; j++) {
        data_dims[0] = foc.data_smooth[i*(FOC_DIFF_LAYERS_PER_GROUP+1)+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL); 
        snprintf(ds_name, sizeof(ds_name), "mox_smooth_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_smooth[i*(FOC_DIFF_LAYERS_PER_GROUP+1)+j].at(idx).reading[0]), FOC_NUM_SENSORS*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
    }

    // save data_diff
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
    for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++) {
        data_dims[0] = foc.data_diff[i*FOC_DIFF_LAYERS_PER_GROUP+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "mox_diff_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_diff[i*FOC_DIFF_LAYERS_PER_GROUP+j].at(idx).reading[0]), FOC_NUM_SENSORS*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
    }
#endif

#if 0
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
#endif

#if 0
    // save data_edge
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
    for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++) {
        data_dims[0] = foc.data_edge_max[i*FOC_DIFF_LAYERS_PER_GROUP+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "mox_edge_max_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_edge_max[i*FOC_DIFF_LAYERS_PER_GROUP+j].at(idx).reading[0]), FOC_NUM_SENSORS*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
    
        data_dims[0] = foc.data_edge_min[i*FOC_DIFF_LAYERS_PER_GROUP+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "mox_edge_min_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_edge_min[i*FOC_DIFF_LAYERS_PER_GROUP+j].at(idx).reading[0]), FOC_NUM_SENSORS*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
    }

    // save data_cp 
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
    for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++) {
        data_dims[0] = foc.data_cp_max[i*FOC_DIFF_LAYERS_PER_GROUP+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "mox_cp_max_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
        int_pointer = (int*)malloc(data_dims[0]*data_dims[1]*sizeof(*int_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(int_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_cp_max[i*FOC_DIFF_LAYERS_PER_GROUP+j].at(idx).index[0]), FOC_NUM_SENSORS*sizeof(int));
        status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, int_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(int_pointer); // free space

        data_dims[0] = foc.data_cp_min[i*FOC_DIFF_LAYERS_PER_GROUP+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "mox_cp_min_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
        int_pointer = (int*)malloc(data_dims[0]*data_dims[1]*sizeof(*int_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(int_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_cp_min[i*FOC_DIFF_LAYERS_PER_GROUP+j].at(idx).index[0]), FOC_NUM_SENSORS*sizeof(int));
        status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, int_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(int_pointer); // free space
    }

    // save data_std
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
    for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++) {
        data_dims[0] = foc.data_std[i*FOC_DIFF_LAYERS_PER_GROUP+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "mox_std_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set 
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_std[i*FOC_DIFF_LAYERS_PER_GROUP+j].at(idx).std[0]), FOC_NUM_SENSORS*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
    }

    // save data_tdoa
    for (int i = 0; i < FOC_DIFF_GROUPS; i++)
    for (int j = 0; j < FOC_DIFF_LAYERS_PER_GROUP; j++) {
        data_dims[0] = foc.data_tdoa[i*FOC_DIFF_LAYERS_PER_GROUP+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "mox_abs_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_tdoa[i*FOC_DIFF_LAYERS_PER_GROUP+j].at(idx).abs[0]), FOC_NUM_SENSORS*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
    
        data_dims[0] = foc.data_tdoa[i*FOC_DIFF_LAYERS_PER_GROUP+j].size();
        data_dims[1] = FOC_NUM_SENSORS;
        dataspace_id = H5Screate_simple(2, data_dims, NULL);
        snprintf(ds_name, sizeof(ds_name), "mox_toa_group_%d_layer_%d", i+1, j+1);
        dataset_id = H5Dcreate2(group_id, ds_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // create data set
        data_pointer = (float*)malloc(data_dims[0]*data_dims[1]*sizeof(*data_pointer));
        for (int idx = 0; idx < data_dims[0]; idx++)    // prepare data
            memcpy(&(data_pointer[idx*FOC_NUM_SENSORS]), &(foc.data_tdoa[i*FOC_DIFF_LAYERS_PER_GROUP+j].at(idx).toa[0]), FOC_NUM_SENSORS*sizeof(float));
        status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data_pointer); // write data
        status = H5Dclose(dataset_id); // End access to the dataset and release resources used by it. 
        status = H5Sclose(dataspace_id); // Terminate access to the data space. 
        free(data_pointer); // free space
    }

    
#endif

    /* Close group "/FOC" */
    status = H5Gclose(group_id);

    /* Terminate access to the file. */
    status = H5Fclose(file_id);
}
