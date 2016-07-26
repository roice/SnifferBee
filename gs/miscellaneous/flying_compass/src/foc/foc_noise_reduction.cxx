/*
 * This file implements a UKF and a Gaussian smoother to suppress noise
 *
 * Author:
 *      Roice Luo
 * Date:
 *      2016.06.25
 */

#include <cmath>

#include "foc/flying_odor_compass.h"
#include "foc/mox_model_ukf.hpp"

#include <Matrix.hpp>
#include <UnscentedKalmanFilter.hpp>

typedef float T;

double previous_time = -1;

float sensor_reading_var_process_noise;
float sensor_reading_var_measurement_noise;
void* sensor_reading_filter[FOC_NUM_SENSORS]; // ukf filters
void* sensor_reading_state[FOC_NUM_SENSORS]; // state vectors
void* sensor_reading_sys[FOC_NUM_SENSORS]; // system model
void* sensor_reading_mm[FOC_NUM_SENSORS]; // measurement model
void* sensor_reading_z[FOC_NUM_SENSORS]; // measurement

void foc_noise_reduction_ukf_init(void)
{
    sensor_reading_var_process_noise = pow(0.85, 2);
    sensor_reading_var_measurement_noise = 0.05*0.05;
    MOX_Sensor::State<T>** x = (MOX_Sensor::State<T>**)sensor_reading_state;
    MOX_Sensor::SystemModel<T>** sys = (MOX_Sensor::SystemModel<T>**)
        sensor_reading_sys;
    MOX_Sensor::MeasurementModel<T>** mm = (MOX_Sensor::MeasurementModel<T>**)
        sensor_reading_mm;
    MOX_Sensor::Measurement<T>** z = (MOX_Sensor::Measurement<T>**)
        sensor_reading_z;
    Kalman::UnscentedKalmanFilter<MOX_Sensor::State<T>>** ukf = 
        (Kalman::UnscentedKalmanFilter<MOX_Sensor::State<T>>**)
            sensor_reading_filter;
    Kalman::Covariance<MOX_Sensor::State<T>> Q;
    Q << sensor_reading_var_process_noise*pow(1.0/FOC_MOX_DAQ_FREQ, 4)/4.0, 
         sensor_reading_var_process_noise*pow(1.0/FOC_MOX_DAQ_FREQ, 3)/2.0,
         sensor_reading_var_process_noise*pow(1.0/FOC_MOX_DAQ_FREQ, 3)/2.0, 
         sensor_reading_var_process_noise*pow(1.0/FOC_MOX_DAQ_FREQ, 2);
    Kalman::Covariance<MOX_Sensor::Measurement<T>> R;
    R << sensor_reading_var_measurement_noise;
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        x[i] = new MOX_Sensor::State<T>; // state variables
        x[i]->setZero();
        sys[i] = new MOX_Sensor::SystemModel<T>; // state transition function
        sys[i]->setCovariance(Q); // process noise matrix
        mm[i] = new MOX_Sensor::MeasurementModel<T>; // measurement function
        mm[i]->setCovariance(R); // measurement noise matrix
        z[i] = new MOX_Sensor::Measurement<T>; // measurement 
        ukf[i] = new Kalman::UnscentedKalmanFilter<MOX_Sensor::State<T>>(1,2,1); // filters
        ukf[i]->init(*x[i]); // init filter
    }
}

FOC_Reading_t foc_noise_reduction_ukf_update(FOC_Input_t& new_in)
{
    float dt;
    if (previous_time > 0)
        dt = new_in.time - previous_time;
    else
        dt = 0;
    previous_time = new_in.time;
    MOX_Sensor::State<T>** x = (MOX_Sensor::State<T>**)sensor_reading_state;
    MOX_Sensor::SystemModel<T>** sys = (MOX_Sensor::SystemModel<T>**)
        sensor_reading_sys;
    MOX_Sensor::MeasurementModel<T>** mm = (MOX_Sensor::MeasurementModel<T>**)
        sensor_reading_mm;
    MOX_Sensor::Measurement<T>** z = (MOX_Sensor::Measurement<T>**)
        sensor_reading_z;
    Kalman::UnscentedKalmanFilter<MOX_Sensor::State<T>>** ukf = 
        (Kalman::UnscentedKalmanFilter<MOX_Sensor::State<T>>**)
            sensor_reading_filter;
    MOX_Sensor::State<T> x_ukf[FOC_NUM_SENSORS];
    Kalman::Covariance<MOX_Sensor::State<T>> Q;
    Q << sensor_reading_var_process_noise*pow(dt, 4)/4.0, 
         sensor_reading_var_process_noise*pow(dt, 3)/2.0,
         sensor_reading_var_process_noise*pow(dt, 3)/2.0, 
         sensor_reading_var_process_noise*pow(dt, 2);
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        sys[i]->dt = dt;
        x_ukf[i] = ukf[i]->predict(*(sys[i]));
        z[i]->z() = new_in.mox_reading[i];
        x_ukf[i] = ukf[i]->update(*mm[i], *z[i]);
    }

    FOC_Reading_t out;
    out.time = new_in.time;
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        out.reading[i] = x_ukf[i].reading();
    }
    return out;
}

static void CreateKernel(float sigma, int order, float* gKernel, int nWindowSize) {
    
    int nCenter = nWindowSize/2;
    double Value, Sum;
   
    // default no derivative
    gKernel[nCenter] = 1.0;
    Sum = 1.0;
    for (int i = 1; i <= nCenter; i++) {
        Value = exp(-0.5*i*i/(sigma*sigma));// /(sqrt(2*M_PI)*sigma);
        gKernel[nCenter+i] = Value;
        gKernel[nCenter-i] = Value;
        Sum += 2.0*Value;
    }
    // normalize
    for (int i = 0; i < nWindowSize; i++)
        gKernel[i] = gKernel[i]/Sum;
    
    if (order == 1) { // 1st derivative
        gKernel[nCenter] = 0.0;
        for (int i = 1; i <= nCenter; i++) {
            Value = -i/(sigma*sigma)*gKernel[nCenter+i];
            gKernel[nCenter+i] = -Value;
            gKernel[nCenter-i] = Value;
        }
    }
    if (order == 2) { // 2nd derivative
        gKernel[nCenter] *= -1.0/(sigma*sigma);
        for (int i = 1; i <= nCenter; i++) {
            Value = (i*i/(sigma*sigma) - 1.0)*gKernel[nCenter+i]/(sigma*sigma);
            gKernel[nCenter+i] = Value;
            gKernel[nCenter-i] = Value;
        }
    }
    if (order == 3) { // 3rd derivative
        gKernel[nCenter] = 0.0;
        for (int i = 1; i <= nCenter; i++) {
            Value = (3.0 - i*i/(sigma*sigma)) * i * gKernel[nCenter+i] / pow(sigma,4);
            gKernel[nCenter+i] = -Value;
            gKernel[nCenter-i] = Value;
        }
    }
}

bool foc_noise_reduction_gaussian_filter(std::vector<FOC_Reading_t>* input,
        std::vector<FOC_Reading_t>* output, int order, float backtrace_time)
{
    // Create gaussian kernel
    float sigma = 1.0;//(1.0/FOC_MOX_DAQ_FREQ);
    int nWindowSize = 1 + 2*ceil(4*sigma*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR); // length of gaussian smoother;
    if (backtrace_time*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR <= nWindowSize)
        return false;
    float* gKernel; // gaussian kernel
    if ((gKernel = (float*)malloc(nWindowSize*sizeof(float)))==NULL)  
    {  
        printf("malloc memory for gKernel failed!");  
        exit(0);  
    }
    CreateKernel(sigma, order, gKernel, nWindowSize);

    // Smooth
    // |------|------|------|  3*backtrace_time
    //        b      e
    int nLen = nWindowSize/2;
    double DotMul[FOC_NUM_SENSORS], WeightSum[FOC_NUM_SENSORS];
    FOC_Reading_t new_out;
    if (input->size() >= backtrace_time*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR) {
        output->clear(); // remove all data
        for(int i = input->size() - backtrace_time*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR; i < input->size(); i++)  {
            for (int sensor_index = 0; sensor_index < FOC_NUM_SENSORS; sensor_index++) {
                DotMul[sensor_index] = 0;
                WeightSum[sensor_index] = 0;
                for (int j = (-nLen); j <= nLen; j++)
                {
                    if ((i+j) >= 0 && (i+j) < input->size())
                        DotMul[sensor_index] += (double)(input->at(i+j).reading[sensor_index] * gKernel[nLen+j]); 
                    else if ((i+j) < 0)
                        DotMul[sensor_index] += (double)(input->at(-i-j).reading[sensor_index] * gKernel[nLen+j]);
                    else if ((i+j) >= input->size())
                        DotMul[sensor_index] += (double)(input->at(2*input->size()-(i+j)-1).reading[sensor_index] * gKernel[nLen+j]);
                    WeightSum[sensor_index] += gKernel[nLen+j];
                }
                new_out.reading[sensor_index] = DotMul[sensor_index]/WeightSum[sensor_index]; 
            }
            new_out.time = input->at(i).time;
            output->push_back(new_out);
        }
        return true;
    }
    else
        return false;
}
