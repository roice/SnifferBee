/*
 * This file implements a UKF and a Gaussian smoother to suppress noise
 *
 * Author:
 *      Roice Luo
 * Date:
 *      2016.09.12
 */

#include <cmath>

#include "mocap/signal_model_ukf.hpp"
#include "mocap/mocap_noise_reduction.h"

#include <Matrix.hpp>
#include <UnscentedKalmanFilter.hpp>

#define MOCAP_NR_NUM_VECTORS   4

typedef float T;

double previous_time = -1;

float signal_reading_var_process_noise = pow(3.0, 2);
float signal_reading_var_measurement_noise = pow(0.0005*50.0, 2); // m/s, 0.001 m, 50 Hz
void* signal_reading_filter[MOCAP_NR_NUM_VECTORS][3]; // ukf filters, enu
void* signal_reading_state[MOCAP_NR_NUM_VECTORS][3]; // state vectors
void* signal_reading_sys[MOCAP_NR_NUM_VECTORS][3]; // system model
void* signal_reading_mm[MOCAP_NR_NUM_VECTORS][3]; // measurement model
void* signal_reading_z[MOCAP_NR_NUM_VECTORS][3]; // measurement

void mocap_noise_reduction_ukf_init(int index)
{
    if (index >= MOCAP_NR_NUM_VECTORS) return;

    float dt = 1.0/50.0;    // 50 Hz
    
    Signal_Model::State<T>** x = (Signal_Model::State<T>**)signal_reading_state[index];
    Signal_Model::SystemModel<T>** sys = (Signal_Model::SystemModel<T>**)
        signal_reading_sys[index];
    Signal_Model::MeasurementModel<T>** mm = (Signal_Model::MeasurementModel<T>**) signal_reading_mm[index];
    Signal_Model::Measurement<T>** z = (Signal_Model::Measurement<T>**)
        signal_reading_z[index];
    Kalman::UnscentedKalmanFilter<Signal_Model::State<T>>** ukf = 
        (Kalman::UnscentedKalmanFilter<Signal_Model::State<T>>**)
            signal_reading_filter[index];
    Kalman::Covariance<Signal_Model::State<T>> Q;
    Q << signal_reading_var_process_noise*pow(dt, 4)/4.0, 
         signal_reading_var_process_noise*pow(dt, 3)/2.0,
         signal_reading_var_process_noise*pow(dt, 3)/2.0, 
         signal_reading_var_process_noise*pow(dt, 2);
    Kalman::Covariance<Signal_Model::Measurement<T>> R;
    R << signal_reading_var_measurement_noise;
    for (int i = 0; i < 3; i++) {
        x[i] = new Signal_Model::State<T>; // state variables
        x[i]->setZero();
        sys[i] = new Signal_Model::SystemModel<T>; // state transition function
        sys[i]->setCovariance(Q); // process noise matrix
        mm[i] = new Signal_Model::MeasurementModel<T>; // measurement function
        mm[i]->setCovariance(R); // measurement noise matrix
        z[i] = new Signal_Model::Measurement<T>; // measurement 
        ukf[i] = new Kalman::UnscentedKalmanFilter<Signal_Model::State<T>>(1,2,1); // filters
        ukf[i]->init(*x[i]); // init filter
    }
}

Mocap_3D_Vector_t mocap_noise_reduction_ukf_update(int index, Mocap_3D_Vector_t& new_in)
{
    Mocap_3D_Vector_t out = {0};
    
    if (index >= MOCAP_NR_NUM_VECTORS) return out;

    float dt = 1.0/50.0;    // 50 Hz

    Signal_Model::State<T>** x = (Signal_Model::State<T>**)signal_reading_state[index];
    Signal_Model::SystemModel<T>** sys = (Signal_Model::SystemModel<T>**)
        signal_reading_sys[index];
    Signal_Model::MeasurementModel<T>** mm = (Signal_Model::MeasurementModel<T>**)
        signal_reading_mm[index];
    Signal_Model::Measurement<T>** z = (Signal_Model::Measurement<T>**)
        signal_reading_z[index];
    Kalman::UnscentedKalmanFilter<Signal_Model::State<T>>** ukf = 
        (Kalman::UnscentedKalmanFilter<Signal_Model::State<T>>**)
            signal_reading_filter[index];
    Signal_Model::State<T> x_ukf[3];
    Kalman::Covariance<Signal_Model::State<T>> Q;
    Q << signal_reading_var_process_noise*pow(dt, 4)/4.0, 
         signal_reading_var_process_noise*pow(dt, 3)/2.0,
         signal_reading_var_process_noise*pow(dt, 3)/2.0, 
         signal_reading_var_process_noise*pow(dt, 2);
    for (int i = 0; i < 3; i++) {
        sys[i]->dt = dt;
        x_ukf[i] = ukf[i]->predict(*(sys[i]));
        z[i]->z() = new_in.v[i];
        x_ukf[i] = ukf[i]->update(*mm[i], *z[i]);
    }
 
    for (int i = 0; i < 3; i++) {
        out.v[i] = x_ukf[i].reading();
    }
    return out;
}
