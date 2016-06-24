/*
 * Flying Odor Compass
 *
 * This technique digs information from three MOX gas sensors which are
 * equally spaced under propellers of a quadrotor
 *
 * Author:
 *      Roice Luo
 * Date:
 *      2016.06.17
 */
#include <cmath>

#include "foc/flying_odor_compass.h"
#include "foc/noise_suppression.hpp"

#include <Matrix.hpp>
#include <UnscentedKalmanFilter.hpp>

typedef float T;

/* The sensors are placed at vertices of a regular polygon
 * Example: 3 sensors
 *
 *              sensor_1   --------
 *                /  \        R
 *               /    \    --------
 *              /      \
 *      sensor_2 ------ sensor_3
 */

Flying_Odor_Compass::Flying_Odor_Compass(void)
{
    // data
    foc_input.reserve(FOC_RECORD_LEN);
    foc_state.reserve(FOC_RECORD_LEN);
/* init UKF filtering */
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

void Flying_Odor_Compass::update(FOC_Input_t& new_in)
{
/* Step 0: Pre-processing */
    // get dtime (in case of data missing)
    float dt;
    if (foc_input.size() > 0)
        dt = new_in.time - foc_input.back().time;
    else
        dt = 1.0/FOC_MOX_DAQ_FREQ;
/* Step 1: UKF filtering */
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
/* Step : Save record */
    foc_input.push_back(new_in);
    FOC_State_t new_state;
    new_state.time = new_in.time;
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        new_state.smoothed_mox_reading[i] = x_ukf[i].reading();
    }
    foc_state.push_back(new_state);
}
