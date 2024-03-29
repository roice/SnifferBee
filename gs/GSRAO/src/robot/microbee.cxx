/*
 * MicroBee Robot
 *         
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-05-23 create this file (robot_control.cxx)
 *       2016-05-25 change this file to microbee.cxx
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <cmath>
#include <math.h>
/* thread */
#include <pthread.h>
/* GSRAO */
#include "mocap/packet_client.h"
#include "robot/microbee.h"
#include "robot/robot.h"
#include "io/serial.h"
#include "common/vector_rotation.h"
#include "GSRAO_Config.h"
#include "GSRAO_thread_comm.h"
/* CBLAS */
#include "cblas.h"
/* Liquid */
#include "liquid.h"

#ifndef MICROBEE_LANDING_THRESHOLD
#define MICROBEE_LANDING_THRESHOLD 0.06 // shutdown when bee lands neer ground
#endif

typedef struct {
    bool* exit_thread;
    int index_robot; // 0/1/2/3, to indicate which robot this thread is controlling
} MicroBee_Control_Thread_Args_t;

static pthread_t microbee_control_thread_handle[4]; // 4 robots max
static bool exit_microbee_control_thread = false;
MicroBee_Control_Thread_Args_t microbee_control_thread_args[4]; // 4 robots max

static pthread_t microbee_state_thread_handle;
static bool exit_microbee_state_thread = false;

static MicroBee_t microbee[4] = {0}; // 4 robots max

static float microbee_pos_fix[4][3] = {{0, 0, 0}, {0}, {0}, {0}}; // position fix to align rigid body and microbee centers

static bool microbee_manual_control[4] = {false, false, false, false};

#ifdef MICROBEE_DENOISE_BEFORE_CONTROL
#define DENOISE_H_LEN   (6)
static firfilt_rrrf f_denoise[4][3]; // pos vel acc att, xyz
static float h_denoise[4][3][DENOISE_H_LEN];
#endif

static void* microbee_control_loop(void*);
static void* microbee_state_loop(void*);
static void microbee_pos_control(float, int);

static void CreateGaussianKernel(float sigma, float* gKernel, int nWindowSize);

/*-------- MicroBee State Refresh --------*/

/* microbee state refresh init */
bool microbee_state_init(void)
{
    // clear microbee states
    for (int i = 0; i < 4; i++) // 4 robots max
        memset(&(microbee[i].state), 0, sizeof(microbee[i].state));

    float gain;
#ifdef MICROBEE_DENOISE_BEFORE_CONTROL
    // denoise, calculate response
    for (int i = 0; i < 4; i++) // pos vel acc att
        for (int j = 0; j < 3; j++) // xyz
            CreateGaussianKernel(20.0, h_denoise[i][j], DENOISE_H_LEN);
    // denoise, calculate filter gain and tune them to the same gain
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {// xyz
            gain = 0;
            for (int idx = 0; idx < DENOISE_H_LEN; idx++)
                gain += h_denoise[i][j][idx];
            for (int idx = 0; idx < DENOISE_H_LEN; idx++)
                h_denoise[i][j][idx] /= gain;
        }
    // denoise, create filter from response
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            f_denoise[i][j] = firfilt_rrrf_create(h_denoise[i][j], DENOISE_H_LEN);
#endif

    /* create state refresh loop */
    exit_microbee_state_thread = false;
    if (pthread_create(&microbee_state_thread_handle, NULL, &microbee_state_loop, (void*)&exit_microbee_state_thread) != 0)
        return false;

    return true;
}

/* close microbee state refresh loop */
void microbee_state_close(void)
{
    if (!exit_microbee_state_thread) // to avoid close twice
    {
        // exit microbee state refresh thread
        exit_microbee_state_thread = true;
        pthread_join(microbee_state_thread_handle, NULL);
        printf("microbee state thread terminated\n");
    }
}

MicroBee_t* microbee_get_states(void)
{
    return microbee;
}

static void* microbee_state_loop(void* exit)
{
    struct timespec req, rem, time;
    double current_time;
    int link_check_count = 0; // counter for link state check

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 100000000L; // 100 ms

    while (!*((bool*)exit))
    {
        // check link state
        if (++link_check_count >= 5) // 0.5 sec check once
        {
            clock_gettime(CLOCK_REALTIME, &time);
            current_time = time.tv_sec + time.tv_nsec/1.0e9;
            for (int i = 0; i < 4; i++) // 4 robots max
            {
                if (current_time - microbee[i].time > 0.5)
                    microbee[i].state.linked = false;
                else
                    microbee[i].state.linked = true;
            }
            link_check_count = 0;
        }

        // 10 Hz
        nanosleep(&req, &rem); // 100 ms
    }
}

/*-------- MicroBee Control ---------*/

/* microbee control init */
bool microbee_control_init(int num_of_mbs)
{
    /* init thread args */ 
    for (int i = 0; i < 4; i++) // 4 robots max
    {
        microbee_control_thread_args[i].exit_thread = &exit_microbee_control_thread;
        microbee_control_thread_args[i].index_robot = i;
    }

    /* create trajectory control loop */
    exit_microbee_control_thread = false; // control robots to stop (take landing action) at the same time
    for (int i = 0; i < num_of_mbs; i++)
    {
        if (pthread_create(&(microbee_control_thread_handle[i]), NULL, &microbee_control_loop, (void*)&(microbee_control_thread_args[i])) != 0)
        {
            return false;
        }
    }

    return true;
}

/* close microbee control loop */
void microbee_control_close(int num_of_mbs)
{
    if (num_of_mbs < 1 || num_of_mbs > 4)
    {
        printf("MicroBee control close: num_of_mbs not in range 1-4, close failed.\n");
        return;
    }

    if (!exit_microbee_control_thread) // to avoid close twice
    {
        // exit microbee control thread
        exit_microbee_control_thread = true;
        for (int i = 0; i < num_of_mbs; i++) // 4 robots max
            pthread_join(microbee_control_thread_handle[i], NULL);
        printf("microbee control thread terminated\n");
    }
}

static void* microbee_control_loop(void* args)
{
    bool* exit = ((MicroBee_Control_Thread_Args_t*)args)->exit_thread;
    int idx_robot = ((MicroBee_Control_Thread_Args_t*)args)->index_robot;
    struct timespec req, rem, time;
    double previous_time, current_time;
    float dtime;
    MocapData_t* data  = mocap_get_data(); // get mocap data

    // init previous_time, current_time, dtime
    clock_gettime(CLOCK_REALTIME, &time);
    current_time = time.tv_sec + time.tv_nsec/1.0e9;
    previous_time = current_time;
    dtime = 0;

    Robot_Ref_State_t* robot_ref = robot_get_ref_state();

/* Step 1: Take off */ 
    // arm, throttle min, yaw max
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
#if 1
    rc_data[idx_robot].throttle = 1000;
    rc_data[idx_robot].roll = 1500;
    rc_data[idx_robot].pitch = 1500;
    rc_data[idx_robot].yaw = 2000;
    req.tv_sec = 0;
    req.tv_nsec = 200000000L; // 200 ms
    while (!*((bool*)exit) && !microbee[idx_robot].state.armed)
        nanosleep(&req, &rem); // 1 s
    // recover yaw to middle
    rc_data[idx_robot].yaw = 1500;
#endif
/* Step 2: Fly */
    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 20000000L; // 20 ms
    while (!*((bool*)exit) && microbee[idx_robot].state.armed)
    {
        clock_gettime(CLOCK_REALTIME, &time);
        current_time = time.tv_sec + time.tv_nsec/1.0e9;
        dtime = current_time - previous_time;
        previous_time = current_time;

        // check if at manual control
        if (microbee_manual_control[idx_robot])
            continue;

        // position control update
        //microbee_pos_control(dtime, idx_robot);
        microbee_pos_control(1.0/50.0, idx_robot);

        // 50 Hz
        nanosleep(&req, &rem); // 20 ms
    }

/* Step 3: Land */
    req.tv_sec = 0;
    req.tv_nsec = 20000000L; // 20 ms
    while(data->robot[idx_robot].enu[2]+microbee_pos_fix[idx_robot][2] > MICROBEE_LANDING_THRESHOLD)
    {
        clock_gettime(CLOCK_REALTIME, &time);
        current_time = time.tv_sec + time.tv_nsec/1.0e9;
        dtime = current_time - previous_time;
        previous_time = current_time;

        robot_ref[idx_robot].enu[2] -= 0.5/50.0; // 0.5 m/s descending, 50 Hz control

        // position control update
        microbee_pos_control(dtime, idx_robot);

        // 50 Hz
        nanosleep(&req, &rem); // 20 ms
    }

/* Final Step: shutdown */
    rc_data[idx_robot].throttle = 1000;
    rc_data[idx_robot].roll = 1500;
    rc_data[idx_robot].pitch = 1500;
    rc_data[idx_robot].yaw = 1500;
    req.tv_sec = 0;
    req.tv_nsec = 500000000; // 0.5 s
    nanosleep(&req, &rem);
}

static float constrain(float amt, float low, float high)
{
    if (amt < low)
        return low;
    else if (amt > high)
        return high;
    else
        return amt;
}

static float applyDeadband(float value, float deadband)
{
    if (fabs(value) < deadband) {
        value = 0;
    } else if (value > 0) {
        value -= deadband;
    } else if (value < 0) {
        value += deadband;
    }
    return value;
}

/*
 * TODO:
 *  dt not used
 */
static void microbee_throttle_control_pid(float dt, int robot_index, float AltHold, float EstAlt, float vel_temp, float accZ_temp)
{
    static float errorVelocityI[4] = {0}; // 4 robots max

    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile; 

    /* altitude control, throttle */
    // Altitude P-Controller
    float error = constrain(AltHold - EstAlt, -1.0, 1.0); // -0.5 - 0.5 m boundary
    error = applyDeadband(error, 0.01); // 1 cm deadband, remove small P parameter to reduce noise near zero position
    float setVel = constrain((pidProfile[robot_index].P[PIDALT]*error), -2.0, 2.0); // limit velocity to +/- 2.0 m/s

    //printf("Alt error is %f m, setVel is %f\n", error, setVel);

    // Velocity PID-Controller
    // P
    error = setVel - vel_temp;
    float result = constrain((pidProfile[robot_index].P[PIDVEL]*error), -300, 300); // limit to +/- 300

    //printf("Alt vel P is %f\n", result);

    // I
    errorVelocityI[robot_index] += (pidProfile[robot_index].I[PIDVEL]*error);
    errorVelocityI[robot_index] = constrain(errorVelocityI[robot_index], -700.0, 700.0); // limit to +/- 700
    result += errorVelocityI[robot_index];

    //printf("Alt vel I is %f\n", errorVelocityI[robot_index]);

    // D
    result -= constrain(pidProfile[robot_index].D[PIDVEL]*accZ_temp, -100, 100); // limit

    //printf("Alt adj = %f\n", result);

    // update throttle value
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
    rc_data[robot_index].throttle = constrain(1050 + result, 1050, 1950);
}

static void microbee_roll_pitch_control_pid(float dt, int robot_index, float* pos_ref, float* pos, float* vel, float* acc, float* att)
{
    static float errorPositionI[4][2] = {0}; 

    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile;
    
    // get rc_data
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
    
    // get position error vector in earth coordinate
    float error_en[2]; // error vector in earth coordinate
    for (int i = 0; i < 2; i++)
        error_en[i] = pos_ref[i] - pos[i];

    // Velocity-PID
    float target_vel[2]; // in inertial frame
    float err_pos_vel;
    float result_pos_i[2]; // in inertial frame
    for (int i = 0; i < 2; i++) // 0 for roll, 1 for pitch
    {
        // Position PID-Controller for east(x)/north(y) axis
        target_vel[i] = constrain(pidProfile[robot_index].P[PIDPOS]*error_en[i], -1.0, 1.0); // limit error to +/- 1.0 m/s;
        target_vel[i] = applyDeadband(target_vel[i], 0.01); // 1 cm/s

        // Velocity PID-Controller
        err_pos_vel = target_vel[i]-vel[i];

        // P
        result_pos_i[i] = constrain((pidProfile[robot_index].P[PIDPOSR]*err_pos_vel), -200, 200); // limit to +/- 100
        // I
        errorPositionI[robot_index][i] += (pidProfile[robot_index].I[PIDPOSR]*err_pos_vel);
        errorPositionI[robot_index][i] = constrain(errorPositionI[robot_index][i], -200.0, 200.0); // limit to +/- 200
        result_pos_i[i] += errorPositionI[robot_index][i];
        // D
        result_pos_i[i] -= constrain(pidProfile[robot_index].D[PIDPOSR]*acc[i], -100, 100); // limit
    }
    // transform result_pos from inertial frame to body frame
    float heading_angle = att[2]; // in inertial frame
    float result_pos[2]; // in body frame
    result_pos[0] = std::cos(heading_angle)*result_pos_i[0] + std::sin(heading_angle)*result_pos_i[1];
    result_pos[1] = -std::sin(heading_angle)*result_pos_i[0] + std::cos(heading_angle)*result_pos_i[1];
    // update roll/pitch value 
    rc_data[robot_index].roll = constrain(1500 + result_pos[0], 1000, 2000);
    rc_data[robot_index].pitch = constrain(1500 + result_pos[1], 1000, 2000); 
}

/* ADRC state vector type */
typedef struct {
    float z1;
    float z2;
    float z3;
    float u;
} ADRC_State_t;

static void microbee_yaw_control_adrc(float dt, int robot_index, float heading_ref, float heading)
{
    static ADRC_State_t state[4] = {{0},{0},{0},{0}}; // 4 robots max
    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    adrcProfile_t*   adrcProfile = configs->robot.adrcProfile;

    /* LADRC */
    float leso_err = heading - state[robot_index].z1;
    /*if (leso_err > M_PI)
        leso_err -= 2*M_PI;
    else if (leso_err < -M_PI)
        leso_err += 2*M_PI;
    leso_err = constrain(leso_err, -M_PI, M_PI);*/
    // LESO
    state[robot_index].z1 += dt*(state[robot_index].z2 + 3*adrcProfile[robot_index].w0[ADRCMAG]*leso_err);
    state[robot_index].z2 += dt*(state[robot_index].z3 + 3*pow(adrcProfile[robot_index].w0[ADRCMAG],2)*leso_err + state[robot_index].u);
    state[robot_index].z3 += dt*(pow(adrcProfile[robot_index].w0[ADRCMAG],3)*leso_err);
    // PD
    float pd_err = state[robot_index].z1 - heading_ref;
    if (pd_err > M_PI)
        pd_err -= 2*M_PI;
    else if (pd_err < -M_PI)
        pd_err += 2*M_PI;
    float u0 = adrcProfile[robot_index].kp[ADRCMAG]*pd_err + adrcProfile[robot_index].kd[ADRCMAG]*state[robot_index].z2;
    state[robot_index].u = u0 - state[robot_index].z3;

    // convert u to result
    float result = 0.4*state[robot_index].u;

// for DEBUG
//printf("heading_ref = %f, heading = %f, pd_err = %f, z2 = %f, z3 = %f, result = %f\n", heading_ref, heading, pd_err, state[robot_index].z2, state[robot_index].z3, result);

    // update yaw value
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
    rc_data[robot_index].yaw = constrain(1500 + result, 1050, 1950);
}

static void microbee_wind_estimation_leso(float dt, int robot_index, float* pos, float* vel, float* acc, float* att)
{
    float w0 = 18;
    float m = 0.124; // kg
    float thrust_U0 = 2.90324338;
    float c_x = 0.2;
    float c_y = 0.2;
    float c_z = 0.8;

    static ADRC_State_t state[4][3] = {{0}, {0}, {0}, {0}}; // 4 robots max
    float *z1[3], *z2[3], *z3[3];
    for (int i = 0; i < 3; i++) {
        z1[i] = &(state[robot_index][i].z1);
        z2[i] = &(state[robot_index][i].z2);
        z3[i] = &(state[robot_index][i].z3);
    }

    // result
    float wind_estimated[3];

    // body frame to inertial frame
    float R_BI[9] = {
        std::cos(att[1])*std::cos(att[2]),
        std::sin(att[0])*std::sin(att[1])*std::cos(att[2]) - std::cos(att[0])*std::sin(att[2]),
        std::cos(att[2])*std::sin(att[1])*std::cos(att[0]) + std::sin(att[0])*std::sin(att[2]),
        std::cos(att[1])*std::sin(att[2]),
        std::sin(att[1])*std::sin(att[0])*std::sin(att[2]) + std::cos(att[0])*std::cos(att[2]),
        std::cos(att[0])*std::sin(att[1])*std::sin(att[2]) - std::sin(att[0])*std::cos(att[2]),
        -std::sin(att[1]),
        std::sin(att[0])*std::cos(att[1]),
        std::cos(att[0])*std::cos(att[1]) 
    };
    // inertial frame to body frame
    float R_IB[9] = {
        std::cos(att[2])*std::cos(att[1]),
        std::sin(att[2])*std::cos(att[1]),
        -std::sin(att[1]),
        std::sin(att[0])*std::cos(att[2])*std::sin(att[1]) - std::cos(att[0])*std::sin(att[2]),
        std::sin(att[0])*std::sin(att[2])*std::sin(att[1]) + std::cos(att[0])*std::cos(att[2]),
        std::sin(att[0])*std::cos(att[1]),
        std::cos(att[0])*std::cos(att[2])*std::sin(att[1]) + std::sin(att[0])*std::sin(att[2]),
        std::cos(att[0])*std::sin(att[2])*std::sin(att[1]) - std::sin(att[0])*std::cos(att[2]),
        std::cos(att[0])*std::cos(att[1])
    };
    
    float leso_err[3] = {
        pos[0] - *(z1[0]),
        pos[1] - *(z1[1]),
        pos[2] - *(z1[2]) };
    
    // get motor value and calculate force vector
    float scale_motor_value = 0.0003;
    float thrust_U_B[3] = {
        0., 
        0., 
        (float)(std::pow(microbee[robot_index].motor[0]*scale_motor_value,2)+std::pow(microbee[robot_index].motor[1]*scale_motor_value,2)+std::pow(microbee[robot_index].motor[2]*scale_motor_value,2)+std::pow(microbee[robot_index].motor[3]*scale_motor_value,2))*microbee[robot_index].state.bat_volt };  
    //printf("thrust_U_B = [ %f, %f, %f ]\n", thrust_U_B[0], thrust_U_B[1], thrust_U_B[2]); 
    float thrust_B[3] = {0., 0., thrust_U_B[2]*9.8/thrust_U0};
    float thrust[3] = {0};
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, R_BI, 3, thrust_B, 1, 1.0, thrust, 1); // B to I
    // kappa
    float kappa[3] = {0, 0, -9.8};
    cblas_saxpy(3, 1.0, thrust, 1, kappa, 1); // kappa <- 1.0*thrust+G

    for (int i = 0; i < 3; i++) // 0 for roll, 1 for pitch, 2 for throttle
    {
        *(z1[i]) += dt*(*(z2[i]) + 3*w0*leso_err[i]);
        *(z2[i]) += dt*(*(z3[i]) + 3*std::pow(w0,2)*leso_err[i] + kappa[i]);
        *(z3[i]) += dt*(std::pow(w0,3)*leso_err[i]);
    }

    float a_v[3] = {-*(z3[0]), -*(z3[1]), -*(z3[2])};
    
    // convert a_v to wind vector
    //                 [ 1/c_x, 0, 0 ]
    // v = m * R_B^I * [ 0, 1/c_y, 0 ] * R_I^B * a_v
    //                 [ 0, 0, 1/c_z ]
    // u = qr_speed - v 
    float c[9] = { 1./c_x, 0., 0.,
                   0., 1./c_y, 0.,
                   0., 0., 1./c_z};
    float c_B[9] = {0};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, R_BI, 3, c, 3, 0.0, c_B, 3); // B to I
    float c_BI[9] = {0};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, c_B, 3, R_IB, 3, 0.0, c_BI, 3); // B to I
    float v[3] = {0};
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, m, c_BI, 3, a_v, 1, 0.0, v, 1);
    float u[3]; memcpy(u, vel, 3*sizeof(float));
    cblas_saxpy(3, -1.0, v, 1, u, 1); // u <- -1.0*v+vel
    memcpy(wind_estimated, u, 3*sizeof(float));

#if 0
    printf("pos =  [ %f, %f, %f ]\n", pos[0], pos[1], pos[2]);
    printf("*z1  = [ %f, %f, %f ]\n", *(z1[0]), *(z1[1]), *(z1[2]));
    printf("*z2  = [ %f, %f, %f ]\n", *(z2[0]), *(z2[1]), *(z2[2]));
    printf("*z3  = [ %f, %f, %f ]\n", *(z3[0]), *(z3[1]), *(z3[2]));
    printf("u   = [ %f, %f, %f ]\n", kappa[0], kappa[1], kappa[2]);
    printf("a_v = [ %f, %f, %f ]\n", a_v[0], a_v[1], a_v[2]);
    printf("c_B = [ %f, %f, %f \n \
                    %f, %f, %f \n \
                    %f, %f, %f ]\n", c_B[0], c_B[1], c_B[2], c_B[3], c_B[4], c_B[5], c_B[6], c_B[7], c_B[8]);
    printf("v   = [ %f, %f, %f ]\n", v[0], v[1], v[2]);
    printf("u   = [ %f, %f, %f ]\n", u[0], u[1], u[2]);
#endif

#if 1 // calculate c
    // vel_B
    float vel_B[3];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, R_IB, 3, vel, 1, 0.0, vel_B, 1);
    // av_B
    float av_B[3];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, R_IB, 3, a_v, 1, 0.0, av_B, 1);
    float c_est[3];
    for (int i = 0; i < 3; i++)
        c_est[i] = m*av_B[i]/vel_B[i];
#endif

    // robot state
    Robot_State_t* robot_state = robot_get_state();
    memcpy(robot_state[robot_index].wind, wind_estimated, 3*sizeof(float));

    // for debug
    Anemometer_Data_t* wind_data = sonic_anemometer_get_wind_data();
    Robot_Debug_Record_t    new_dbg_rec;
    std::vector<Robot_Debug_Record_t>* robot_debug_rec = robot_get_debug_record();
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
    memcpy(new_dbg_rec.enu, pos, 3*sizeof(float));
    memcpy(new_dbg_rec.att, att, 3*sizeof(float));
    memcpy(new_dbg_rec.vel, vel, 3*sizeof(float));
    memcpy(new_dbg_rec.acc, acc, 3*sizeof(float));
    new_dbg_rec.throttle = rc_data[robot_index].throttle;
    new_dbg_rec.roll = rc_data[robot_index].roll;
    new_dbg_rec.pitch = rc_data[robot_index].pitch;
    new_dbg_rec.yaw = rc_data[robot_index].yaw;
    new_dbg_rec.leso_z1[0] = state[robot_index][0].z1;
    new_dbg_rec.leso_z1[1] = state[robot_index][1].z1;
    new_dbg_rec.leso_z1[2] = state[robot_index][2].z1;
    new_dbg_rec.leso_z2[0] = state[robot_index][0].z2;
    new_dbg_rec.leso_z2[1] = state[robot_index][1].z2;
    new_dbg_rec.leso_z2[2] = state[robot_index][2].z2;
    new_dbg_rec.leso_z3[0] = state[robot_index][0].z3;
    new_dbg_rec.leso_z3[1] = state[robot_index][1].z3;
    new_dbg_rec.leso_z3[2] = state[robot_index][2].z3;
    memcpy(new_dbg_rec.wind_estimated, wind_estimated, 3*sizeof(float));
    memcpy(new_dbg_rec.wind_resist_coef, c_est, 3*sizeof(float));
    memcpy(new_dbg_rec.anemometer[0], wind_data[0].speed, 3*sizeof(float));
    memcpy(new_dbg_rec.anemometer[1], wind_data[1].speed, 3*sizeof(float));
    memcpy(new_dbg_rec.anemometer[2], wind_data[2].speed, 3*sizeof(float));
    robot_debug_rec[robot_index].push_back(new_dbg_rec);
}

void microbee_wind_estimation_incl(int robot_index, float* att)
{
    float angle_incl;
    float e[3]; // -R_B^I * [0,0,1]^T
    float e_proj[3]; // e . [1,1,0]
    float v[3] = {0};

    // body frame to inertial frame
    float R[9] = {
        std::cos(att[1])*std::cos(att[2]),
        std::sin(att[0])*std::sin(att[1])*std::cos(att[2]) - std::cos(att[0])*std::sin(att[2]),
        std::cos(att[2])*std::sin(att[1])*std::cos(att[0]) + std::sin(att[0])*std::sin(att[2]),
        std::cos(att[1])*std::sin(att[2]),
        std::sin(att[1])*std::sin(att[0])*std::sin(att[2]) + std::cos(att[0])*std::cos(att[2]),
        std::cos(att[0])*std::sin(att[1])*std::sin(att[2]) - std::sin(att[0])*std::cos(att[2]),
        -std::sin(att[1]),
        std::sin(att[0])*std::cos(att[1]),
        std::cos(att[0])*std::cos(att[1]) 
    };
    // e = -R_B^I * [0,0,1]^T
    memset(e, 0, 3*sizeof(float));
    float unit[3] = {0., 0., -1.};
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, R, 3, unit, 1, 1.0, e, 1); // Fd B to I
    // e_proj = e . [1,1,0]
    memcpy(e_proj, e, 2*sizeof(float));
    // |e_proj|
    float nrm_e_proj = std::sqrt(e_proj[0]*e_proj[0] + e_proj[1]*e_proj[1]);
    if (nrm_e_proj < 0.001) {
        // no wind
        angle_incl = 0.;
        memset(v, 0, 3*sizeof(float));
    }
    else {
        angle_incl = std::asin(nrm_e_proj);
        // calculate wind strength
        float alpha = std::asin(nrm_e_proj)*180./M_PI;
        float strength = 0.00086*alpha*alpha + 0.08794*alpha + 0.06383;
        for (int i = 0; i < 2; i++)
            v[i] = strength*e_proj[i]/nrm_e_proj;
    }
    // triangle
    //memcpy(wind_estimated, QR_vel, 3*sizeof(float));
    //cblas_saxpy(3, -1.0, v, 1, wind_estimated, 1); // wind <- -1.0*v+vel
    
    // robot state
    Robot_State_t* robot_state = robot_get_state();
    memcpy(robot_state[robot_index].wind, v, 3*sizeof(float));
}

/*
 * TODO:
 *  dt not used
 */
static void microbee_pos_control(float dt, int robot_index)
{
/* Get position, velocity, acceleration, attitude ... */
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    float pos_ref[3], pos[3], vel[3], acc[3], att[3];
    memcpy(pos_ref, robot_ref[robot_index].enu, 3*sizeof(float)); // reference x/y/z
#ifdef MICROBEE_DENOISE_BEFORE_CONTROL
    for (int i = 0; i < 3; i++) // e/n/u
    {
        firfilt_rrrf_push(f_denoise[0][i], data->robot[robot_index].enu[i] + microbee_pos_fix[robot_index][i]);
        firfilt_rrrf_execute(f_denoise[0][i], &(pos[i]));
        firfilt_rrrf_push(f_denoise[1][i], data->robot[robot_index].vel[i]);
        firfilt_rrrf_execute(f_denoise[1][i], &(vel[i]));
        firfilt_rrrf_push(f_denoise[2][i], data->robot[robot_index].acc[i]);
        firfilt_rrrf_execute(f_denoise[2][i], &(acc[i]));
        firfilt_rrrf_push(f_denoise[3][i], data->robot[robot_index].att[i]);
        firfilt_rrrf_execute(f_denoise[3][i], &(att[i]));
    }
#else
    for (int i = 0; i < 3; i++) // e/n/u
    {
        pos[i] = data->robot[robot_index].enu[i] + microbee_pos_fix[robot_index][i]; // e/n/u axis, robot's real-time xyz axis pos
    }
    memcpy(vel, data->robot[robot_index].vel, 3*sizeof(float));
    memcpy(acc, data->robot[robot_index].acc, 3*sizeof(float));
    memcpy(att, data->robot[robot_index].att, 3*sizeof(float));
#endif

/* Position control */
    microbee_throttle_control_pid(dt, robot_index, pos_ref[2], pos[2], vel[2], acc[2]);
    microbee_roll_pitch_control_pid(dt, robot_index, pos_ref, pos, vel, acc, att);
    microbee_yaw_control_adrc(dt, robot_index, robot_ref[robot_index].heading, att[2]);
/* Wind estimation */
    microbee_wind_estimation_leso(dt, robot_index, pos, vel, acc, att);
    //microbee_wind_estimation_incl(robot_index, att);

//printf("pos = [ %f, %f, %f ]\n", pos[0], pos[1], pos[2]);
}

static void CreateGaussianKernel(float sigma, float* gKernel, int nWindowSize) {
    
    int nCenter = nWindowSize/2;
    double Value, Sum;
   
    // default no derivative
    gKernel[nCenter] = 1.0;
    Sum = 1.0;
    for (int i = 1; i <= nCenter; i++) {
        Value = 1.0/sigma*exp(-0.5*i*i/(sigma*sigma));
        if (nCenter+i < nWindowSize)
            gKernel[nCenter+i] = Value;
        gKernel[nCenter-i] = Value;
        Sum += 2.0*Value;
    }
    // normalize
    for (int i = 0; i < nWindowSize; i++)
        gKernel[i] = gKernel[i]/Sum;
}

void microbee_switch_to_manual(int idx_robot)
{
    if (idx_robot < 0 or idx_robot >= 4) // 0 1 2 3
        return;
    microbee_manual_control[idx_robot] = true;
}

void microbee_switch_to_auto(int idx_robot)
{
    if (idx_robot < 0 or idx_robot >= 4) // 0 1 2 3
        return;
    microbee_manual_control[idx_robot] = false;
}

void microbee_switch_all_to_manual(void)
{
    for (int i = 0; i < 4; i++)
        microbee_manual_control[i] = true;
}

void microbee_switch_all_to_auto(void)
{
    for (int i = 0; i < 4; i++)
        microbee_manual_control[i] = false;
}
