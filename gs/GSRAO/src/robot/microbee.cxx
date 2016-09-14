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
#include <math.h>
/* thread */
#include <pthread.h>
/* GSRAO */
#include "mocap/packet_client.h"
#include "robot/microbee.h"
#include "robot/robot.h"
#include "io/serial.h"
#include "GSRAO_Config.h"
/* CBLAS */
#include "cblas.h"

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

static void* microbee_control_loop(void*);
static void* microbee_state_loop(void*);
static void microbee_pos_control(float, int);

/*-------- MicroBee State Refresh --------*/

/* microbee state refresh init */
bool microbee_state_init(void)
{
    // clear microbee states
    for (int i = 0; i < 4; i++) // 4 robots max
        memset(&(microbee[i].state), 0, sizeof(microbee[i].state));

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

        // position control update
        microbee_pos_control(dtime, idx_robot);

        // 50 Hz
        nanosleep(&req, &rem); // 20 ms
    }

/* Step 3: Land */
    req.tv_sec = 0;
    req.tv_nsec = 20000000L; // 20 ms
    while(data->robot[idx_robot].enu[2] > MICROBEE_LANDING_THRESHOLD)
    {
        clock_gettime(CLOCK_REALTIME, &time);
        current_time = time.tv_sec + time.tv_nsec/1.0e9;
        dtime = current_time - previous_time;
        previous_time = current_time;

        robot_ref[idx_robot].enu[2] -= 0.3/50.0; // 0.3 m/s descending, 50 Hz control

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
static void microbee_throttle_control_pid(float dt, int robot_index)
{
    static float errorVelocityI[4] = {0}; // 4 robots max

    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile;

    // get alt
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    float EstAlt = data->robot[robot_index].enu[2]; // z axis, robot's real-time z axis pos
    float AltHold = robot_ref[robot_index].enu[2]; // reference z axis pos
    float vel_temp = data->robot[robot_index].vel[2]; // current alt velocity
    float accZ_temp = data->robot[robot_index].acc[2]; // current alt acc

    /* altitude control, throttle */
    // Altitude P-Controller
    float error = constrain(AltHold - EstAlt, -0.5, 0.5); // -0.5 - 0.5 m boundary
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
/*
 * TODO:
 *  dt not used
 */
static void microbee_roll_pitch_control_pid(float dt, int robot_index)
{
    static float errorPositionI[4][2] = {0};
    float error_enu[2]; // error vector in earth coordinate
    float error_p[2];// error vector in robot's coordinate
    float heading_e_front[2]; // heading unit vector, indicating front direction
    float heading_e_right[2]; // perpendicular vector of heading, indicating right
    float target_vel[2];
    float error;
    float result;

    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile;

    // get east/north
    float pos[2], pos_ref[2], vel[2]; // e/n
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref states
    for (int i = 0; i < 2; i++) // e/n
    {
        pos[i] = data->robot[robot_index].enu[i]; // e/n axis, robot's real-time x y axis pos
        pos_ref[i] = robot_ref[robot_index].enu[i]; // reference x y axis pos
        vel[i] = data->robot[robot_index].vel[i]; // current e/n velocity
    }

    // get position error vector in earth coordinate
    for (int i = 0; i < 2; i++)
        error_enu[i] = pos_ref[i] - pos[i];

    // get heading unit vectors
    float heading_angle = data->robot[robot_index].att[2];
    heading_e_front[0] = -sin(heading_angle); // for pitch
    heading_e_front[1] = cos(heading_angle);
    heading_e_right[0] = heading_e_front[1]; // for roll
    heading_e_right[1] = -heading_e_front[0];
    
    // convert error to robot's coordinate
    error_p[0] = cblas_sdot(2, error_enu, 1, heading_e_right, 1); // for roll
    error_p[1] = cblas_sdot(2, error_enu, 1, heading_e_front, 1); // for pitch

    // convert velocity to robot's coordinate
    float vel_p[2];
    vel_p[0] = cblas_sdot(2, vel, 1, heading_e_right, 1); // for roll
    vel_p[1] = cblas_sdot(2, vel, 1, heading_e_front, 1); // for pitch

    // convert acceleration to robot's coordinate
    float acc_p[2];
    float* acc = &(data->robot[robot_index].acc[0]); // current e/n acc
    acc_p[0] = cblas_sdot(2, acc, 1, heading_e_right, 1); // for roll
    acc_p[1] = cblas_sdot(2, acc, 1, heading_e_front, 1); // for pitch

#if 0
    printf("heading angle is %f\n", heading_angle);
    printf("pos_ref is [%f, %f] m, pos is [%f, %f] m\n", pos_ref[0], pos_ref[1], pos[0], pos[1]);
    printf("front error is %f m, right error is %f m\n", error_p[1], error_p[0]);
#endif
#if 0
    printf("error_enu is [%f, %f] m\n", error_enu[0], error_enu[1]);
#endif

    for (int i = 0; i < 2; i++) // 0 for roll, 1 for pitch
    {
        // Position PID-Controller for east(x)/north(y) axis
        target_vel[i] = constrain(pidProfile[robot_index].P[PIDPOS]*error_p[i], -0.3, 0.3); // limit error to +/- 0.3 m/s;
        target_vel[i] = applyDeadband(target_vel[i], 0.01); // 1 cm/s

        // Velocity PID-Controller
        error = target_vel[i]-vel_p[i];

        // P
        result = constrain((pidProfile[robot_index].P[PIDPOSR]*error), -100, 100); // limit to +/- 100
        // I
        errorPositionI[robot_index][i] += (pidProfile[robot_index].I[PIDPOSR]*error);
        errorPositionI[robot_index][i] = constrain(errorPositionI[robot_index][i], -200.0, 200.0); // limit to +/- 200
        result += errorPositionI[robot_index][i];
        // D
        result -= constrain(pidProfile[robot_index].D[PIDPOSR]*acc_p[i], -100, 100); // limit

        // update roll/pitch value
        SPP_RC_DATA_t* rc_data = spp_get_rc_data();
        if (i == 0)
            rc_data[robot_index].roll = constrain(1500 + result, 1000, 2000);
        else if (i == 1)
            rc_data[robot_index].pitch = constrain(1500 + result, 1000, 2000);
    }
}
/*
 * TODO:
 *  dt not used
 */
static void microbee_yaw_control_pid(float dt, int robot_index)
{
    static float errorYawI[4] = {0}; // 4 robots max
    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile;

    // get heading/heading_ref in earth coordinate
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    float heading = data->robot[robot_index].att[2]; // real-time heading angle
    float heading_ref = robot_ref[robot_index].heading; // reference heading angle

    // get yaw error in robot coordinate
    float yaw_error = heading - heading_ref;
    if (yaw_error > M_PI)
        yaw_error -= 2*M_PI;
    else if (yaw_error < -M_PI)
        yaw_error += 2*M_PI;

    // Position PID-Controller
    float error = constrain(yaw_error, -M_PI, M_PI);
    error = applyDeadband(error, M_PI/180.0); // 1 degree
    // P
    float result = constrain(pidProfile[robot_index].P[PIDMAG]*error, -200, 200);
    // I
    errorYawI[robot_index] += (pidProfile[robot_index].I[PIDMAG]*error);
    errorYawI[robot_index] = constrain(errorYawI[robot_index], -300.0, 300.0);
    result += errorYawI[robot_index];
    // D
    result += constrain(pidProfile[robot_index].D[PIDMAG]*data->robot[robot_index].omega[2], -100, 100); // limit

    //printf("yaw_error = %f, result = %f\n", yaw_error, result);

    // update yaw value
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
    rc_data[robot_index].yaw = constrain(1500 + result, 1050, 1950);
}

/* ADRC state vector type */
typedef struct {
    float z1;
    float z2;
    float z3;
    float u;
} ADRC_State_t;

/* LADRC
 *  z1(k) = z1(k-1)+T*(z2(k-1)+3*w0*(y-z1(k-1)))
 *  z2(k) = z2(k-1)+T*(z3(k-1)+3*w0^2*(y-z1(k-1))+u(k-1))
 *  z3(k) = z3(k-1)+T*(w0^3*(y-z1(k-1)))
 *  u0(k) = kp*(r-z1(k))+kd*(dr-z2(k))
 *  u(k) = u0(k)-z3(k)
 */
static void microbee_throttle_control_adrc(float dt, int robot_index)
{
    static ADRC_State_t state[4] = {{0},{0},{0},{0}}; // 4 robots max
    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    adrcProfile_t*   adrcProfile = configs->robot.adrcProfile;

    // get alt
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    float EstAlt = data->robot[robot_index].enu[2]; // z axis, robot's real-time z axis pos
    float AltHold = robot_ref[robot_index].enu[2]; // reference z axis pos

    /* altitude control, throttle */
    /* LADRC */
    float leso_err = EstAlt - state[robot_index].z1;
    // LESO
    state[robot_index].z1 += dt*(state[robot_index].z2 + 3*adrcProfile[robot_index].w0[ADRCALT]*leso_err);
    state[robot_index].z2 += dt*(state[robot_index].z3 + 3*pow(adrcProfile[robot_index].w0[ADRCALT],2)*leso_err + state[robot_index].u);
    state[robot_index].z3 += dt*(pow(adrcProfile[robot_index].w0[ADRCALT],3)*leso_err);
    // PD
    float pd_err = AltHold - state[robot_index].z1;
    float u0 = adrcProfile[robot_index].kp[ADRCALT]*pd_err + adrcProfile[robot_index].kd[ADRCALT]*(-state[robot_index].z2);
    state[robot_index].u = u0 - state[robot_index].z3;

    // convert u to result
    float result = 1.0*state[robot_index].u;

// for DEBUG
printf("AltHold = %f, EstAlt = %f, pd_err = %f, u0 = %f, z3 = %f, result = %f\n", AltHold, EstAlt, pd_err, u0, state[robot_index].z3, result);

    // update throttle value
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
    rc_data[robot_index].throttle = constrain(1050 + result, 1050, 1950);
}

static void microbee_roll_pitch_control_adrc(float dt, int robot_index)
{
    static ADRC_State_t state[4][2] = {{{0},{0}}, {{0},{0}}, {{0},{0}}, {{0},{0}}}; // 4 robots max 

    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    adrcProfile_t*   adrcProfile = configs->robot.adrcProfile;

    // get east/north
    float pos[2], pos_ref[2], vel[2]; // e/n
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref states
    for (int i = 0; i < 2; i++) // e/n
    {
        pos[i] = data->robot[robot_index].enu[i]; // e/n axis, robot's real-time x y axis pos
        pos_ref[i] = robot_ref[robot_index].enu[i]; // reference x y axis pos
        vel[i] = data->robot[robot_index].vel[i]; // current e/n velocity
    }

    // get position error vector in earth coordinate
    float error_enu[2]; // error vector in earth coordinate
    for (int i = 0; i < 2; i++)
        error_enu[i] = pos_ref[i] - pos[i];

    // get heading unit vectors
    float heading_angle = data->robot[robot_index].att[2];
    float heading_e_front[2]; // heading unit vector, indicating front direction
    float heading_e_right[2]; // perpendicular vector of heading, indicating right
    heading_e_front[0] = -sin(heading_angle); // for pitch
    heading_e_front[1] = cos(heading_angle);
    heading_e_right[0] = heading_e_front[1]; // for roll
    heading_e_right[1] = -heading_e_front[0];
    
    // convert error to robot's coordinate
    float error_p[2];// error vector in robot's coordinate
    error_p[0] = cblas_sdot(2, error_enu, 1, heading_e_right, 1); // for roll
    error_p[1] = cblas_sdot(2, error_enu, 1, heading_e_front, 1); // for pitch

    // convert velocity to robot's coordinate
    float vel_p[3];
    vel_p[0] = cblas_sdot(2, vel, 1, heading_e_right, 1); // for roll
    vel_p[1] = cblas_sdot(2, vel, 1, heading_e_front, 1); // for pitch
    vel_p[2] = data->robot[robot_index].vel[2];

    // convert acceleration to robot's coordinate
    float acc_p[3];
    float* acc = &(data->robot[robot_index].acc[0]); // current e/n acc
    acc_p[0] = cblas_sdot(2, acc, 1, heading_e_right, 1); // for roll
    acc_p[1] = cblas_sdot(2, acc, 1, heading_e_front, 1); // for pitch
    acc_p[2] = data->robot[robot_index].acc[2];

    /* position control, roll & pitch */
    float target_vel[2];
    for (int i = 0; i < 2; i++) {
        // Position P-Controller for east(x)/north(y) axis
        target_vel[i] = constrain(0.5*error_p[i], -0.3, 0.3); // limit error to +/- 0.3 m/s;
        target_vel[i] = applyDeadband(target_vel[i], 0.01); // 1 cm/s
    }
    // LADRC Velocity controller
    float b = 2.0;
    float leso_err[2] = {vel_p[0]-state[robot_index][0].z1, vel_p[1]-state[robot_index][1].z1};
    for (int i = 0; i < 2; i++) {
        // LESO
        state[robot_index][i].z1 += dt*(state[robot_index][i].z2 + 3*adrcProfile[robot_index].w0[ADRCPOS]*leso_err[i]);
        state[robot_index][i].z2 += dt*(state[robot_index][i].z3 + 3*pow(adrcProfile[robot_index].w0[ADRCPOS],2)*leso_err[i] + b*state[robot_index][i].u);
        state[robot_index][i].z3 += dt*(pow(adrcProfile[robot_index].w0[ADRCPOS],3)*leso_err[i]);
    }
    // PD
    float pd_err[2] = {target_vel[0]-state[robot_index][0].z1, target_vel[1]-state[robot_index][1].z1};
    float u0[2];
    for (int i = 0; i < 2; i++) {
        u0[i] = adrcProfile[robot_index].kp[ADRCPOS]*pd_err[i] + adrcProfile[robot_index].kd[ADRCPOS]*(-state[robot_index][i].z2);
        state[robot_index][i].u = u0[i] - state[robot_index][i].z3;
    }

    // convert u to result 
    float result[2] = {state[robot_index][0].u/b, state[robot_index][1].u/b};

    // DEBUG
printf("vel_enu = [%f, %f], vel_p = [%f, %f], vel_err = [%f, %f], pd_err = [%f, %f], u0 = [%f, %f], result = [%f, %f]\n", vel[0], vel[1], vel_p[0], vel_p[1], target_vel[0]-vel_p[0], target_vel[1]-vel_p[1], pd_err[0], pd_err[1], u0[0], u0[1], result[0], result[1]);

    // update roll/pitch value
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();
    for (int i = 0; i < 2; i++) {    
        if (i == 0)
            rc_data[robot_index].roll = constrain(1500 + result[i], 1000, 2000);
        else if (i == 1)
            rc_data[robot_index].pitch = constrain(1500 + result[i], 1000, 2000);
    }
}

static void microbee_roll_pitch_control_pid_leso(float dt, int robot_index)
{
    static float errorPositionI[4][2] = {0};
    float error_enu[2]; // error vector in earth coordinate
    float error_p[2];// error vector in robot's coordinate
    float heading_e_front[2]; // heading unit vector, indicating front direction
    float heading_e_right[2]; // perpendicular vector of heading, indicating right
    float target_vel[2];
    float error;
    float result;

    static ADRC_State_t state[4][2] = {{{0},{0}}, {{0},{0}}, {{0},{0}}, {{0},{0}}}; // 4 robots max

    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    pidProfile_t*   pidProfile = configs->robot.pidProfile;
    adrcProfile_t*   adrcProfile = configs->robot.adrcProfile;

    // get rc_data
    SPP_RC_DATA_t* rc_data = spp_get_rc_data();

    // get east/north
    float pos[2], pos_ref[2], vel[2]; // e/n
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref states
    for (int i = 0; i < 2; i++) // e/n
    {
        pos[i] = data->robot[robot_index].enu[i]; // e/n axis, robot's real-time x y axis pos
        pos_ref[i] = robot_ref[robot_index].enu[i]; // reference x y axis pos
        vel[i] = data->robot[robot_index].vel[i]; // current e/n velocity
    }

    // get position error vector in earth coordinate
    for (int i = 0; i < 2; i++)
        error_enu[i] = pos_ref[i] - pos[i];

    // get heading unit vectors
    float heading_angle = data->robot[robot_index].att[2];
    heading_e_front[0] = -sin(heading_angle); // for pitch
    heading_e_front[1] = cos(heading_angle);
    heading_e_right[0] = heading_e_front[1]; // for roll
    heading_e_right[1] = -heading_e_front[0];
    
    // convert error to robot's coordinate
    error_p[0] = cblas_sdot(2, error_enu, 1, heading_e_right, 1); // for roll
    error_p[1] = cblas_sdot(2, error_enu, 1, heading_e_front, 1); // for pitch

    // convert velocity to robot's coordinate
    float vel_p[3];
    vel_p[0] = cblas_sdot(2, vel, 1, heading_e_right, 1); // for roll
    vel_p[1] = cblas_sdot(2, vel, 1, heading_e_front, 1); // for pitch
    vel_p[2] = data->robot[robot_index].vel[2];

    // convert acceleration to robot's coordinate
    float acc_p[3];
    float* acc = &(data->robot[robot_index].acc[0]); // current e/n acc
    acc_p[0] = cblas_sdot(2, acc, 1, heading_e_right, 1); // for roll
    acc_p[1] = cblas_sdot(2, acc, 1, heading_e_front, 1); // for pitch
    acc_p[2] = data->robot[robot_index].acc[2];

#if 0
    printf("heading angle is %f\n", heading_angle);
    printf("pos_ref is [%f, %f] m, pos is [%f, %f] m\n", pos_ref[0], pos_ref[1], pos[0], pos[1]);
    printf("front error is %f m, right error is %f m\n", error_p[1], error_p[0]);
#endif
#if 0
    printf("error_enu is [%f, %f] m\n", error_enu[0], error_enu[1]);
#endif

    // LESO
    float leso_err[2] = {error_p[0]-state[robot_index][0].z1, error_p[1]-state[robot_index][1].z1};
    //float leso_err[2] = {vel_p[0]-state[robot_index][0].z1, vel_p[1]-state[robot_index][1].z1};
    //float leso_err[2] = {acc_p[0]-state[robot_index][0].z1, acc_p[1]-state[robot_index][1].z1};

    // Velocity-PID
    for (int i = 0; i < 2; i++) // 0 for roll, 1 for pitch
    {
        // Position PID-Controller for east(x)/north(y) axis
        target_vel[i] = constrain(pidProfile[robot_index].P[PIDPOS]*error_p[i], -0.3, 0.3); // limit error to +/- 0.3 m/s;
        target_vel[i] = applyDeadband(target_vel[i], 0.01); // 1 cm/s

        // Velocity PID-Controller
        error = target_vel[i]-vel_p[i];

        // P
        result = constrain((pidProfile[robot_index].P[PIDPOSR]*error), -100, 100); // limit to +/- 100
        // I
        errorPositionI[robot_index][i] += (pidProfile[robot_index].I[PIDPOSR]*error);
        errorPositionI[robot_index][i] = constrain(errorPositionI[robot_index][i], -200.0, 200.0); // limit to +/- 200
        result += errorPositionI[robot_index][i];
        // D
        result -= constrain(pidProfile[robot_index].D[PIDPOSR]*acc_p[i], -100, 100); // limit

        // update roll/pitch value 
        if (i == 0)
            rc_data[robot_index].roll = constrain(1500 + result, 1000, 2000);
        else if (i == 1)
            rc_data[robot_index].pitch = constrain(1500 + result, 1000, 2000);

        // LESO
        state[robot_index][i].z1 += dt*(state[robot_index][i].z2 + 3*adrcProfile[robot_index].w0[ADRCPOS]*leso_err[i]);
        state[robot_index][i].z2 += dt*(state[robot_index][i].z3 + 3*pow(adrcProfile[robot_index].w0[ADRCPOS],2)*leso_err[i] + result);
        state[robot_index][i].z3 += dt*(pow(adrcProfile[robot_index].w0[ADRCPOS],3)*leso_err[i]);
    }

// DEBUG
//printf("error_p = [%f, %f], z1 = [%f, %f], z2 = [%f, %f], z3 = [%f, %f]\n", error_p[0], error_p[1], state[robot_index][0].z1, state[robot_index][1].z1, state[robot_index][0].z2, state[robot_index][1].z2, state[robot_index][0].z3, state[robot_index][1].z3);

    // save z3 to wind est
    // wind vector = ground vector - flight/air vector
    Robot_State_t* robot_state = robot_get_state();
    robot_state[robot_index].wind[0] = 0.007*state[robot_index][0].z3;
    robot_state[robot_index].wind[1] = 0.007*state[robot_index][1].z3;
    //robot_state[robot_index].wind[0] = vel_p[0] - 0.007*state[robot_index][0].z3;
    //robot_state[robot_index].wind[1] = vel_p[1] - 0.007*state[robot_index][1].z3;
    // for debug
    Robot_Debug_Record_t    new_dbg_rec;
    std::vector<Robot_Debug_Record_t>* robot_debug_rec = robot_get_debug_record();
    memcpy(new_dbg_rec.enu, data->robot[robot_index].enu, 3*sizeof(float));
    memcpy(new_dbg_rec.att, data->robot[robot_index].att, 3*sizeof(float));
    memcpy(new_dbg_rec.vel, data->robot[robot_index].vel, 3*sizeof(float));
    memcpy(new_dbg_rec.acc, data->robot[robot_index].acc, 3*sizeof(float));
    memcpy(new_dbg_rec.vel_p, vel_p, 3*sizeof(float));
    memcpy(new_dbg_rec.acc_p, acc_p, 3*sizeof(float));
    new_dbg_rec.throttle = rc_data[robot_index].throttle;
    new_dbg_rec.roll = rc_data[robot_index].roll - 1500;
    new_dbg_rec.pitch = rc_data[robot_index].pitch - 1500;
    new_dbg_rec.yaw = rc_data[robot_index].yaw - 1500;
    new_dbg_rec.leso_z1[0] = state[robot_index][0].z1;
    new_dbg_rec.leso_z1[1] = state[robot_index][1].z1;
    new_dbg_rec.leso_z2[0] = state[robot_index][0].z2;
    new_dbg_rec.leso_z2[1] = state[robot_index][1].z2;
    new_dbg_rec.leso_z3[0] = state[robot_index][0].z3;
    new_dbg_rec.leso_z3[1] = state[robot_index][1].z3;
    robot_debug_rec[robot_index].push_back(new_dbg_rec);
}

static void microbee_yaw_control_adrc(float dt, int robot_index)
{
    static ADRC_State_t state[4] = {{0},{0},{0},{0}}; // 4 robots max
    // get configs
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    adrcProfile_t*   adrcProfile = configs->robot.adrcProfile;

    // get heading/heading_ref in earth coordinate
    MocapData_t* data = mocap_get_data(); // get mocap data
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state
    float heading = data->robot[robot_index].att[2]; // real-time heading angle
    float heading_ref = robot_ref[robot_index].heading; // reference heading angle

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

/*
 * TODO:
 *  dt not used
 */
static void microbee_pos_control(float dt, int robot_index)
{
#ifdef MICROBEE_POS_CONTROL_USE_ADRC
    microbee_throttle_control_pid(dt, robot_index);
    microbee_roll_pitch_control_pid_leso(dt, robot_index);
    microbee_yaw_control_adrc(dt, robot_index);
#else // use PID
    microbee_throttle_control_pid(dt, robot_index);
    microbee_roll_pitch_control_pid(dt, robot_index);
    microbee_yaw_control_pid(dt, robot_index);
#endif
}
