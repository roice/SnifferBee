/*
 * MicroBee Robot
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.25
 */
#ifndef ROBOT_H
#define ROBOT_H

#include <vector>

typedef enum {
    PIDALT = 0,
    PIDVEL,
    PIDPOS,
    PIDPOSR,
    PIDMAG,
    PID_ITEM_COUNT
} pidIndex_e;

typedef struct {
    float P[PID_ITEM_COUNT];
    float I[PID_ITEM_COUNT];
    float D[PID_ITEM_COUNT];
} pidProfile_t;

typedef enum {
    ADRCALT = 0,
    ADRCPOS,
    ADRCMAG,
    ADRC_ITEM_COUNT
} adrcIndex_e;

typedef struct {
    float w0[ADRC_ITEM_COUNT];
    float kp[ADRC_ITEM_COUNT];
    float kd[ADRC_ITEM_COUNT];
} adrcProfile_t;

// robot reference state
typedef struct {
    float enu[3]; // e/n/u
    float heading; // heading
} Robot_Ref_State_t;

// robot state
typedef struct {
    float enu[3];
    float heading;
    float wind[3];
} Robot_State_t;

// robot record
typedef struct {
    float enu[3]; // ENU position
    float att[3]; // roll/pitch/yaw
    float sensor[3]; // sensor readings
    float wind[3]; // wind measurement/estimation
    int motor[4]; // values of 4 motors
    float bat_volt; // battery voltage
    int count;
    double time;
} Robot_Record_t;

Robot_State_t* robot_get_state(void);

Robot_Ref_State_t* robot_get_ref_state(void);

std::vector<Robot_Record_t>* robot_get_record(void);

bool robot_init(int);
void robot_shutdown(void);

#endif
