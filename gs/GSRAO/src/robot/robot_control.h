/*
 * Robot control
 *         
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-05-23 create this file
 */

#ifndef ROBOT_CONTROL_H
#define ROBOT_CONTROL_H

typedef struct {
    float pos[3]; // position, xyz
    float att[3]; // attitude, roll pitch yaw
} RobotState_t;

typedef enum {
    PIDALT = 0,
    PIDPOS,
    PIDMAG,
    PIDVEL,
    PID_ITEM_COUNT
} pidIndex_e;

typedef struct {
    float P[PID_ITEM_COUNT];
    float I[PID_ITEM_COUNT];
    float D[PID_ITEM_COUNT];
} pidProfile_t;

bool robot_control_init(void);
void robot_control_close(void);
RobotState_t* robot_get_ref_state(void);

#endif
