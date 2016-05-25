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

// robot reference state
typedef struct {
    float enu[3]; // e/n/u
    float heading; // heading
} Robot_Ref_State_t;

Robot_Ref_State_t* robot_get_ref_state(void);

bool robot_init(void);
void robot_shutdown(void);

#endif
