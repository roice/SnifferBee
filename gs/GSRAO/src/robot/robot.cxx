/*
 * MicroBee Robot
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.25
 */
#include <stdio.h>
#include <vector>
#include "robot/robot.h"
#include "robot/microbee.h"

static Robot_Ref_State_t robot_ref_state[4]; // 4 robots max

std::vector<Robot_Record_t> robot_record[4]; // 4 robots max

//#define DEBUG_HANDHELD_DEVICE

/* robot init */
bool robot_init(void)
{
    if (!microbee_state_init())
        return false;
#ifndef DEBUG_HANDHELD_DEVICE
    if (!microbee_control_init())
        return false;
#endif

    // prepare for the robot record
    for (int i = 0; i < 4; i++) // 4 robots max
        robot_record[i].reserve(10*60*10); // 10 min record for 10 Hz sample

    return true;
}

void robot_shutdown(void)
{
#ifndef DEBUG_HANDHELD_DEVICE
    microbee_control_close();
#endif
    microbee_state_close();
}

Robot_Ref_State_t* robot_get_ref_state(void)
{
    return robot_ref_state;
}

std::vector<Robot_Record_t>* robot_get_record(void)
{
    return robot_record;
}
