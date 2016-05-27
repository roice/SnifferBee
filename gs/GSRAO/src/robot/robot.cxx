/*
 * MicroBee Robot
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.25
 */
#include <stdio.h>
#include "robot/robot.h"
#include "robot/microbee.h"

static Robot_Ref_State_t robot_ref_state[4]; // 4 robots max

/* robot init */
bool robot_init(void)
{
    if (!microbee_state_init())
        return false;

    if (!microbee_control_init())
        return false;

    return true;
}

void robot_shutdown(void)
{
    microbee_control_close();
    microbee_state_close();
}

Robot_Ref_State_t* robot_get_ref_state(void)
{
    return robot_ref_state;
}
