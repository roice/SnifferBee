/*
 * MicroBee Robot
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.25
 */
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include "robot/robot.h"
#include "robot/microbee.h"
#include "robot/pioneer.h"

static int type_of_robot; // 0 for ground robot, 1 for flying robot

static int amount_of_robots; // 1/2/3/4

static Robot_Ref_State_t robot_ref_state[4]; // 4 robots max
static Robot_State_t robot_state[4] = {{0},{0},{0},{0}};

std::vector<Robot_Record_t> robot_record[4]; // 4 robots max
std::vector<Robot_Debug_Record_t> robot_debug_record[4]; // 4 robots max

//#define DEBUG_HANDHELD_DEVICE

/* robot init */
bool robot_init(int type_robot, int num_robots)
{
    if (type_robot < 0 || type_robot > 1)
    {
        type_of_robot = 1;
        printf("Robot init: type_robot not recognized, set to flying robot by default.\n");
    }
    else
        type_of_robot = type_robot;

    if (num_robots < 1 || num_robots > 4)
    {
        amount_of_robots = 1;
        printf("Robot init: num_robots not in range 1-4, set to 1 by default.\n");
    }
    else
        amount_of_robots = num_robots;

    // init robot reference state
    for (int i = 0; i < 4; i++) // 4 robots max
        memset(&(robot_ref_state[i]), 0, sizeof(Robot_Ref_State_t));

    // prepare for the robot record
    for (int i = 0; i < amount_of_robots; i++) // 4 robots max
        robot_record[i].reserve(10*60*10); // 10 min record for 10 Hz sample

    if (type_robot == 0) { // ground robot
        if (!pioneer_control_init())
            return false;
    }
    else { // flying robot
        if (!microbee_state_init())
            return false;
#ifndef DEBUG_HANDHELD_DEVICE
        if (!microbee_control_init(amount_of_robots))
            return false;
#endif
    }

    return true;
}

void robot_shutdown(void)
{
    if (type_of_robot == 0) // ground robot
        pioneer_control_close();
    else {
#ifndef DEBUG_HANDHELD_DEVICE
        microbee_control_close(amount_of_robots);
#endif
        microbee_state_close();
    }
}

Robot_Ref_State_t* robot_get_ref_state(void)
{
    return robot_ref_state;
}

Robot_State_t* robot_get_state(void)
{
    return robot_state;
}

std::vector<Robot_Record_t>* robot_get_record(void)
{
    return robot_record;
}

std::vector<Robot_Debug_Record_t>* robot_get_debug_record(void)
{
    return robot_debug_record;
}
