#include "robot/robot.h"

robot_state_t robot_state;

robot_state_t* robot_get_state(void)
{
    return &robot_state;
}
