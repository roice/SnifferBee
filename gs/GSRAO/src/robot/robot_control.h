/*
 * Robot control
 *         
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-05-23 create this file
 */

typedef struct {
    float pos[3]; // position, xyz
    float att[3]; // attitude, roll pitch yaw
} RobotState_t;

bool robot_control_init(void);
void robot_control_close(void);
RobotState_t* robot_get_ref_state(void);

