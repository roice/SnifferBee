/*
 * 3D Robots Drawing
 *          using OpenGL GLUT
 *
 * This file contains stuff for drawing the robots for 3D
 * Robot Active Olfaction using OpenGL GLUT.
 * The declarations of the classes or functions are written in 
 * file draw_robots.h, which is included by DrawScene.h
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-03-09 create this file (RAOS)
 *       2016-05-30 modified this file (GSRAO)
 *       2o16-08-07 modified this file (Player)
 */

#include <string.h>
#include <FL/gl.h>
#include "Player_Config.h"
#include "ui/draw/draw_qr.h"
#include "ui/draw/draw_arrow.h"
#include "io/play_thread.h"
#include "robot/robot.h"
#include "foc/vector_rotation.h"
#include "io/play_thread.h"
#include "foc/flying_odor_compass.h"

void draw_robots(void)
{
    // get robot number
    //Config_t* configs = Config_get_configs();

    // get robot info
    robot_state_t* robot_state = robot_get_state();

    // TODO: multiple robots
    for (int idx_robot = 0; idx_robot < 1; idx_robot++)
    {// draw every robot

        /* draw robot according to its type, configures... */
        // draw quadrotor
        draw_qr(robot_state); 
    }

    if ((Flying_Odor_Compass*)play_thread_get_data()) {
        // get wind info
        std::vector<FOC_Input_t>& data_raw = ((Flying_Odor_Compass*)play_thread_get_data())->data_raw;
        // draw wind arrow
        if (data_raw.size() > 0)
            draw_arrow(robot_state->position[0], robot_state->position[1], robot_state->position[2], robot_state->position[0]+data_raw.back().wind[0], robot_state->position[1]+data_raw.back().wind[1], robot_state->position[2]+data_raw.back().wind[2], 0, 1.0, 0);
    }
}

/* End of draw_robots.cxx */
