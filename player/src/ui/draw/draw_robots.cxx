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

#include <FL/gl.h>
#include "Config.h"
#include "ui/draw/draw_qr.h"
#include "ui/draw/draw_arrow.h"

void draw_robots(void)
{
    // get robot number
    Config_t* configs = Config_get_configs();
    int num_robots = configs->robot.num_of_robots;

    // get mocap data
    MocapData_t* data = mocap_get_data();

    // get robot state
    Robot_State_t* robot_state = robot_get_state();

    for (int idx_robot = 0; idx_robot < num_robots; idx_robot++)
    {// draw every robot

        /* draw robot according to its type, configures... */
        // draw quadrotor
        draw_qr(&(data->robot[idx_robot]));

        // draw wind vector measurement/estimation
        draw_arrow(data->robot[idx_robot].enu[0],
            data->robot[idx_robot].enu[1],
            data->robot[idx_robot].enu[2],
            data->robot[idx_robot].enu[0] + 0.01*robot_state[idx_robot].wind[0],
            data->robot[idx_robot].enu[1] + 0.01*robot_state[idx_robot].wind[1],
            data->robot[idx_robot].enu[2] + 0.01*robot_state[idx_robot].wind[2]);
        
    }
}

/* End of draw_robots.cxx */
