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
 */

#include <FL/gl.h>
#include "GSRAO_Config.h"
#include "mocap/packet_client.h"
#include "ui/draw/draw_qr.h"

void draw_robots(void)
{
    // get robot number
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    int num_robots = configs->robot.num_of_robots;

    // get mocap data
    MocapData_t* data = mocap_get_data();

    for (int idx_robot = 0; idx_robot < num_robots; idx_robot++)
    {// draw every robot

        /* draw robot according to its type, configures... */
        // draw quadrotor
        draw_qr(&(data->robot[idx_robot]));
        
    }
}

/* End of draw_robots.cxx */
