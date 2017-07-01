/*
 * 3D reference position drawing
 *          using OpenGL GLUT
 *
 * This file contains stuff for drawing wind vectors for 3D
 * Robot Active Olfaction using OpenGL GLUT.
 * The declarations of the classes or functions are written in 
 * file draw_wind.h, which is included by DrawScene.h
 *
 * Author: Roice (LUO Bing)
 * Date: 2017-06-24 create this file
 */

#include <FL/gl.h>
#include "GSRAO_Config.h"
#include "robot/robot.h"

/* Args:
 *      pos     3D position
 */
void draw_robot_ref_pos_record(void)
{
    // get robot number
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    int num_robot = configs->robot.num_of_robots;

    // get robot ref position record
    std::vector<Robot_Record_t>* robot_rec = robot_get_record();

    for (int idx_robot = 0; idx_robot < num_robot; idx_robot++ ) {
        if (robot_rec[idx_robot].size() >= 2) {
            glDisable(GL_LIGHTING);
            glColor3f(0.5, 0.3, 0.3);
            glBegin(GL_LINES);
            for (int i = 0; i < robot_rec[idx_robot].size()-1; i++) {
                glVertex3f(robot_rec[idx_robot].at(i).ref_enu[0], 
                        robot_rec[idx_robot].at(i).ref_enu[2],
                        -robot_rec[idx_robot].at(i).ref_enu[1]);
                glVertex3f(robot_rec[idx_robot].at(i+1).ref_enu[0], 
                        robot_rec[idx_robot].at(i+1).ref_enu[2],
                        -robot_rec[idx_robot].at(i+1).ref_enu[1]);
            }
            glEnd();
            glEnable(GL_LIGHTING);
        }
    }
}
