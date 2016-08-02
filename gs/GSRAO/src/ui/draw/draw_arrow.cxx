/*
 * 3D Arrow Drawing
 *          using OpenGL GLUT
 *
 * This file contains stuff for drawing a quadrotor for 3D
 * Robot Active Olfaction using OpenGL GLUT.
 * The declarations of the classes or functions are written in 
 * file draw_qr.h, which is included by DrawScene.h
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-08-02 create this file
 */

#include <FL/gl.h>

void draw_arrow(float start_x, float start_y, float start_z, float end_x, float end_y, float end_z)
{
    glDisable(GL_LIGHTING);
    glColor3f(0.0, 1.0, 0.0); // green
    glBegin(GL_LINES);
    glVertex3f(start_x, start_z, -start_y);
    glVertex3f(end_x, end_z, -end_y);
    glEnd();
    glEnable(GL_LIGHTING);
}
