/*
 * 3D QuadRotor Drawing
 *          using OpenGL GLUT
 *
 * This file defines function for drawing a quadrotor for 3D
 * Robot Active Olfaction using OpenGL GLUT.
 * The implementations of the classes or functions are written in 
 * file draw_qr.cxx. 
 * This file is included by DrawScene.h
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-02-23 create this file (RAOS)
 *       2016-05-30 modified this file (GSRAO)
 *       2016-08-11 modified this file (Player)
 */
#ifndef DRAW_QR_H
#define DRAW_QR_H

#include "robot/robot.h"

void draw_qr(robot_state_t*);

#endif
/* End of draw_qr.h */
