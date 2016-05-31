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
 */
#ifndef DRAW_QR_H
#define DRAW_QR_H

#include "mocap/packet_client.h"

void draw_qr(MocapRigidBody_t*);

#endif
/* End of draw_qr.h */
