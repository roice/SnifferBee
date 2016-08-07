/*
 * Arena Drawing
 *
 * This file contains stuff for drawing an arena for 3D
 * Robot Active Olfaction using OpenGL GLUT.
 * The declarations of the classes or functions are written in 
 * file draw_arena.h, which is included by DrawScene.h
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-02-24 create this file
 */
#include <FL/gl.h>
#include <FL/glu.h>
#include <math.h> // floor()
#include "ui/draw/draw_arena.h"
#include "ui/draw/materials.h" // use material lists
#include "Config.h" // get configurations about Arena

void draw_arena()
{
    /* get configs of arena */
    Config_t *config = Config_get_configs();

    /* draw Ground */
    // calculate the four vertex of ground
    GLfloat va[3] = {config->arena.w/2.0, 0, -config->arena.l/2.0},
            vb[3] = {-config->arena.w/2.0, 0, -config->arena.l/2.0},
            vc[3] = {-config->arena.w/2.0, 0, config->arena.l/2.0},
            vd[3] = {config->arena.w/2.0, 0, config->arena.l/2.0};
    glPushMatrix();
    glTranslatef(0, -0.02, 0); // not 0 to avoid conflict with other objs
    glPushAttrib(GL_LIGHTING_BIT);

    glCallList(LAND_MAT);
  	glBegin(GL_POLYGON);
  	glNormal3f(0.0, 1.0, 0.0);
  	glVertex3fv(va);
  	glVertex3fv(vb);
  	glVertex3fv(vc);
  	glVertex3fv(vd);
  	glEnd();

    glPopAttrib(); 
    glPopMatrix();

    /* draw grid */ 
    glPushAttrib(GL_LIGHTING_BIT);
    glCallList(GRASS_MAT);
    glPushMatrix();
    glTranslatef(0, -0.01, 0); // not 0 to avoid conflict with other objs
    glEnable(GL_BLEND);
  	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	
    glBegin(GL_LINES);
    float e;
    for (e = -floor(va[0]);
            e <= floor(va[0]); e += 1.0)
    {
        glVertex3f(e, 0, va[2]);
        glVertex3f(e, 0, vd[2]);
    }
    for (e = -floor(vd[2]);
            e <= floor(vd[2]); e += 1.0)
    {
        glVertex3f(vc[0], 0, e);
        glVertex3f(vd[0], 0, e);
    }
    glEnd();
    glDisable(GL_BLEND);
    glPopMatrix();
    glPopAttrib();
}

/* End of draw_arena.cxx */
