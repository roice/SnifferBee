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
#include "GSRAO_Config.h" // get configurations about Arena

/* declarations of local functions */
// functions to create arenas
static void draw_arena_basic(void);

void draw_arena(int arena_name)
{
    switch(arena_name)
    {
        case SIM_ARENA_BASIC:
            draw_arena_basic();
        default:
            draw_arena_basic();
    }
}

/* functions to draw arenas */
static void draw_arena_basic(void)
{
    /* get configs of arena */
    GSRAO_Config_t *config = GSRAO_Config_get_configs();

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

    /* draw chimney (odor source), a cylinder */
    GLUquadricObj * chimney_obj = gluNewQuadric();
    gluQuadricDrawStyle(chimney_obj, GLU_FILL);
    gluQuadricNormals(chimney_obj, GLU_SMOOTH);

    glPushMatrix();
    glTranslatef(config->source.x, 0, -config->source.y);
    glRotatef(-90, 1, 0, 0); // make it upright
    glPushAttrib(GL_LIGHTING_BIT);

    glCallList(CEMENT_MAT);
    gluCylinder(chimney_obj, 
            0.2, // base radius
            0.1, // top radius
            config->source.z, // length
            8, /*slices*/ 3 /*stacks*/);
    glPopAttrib(); 
    glPopMatrix();
}

/* End of draw_arena.cxx */
