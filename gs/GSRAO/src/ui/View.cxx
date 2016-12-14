/*
 * 3D View of Robot Active Olfaction
 *          using OpenGL
 *
 * This file implements the classes of the view displaying for 3D
 * Robot Active Olfaction using OpenGL lib.
 * This file is included by UI.cxx, and the classes are
 * integrated into the FLTK UI.
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-04-16 create this file
 */

/* functions of class View which implements glut callbacks in UI.cxx */

#include <FL/glut.H>
#include <FL/glu.h>
#include <string.h>
#include <time.h> // for srand seeding and FPS calculation
#include <sys/time.h>
#include "ui/agv.h" // eye movement
#include "ui/draw/DrawScene.h" // draw experiment scene
#include "mocap/packet_client.h" // draw mocap fps
#include "GSRAO_Config.h"

// experiment start time
struct timeval  time_count_start;
bool count_experiment_time = false;

// width and height of current window, for redraw function
static int win_width = 1;
static int win_height = 1;

static void View_reshape(int w, int h)
{
    // update width/height of window
    win_width = w;
    win_height = h;

    glViewport(0, 0, w, h);
}

static void draw_axes(void);// draw axes
static void draw_notes(void);//draw notes
static void View_redraw(void)
{
    /* change eye moving */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(30.0, (GLdouble)win_width/win_height, 0.01, 1000);
    agvViewTransform();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    /* Begin drawing RAO scene */
    DrawScene(); // draw RAO scene
    /* End drawing */

    /* draw axes on the ground */
    draw_axes();

    /* draw notes */
    draw_notes();
    
    glutSwapBuffers(); // using two buffers mode

    // Use glFinish() instead of glFlush() to avoid getting many frames
    // ahead of the display (problem with some Linux OpenGL implementations...)
    //glFinish(); 
}

static void View_visible(int v)
{
    if (v == GLUT_VISIBLE)
        agvSetAllowIdle(1);
    else
    {
        glutIdleFunc(NULL);
        agvSetAllowIdle(0);
    }
}

static void draw_axes(void)
{
    float ORG[3] = {0,0,0};
    float XP[3] = {0,0,0}, YP[3] = {0,0,0};
    /* get configs about arena size */
    GSRAO_Config_t *config = GSRAO_Config_get_configs();
    XP[0] = config->arena.w?config->arena.w/2.0+1:5;
    YP[2] = config->arena.l?-(config->arena.l/2.0+1):-5;
    
    // draw x,y axes
    glDisable(GL_LIGHTING);
    {
        glLineWidth(2.0);
        glBegin(GL_LINES);
        glColor3f(1,0,0);
        glVertex3fv(ORG);
        glVertex3fv(XP);    // X axis is red.
        glColor3f(0,0,1);
        glVertex3fv(ORG);
        glVertex3fv(YP);    // y axis is blue.
        glEnd();

        // draw labels
        const char *str_x = "X/East";
        const char *str_y = "Y/North";
        glDisable(GL_DEPTH_TEST);
        glColor3f(0.3, 0.3, 0.3); // gray
        glRasterPos3fv(XP);
        gl_font(FL_HELVETICA_BOLD, 12); 
        gl_draw(str_x, strlen(str_x));
        glRasterPos3fv(YP); 
        gl_draw(str_y, strlen(str_y));
        glEnable(GL_DEPTH_TEST);
    }glEnable(GL_LIGHTING);
}

static void draw_ui_fps_note(void)
{
    time_t curtime; // current time
    char buf[255];
    static time_t fpstime = 0;
    static int fpscount = 0;
    static int fps = 0;

    glDisable(GL_LIGHTING);
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0.0, win_width, 0.0, win_height);
        sprintf(buf, "UI FPS=%d", fps);
        glColor3f(1.0f, 1.0f, 1.0f);
        gl_font(FL_HELVETICA, 12);
        gl_draw(buf, 10, 10);
    }glEnable(GL_LIGHTING);

    // Update frames-per-second
    fpscount ++;
    curtime = time(NULL);
    if ((curtime - fpstime) >= 2)
    {
        fps      = (fps + fpscount / (curtime - fpstime)) / 2;
        fpstime  = curtime;
        fpscount = 0;
    }
}

static void draw_mocap_fps_note(void)
{
    char buf[256];
    MocapData_t* data = mocap_get_data();

    int fps = data->dlatency>0? 1.0f/(data->dlatency):0;

    if (fps) {
        glDisable(GL_LIGHTING);
        {
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluOrtho2D(0.0, win_width, 0.0, win_height);
            sprintf(buf, "Mocap FPS=%d", fps);
            glColor3f(1.0f, 1.0f, 1.0f);
            gl_font(FL_HELVETICA, 12);
            gl_draw(buf, 120, 10);
        }glEnable(GL_LIGHTING);
    }
    else {
        glDisable(GL_LIGHTING);
        {
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluOrtho2D(0.0, win_width, 0.0, win_height);
            sprintf(buf, "No Mocap Data");
            glColor3f(1.0f, 0.0f, 0.0f);
            gl_font(FL_HELVETICA, 12);
            gl_draw(buf, 120, 10);
        }glEnable(GL_LIGHTING);
    }
}

static void draw_time_passed_note(void)
{
    char buf[256];
   
    // get time passed since start
    struct timeval  tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    double time_passed = (tv.tv_sec+tv.tv_usec/1000000.)-(time_count_start.tv_sec+time_count_start.tv_usec/1000000.);        // accurate down to ms

    if (count_experiment_time) {
        glDisable(GL_LIGHTING);
        {
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluOrtho2D(0.0, win_width, 0.0, win_height);
            sprintf(buf, "Time= %d m %.1f s", (int)(time_passed/60), time_passed-((int)(time_passed/60))*60);
            glColor3f(1.0f, 1.0f, 1.0f);
            gl_font(FL_HELVETICA, 12);
            gl_draw(buf, 260, 10);
        }glEnable(GL_LIGHTING);
    }
}

static void draw_notes(void) {
    draw_ui_fps_note();     // frames per second of UI
    draw_mocap_fps_note();  // frames per second of Motion Capture
    draw_time_passed_note();// time passed since start
}

static void View_idle(void) {

    // update view
    if (agvMoving) agvMove();
    View_redraw();
}

void View_init(int width, int height)
{
    // set width/height of window
    win_width = width;
    win_height = height;

    agvInit(0); /* 0 cause we have our own idle */
    // config callbacks for glut
    //  these functions will not be called immediately
    glutReshapeFunc(View_reshape);
    glutDisplayFunc(View_redraw);
    glutVisibilityFunc(View_visible);
    glutIdleFunc(View_idle);

    /* Initialize GL stuff */
    glShadeModel(GL_FLAT);// or use GL_SMOOTH with more computation
    glClearColor(0.49, 0.62, 0.75, 0.0);
  	glClearDepth(1.0f);
  	glEnable(GL_DEPTH_TEST);// can disable for lower computation
  	glDepthFunc(GL_LEQUAL);
  	glDisable(GL_BLEND);// cause GL_DEPTH_TEST enabled
  	glDisable(GL_ALPHA_TEST);
  	glMatrixMode(GL_PROJECTION);
  	glLoadIdentity();
    gluPerspective(30.0, (GLdouble)width/height, 0.01, 1000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    /* init scene drawing */
    DrawScene_init(); 
}

void View_start_count_time(void)
{
    /* start time counting */
    struct timezone tz;
    gettimeofday(&time_count_start, &tz);
    count_experiment_time = true;
}

void View_stop_count_time(void)
{
    /* stop time counting */
    count_experiment_time = false;
}
/* End of View.cxx */
