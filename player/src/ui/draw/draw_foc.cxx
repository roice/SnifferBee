/*
 * Drawing results of flying odor compass algorithm
 *          using OpenGL GLUT
 *
 * This file contains stuff for drawing results of flying odor
 * compass for 3D Robot Active Olfaction using OpenGL GLUT.
 * The declarations of the classes or functions are written in 
 * file draw_foc.h, which is included by DrawScene.h
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-08-12 create this file
 */
#include GL_HEADER
#include GLUT_HEADER
#include <cmath>
#include <vector>
#include "cblas.h"
#include "ui/draw/materials.h"
#include "io/play_thread.h"
#include "foc/flying_odor_compass.h"
#include "robot/robot.h"
#include "foc/wake_qr.h"
#include "ui/draw/draw_arrow.h"
#include "foc/vector_rotation.h"

static void draw_particles(std::vector<FOC_Particle_t>*);
static void draw_wakes(std::vector<Wake_QR_ring_t>*);
static void draw_virtual_plumes(std::vector<FOC_Particle_t>*);
static void draw_est_direction(FOC_Estimation_t&, robot_state_t*);
static void draw_filtered_est_direction(std::vector<FOC_Estimation_t>&, robot_state_t*);

void draw_foc(void)
{
    if (!play_thread_get_data())
        return;

    std::vector<FOC_Estimation_t>& data_est = ((Flying_Odor_Compass*)play_thread_get_data())->data_est;
    //std::vector<FOC_Delta_t>& delta = ((Flying_Odor_Compass*)play_thread_get_data())->data_delta;

    std::vector<Wake_QR_ring_t>* wake_rings = wake_qr_get_info_vortex_rings();

    // get robot info
    robot_state_t* robot_state = robot_get_state();

    if (data_est.size() > 0)
    {
        // draw particles
        //draw_particles(data_est.back().particles);
        // draw qr wakes
        //draw_wakes(wake_rings);
        // draw virtual plumes
        //draw_virtual_plumes(data_est.back().particles);
        // draw estimated direction
        //draw_est_direction(data_est.back(), robot_state);
        // draw filtered est direction
        draw_filtered_est_direction(data_est, robot_state);
    }
}

static void draw_est_direction(FOC_Estimation_t& est, robot_state_t* robot_state)
{
    if (est.valid) {
        draw_arrow(robot_state->position[0],
            robot_state->position[1],
            robot_state->position[2],
            robot_state->position[0] + est.direction[0],
            robot_state->position[1] + est.direction[1],
            robot_state->position[2] + est.direction[2],
            1.0, 0.0, 0.0);
    }
}


static void draw_filtered_est_direction(std::vector<FOC_Estimation_t>& data_est, robot_state_t* robot_state)
{
    if (data_est.size() < 200)
        return;

    float w;
    double direction[3] = {0};
    double norm_direction;
    for (int i = data_est.size()-200; i < data_est.size(); i++) {
        //if (!data_est.at(i).valid)
        //    continue;
        //w = std::sqrt(delta.at(i).std[0]*delta.at(i).std[0] + delta.at(i).std[1]*delta.at(i).std[1] + delta.at(i).std[2]*delta.at(i).std[2]);
        for (int j = 0; j < 3; j++)
            //direction[j] += w*data_est.at(i).direction[j];
            direction[j] += data_est.at(i).direction[j];//*data_est.at(i).belief;
    }
    norm_direction = std::sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
    if (norm_direction == 0)
        return;
   
    //printf("Direction = [%f, %f, %f]\n", direction[0], direction[1], direction[2]);

    for (int i = 0; i < 3; i++)
        direction[i] /= norm_direction;

    draw_arrow(robot_state->position[0],
            robot_state->position[1],
            robot_state->position[2],
            robot_state->position[0] + direction[0],
            robot_state->position[1] + direction[1],
            robot_state->position[2] + direction[2],
            0.0, 0.0, 1.0);
}


static void draw_particles(std::vector<FOC_Particle_t>* particles)
{
    float pos[3];

    if (particles != NULL and particles->size() > 0) {

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        for (int i = 0; i < particles->size(); i++) // for each filament
        {
            if (particles->at(i).plume->size() < 1)
                continue;
            memcpy(pos, particles->at(i).plume->front().pos, 3*sizeof(float));
            glPushMatrix();
            glTranslatef(pos[0], pos[2], -pos[1]);
            glPushAttrib(GL_LIGHTING_BIT);
            SimMaterial_smoke(1.0-particles->at(i).weight);
            glutSolidSphere(particles->at(i).weight, 8, 3);
            glPopAttrib();
            glPopMatrix();
        }
        glDisable(GL_BLEND);
    }
}

static void draw_wakes(std::vector<Wake_QR_ring_t>* wake_rings)
{
    static std::vector<Wake_QR_ring_t> rings;

    rings.clear();
    std::copy(wake_rings->begin(), wake_rings->end(), std::back_inserter(rings));

    if (rings.size() < 1)
        return;

    // draw vortex rings
    float temp_v[3] = {0};
    float mkr[3];
    float theta;
    int n = 10; // 10 segments

    glDisable(GL_LIGHTING);
    glColor3f(0.0, 0.0, 1.0); /* blue */
    for (int idx_rotor = 0; idx_rotor < 4; idx_rotor++) { // for 4 rotors 
        for (int idx_ring = 0; idx_ring < rings.size(); idx_ring++) { // for rings
            glBegin(GL_LINE_LOOP);
            for (int i = 0; i < n; i++)
            {   
                theta = 2.0f*M_PI*i/n;
                temp_v[0] = cosf(theta)*QR_PROP_RADIUS;
                temp_v[1] = sinf(theta)*QR_PROP_RADIUS;
                memset(mkr, 0, 3*sizeof(float));
                rotate_vector(temp_v, mkr, rings.at(idx_ring).att[2], rings.at(idx_ring).att[1], rings.at(idx_ring).att[0]);
                glVertex3f(rings.at(idx_ring).pos[idx_rotor][0]+mkr[0], rings.at(idx_ring).pos[idx_rotor][2]+mkr[2], -rings.at(idx_ring).pos[idx_rotor][1]+mkr[1]);
            }
            glEnd();
        }
    } 
    glEnable(GL_LIGHTING);
}

static void draw_virtual_plumes(std::vector<FOC_Particle_t>* particles)
{
    static std::vector<FOC_Puff_t> plume;

    if (particles != NULL and particles->size() > 0) {

        glEnable(GL_BLEND);
  	    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPushAttrib(GL_LIGHTING_BIT);
        glCallList(VORTICE_MAT);

        for (int i = 0; i < particles->size(); i++) // for each filament
        {
            plume.clear();
            std::copy(particles->at(i).plume->begin(), particles->at(i).plume->end(), std::back_inserter(plume));
            
            if (plume.size() < 2) continue; 

            glBegin(GL_LINES);
            for (int j = 0; j < plume.size()-1; j++) {
                // draw plume as line segments
  	            glVertex3f(plume.at(j).pos[0], plume.at(j).pos[2], -plume.at(j).pos[1]);
                glVertex3f(plume.at(j+1).pos[0], plume.at(j+1).pos[2], -plume.at(j+1).pos[1]);
            }
            glEnd();
        }
        glPopAttrib();
        glDisable(GL_BLEND);
    }
}
