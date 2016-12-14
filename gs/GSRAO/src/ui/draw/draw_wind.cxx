/*
 * 3D wind vector drawing
 *          using OpenGL GLUT
 *
 * This file contains stuff for drawing wind vectors for 3D
 * Robot Active Olfaction using OpenGL GLUT.
 * The declarations of the classes or functions are written in 
 * file draw_wind.h, which is included by DrawScene.h
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-09-07 create this file
 */
#include "GSRAO_Config.h"
#include "io/serial.h"
#include "ui/draw/draw_arrow.h"

/* Args:
 *      pos     3D position
 *      v       3D vector, m/s
 */
void draw_anemometer_results(void)
{
    // get anemometer number
    GSRAO_Config_t* configs = GSRAO_Config_get_configs();
    int num_anemo = configs->miscellaneous.num_of_anemometers;

    // get anemometer results
    Anemometer_Data_t* wind_data = sonic_anemometer_get_wind_data();

    float v[3];
    float sum_v[3] = {0};
    float pos[SERIAL_MAX_ANEMOMETERS][3] = {{0, -2.4, 1.35}, {0, -1.8, 1.35}, {0, -1.2, 1.35}, {0, -0.6, 1.35}};
    for (int i = 0; i < num_anemo; i ++) {
        for (int j = 0; j < 3; j++) {
            v[j] = wind_data[i].speed[j];
            sum_v[j] += v[j];
        }
        draw_arrow(pos[i][0], pos[i][1], pos[i][2], pos[i][0] + v[0], pos[i][1] + v[1], pos[i][2] + v[2], 0.0, 0.0, 1.0);
    }
}
