/*
 * Configuration file of GSRAO
 *
 * This file contains declarations of the configuration data & methods
 * of GSRAO.
 * The implementations of the classes, functions and data are written in 
 * file GSRAO_Config.cxx.
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-04-16 create this file
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

typedef struct {
    /* width, length and height */
    float w;
    float l;
    float h;
}GSRAO_Config_Arena_t;

typedef struct {
    /* source position */
    float x;
    float y;
    float z;
    int pps;
    double mpp;
    /*  */
}GSRAO_Config_Source_t;

typedef struct {
    /* init pos */
    float init_x;
    float init_y;
    float init_z;
    std::string type;
}GSRAO_Config_Robot_t;

/* configuration struct */
typedef struct {
    /* Arena */
    GSRAO_Config_Arena_t arena;
    GSRAO_Config_Source_t source;
    GSRAO_Config_Robot_t robot;
}GSRAO_Config_t;

void GSRAO_Config_restore(void);
void GSRAO_Config_save(void);
void GSRAO_Config_init(void);
// get pointer of configuration data
GSRAO_Config_t* GSRAO_Config_get_configs(void);

#endif

/* End of GSRAO_Config.h */
