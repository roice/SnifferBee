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
#include "robot/robot.h"
#include "io/serial.h"

typedef struct {
    /* width, length and height */
    float w;
    float l;
    float h;
}GSRAO_Config_Arena_t;

typedef struct {
    std::string netcard; // netcard number
    std::string model_name_of_robot[4]; // 4 robots max
}GSRAO_Config_Mocap_t;

typedef struct {
    int num_of_robots; // amount of robots
    std::string ppm_serial_port_path;
    std::string dnet_serial_port_path[4]; // 4 robots max
    pidProfile_t    pidProfile[4]; // 4 robots max
    adrcProfile_t   adrcProfile[4]; // 4 robots max
}GSRAO_Config_Robot_t;

typedef struct {
    bool result_panel_opened;
    bool robot_panel_opened;
    bool remoter_panel_opened;
}GSRAO_Config_System_t;

typedef struct {
    int num_of_anemometers;
    std::string anemometer_serial_port_path[SERIAL_MAX_ANEMOMETERS];
    std::string anemometer_type[SERIAL_MAX_ANEMOMETERS];
} GSRAO_Config_Miscellaneous_t;

/* configuration struct */
typedef struct {
    /* Arena */
    GSRAO_Config_Arena_t arena;
    /* Mocap */
    GSRAO_Config_Mocap_t mocap;
    /* Robot */
    GSRAO_Config_Robot_t robot;
    /* Miscellaneous */
    GSRAO_Config_System_t system;
    GSRAO_Config_Miscellaneous_t miscellaneous;
}GSRAO_Config_t;

void GSRAO_Config_restore(void);
void GSRAO_Config_save(void);
void GSRAO_Config_init(void);
// get pointer of configuration data
GSRAO_Config_t* GSRAO_Config_get_configs(void);

#endif

/* End of GSRAO_Config.h */
