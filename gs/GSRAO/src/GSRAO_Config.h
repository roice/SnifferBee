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
    std::string netcard; // netcard number
    std::string model_name_of_robot[4]; // 4 robots max
}GSRAO_Config_Mocap_t;

typedef struct {
    std::string serial_port_path;
}GSRAO_Config_ppmCnt_t;

typedef struct {
    int num_of_robots; // amount of robots
    std::string ppm_serial_port_path;
    std::string dnet_serial_port_path;
}GSRAO_Config_Robot_t;

typedef struct {
    bool result_panel_opened;
    bool robot_panel_opened;
    bool remoter_panel_opened;
}GSRAO_Config_System_t;

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
}GSRAO_Config_t;

void GSRAO_Config_restore(void);
void GSRAO_Config_save(void);
void GSRAO_Config_init(void);
// get pointer of configuration data
GSRAO_Config_t* GSRAO_Config_get_configs(void);

#endif

/* End of GSRAO_Config.h */
