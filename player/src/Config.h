/*
 * Configuration file of Player
 *
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-08-07 create this file
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

typedef struct {
    /* width, length and height */
    float w;
    float l;
    float h;
} Config_Arena_t;

typedef struct {
    bool result_panel_opened;
    bool robot_panel_opened;
} Config_System_t;

/* configuration struct */
typedef struct {
    /* Arena */
    Config_Arena_t arena;
    /* Miscellaneous */
    Config_System_t system;
} Config_t;

void Config_restore(void);
void Config_save(void);
void Config_init(void);
// get pointer of configuration data
Config_t* Config_get_configs(void);

#endif

/* End of GSRAO_Config.h */
