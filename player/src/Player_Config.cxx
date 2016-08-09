/*
 * Configuration file of Player
 *
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-08-07 create this file
 */
#include "Player_Config.h"
// for .ini file reading
#include <boost/property_tree/ptree.hpp>  
#include <boost/property_tree/ini_parser.hpp>

/* Configuration data */
static Config_t settings;

/* Restore settings from configuration file */
void Config_restore(void)
{
    /* check if there exists a config file */
    if(access("settings.cfg", 0))
    {// config file not exist
        Config_init(); // load default config
        // create new config file
        FILE *fp;
        fp = fopen("settings.cfg", "w+");
        fclose(fp);
    }
    else // config file exist
    {
        /* read configuration files */
        boost::property_tree::ptree pt;
        boost::property_tree::ini_parser::read_ini("settings.cfg", pt);
        /* restore configs */
        // arena
        settings.arena.w = pt.get<float>("Arena.width");
        settings.arena.l = pt.get<float>("Arena.length");
        settings.arena.h = pt.get<float>("Arena.height"); 
        // System
        settings.system.robot_panel_opened = pt.get<bool>("System.robot_panel_opened");
        settings.system.result_panel_opened = pt.get<bool>("System.result_panel_opened");
    }
}

/* Save settings to configuration file */
void Config_save(void)
{
    /* prepare to write configuration files */
    boost::property_tree::ptree pt;
    // arena size
    pt.put("Arena.width", settings.arena.w);
    pt.put("Arena.length", settings.arena.l);
    pt.put("Arena.height", settings.arena.h); 
    // System
    pt.put("System.robot_panel_opened", settings.system.robot_panel_opened);
    pt.put("System.result_panel_opened", settings.system.result_panel_opened);
    /* write */
    boost::property_tree::ini_parser::write_ini("settings.cfg", pt);
}

/* init settings (obsolete) */
void Config_init(void)
{
    /* init arena settings */
    // arena
    settings.arena.w = 10; // x
    settings.arena.l = 10; // y
    settings.arena.h = 10; // z
    // system
    settings.system.robot_panel_opened = false;
    settings.system.result_panel_opened = false;
}

/* get pointer of config data */
Config_t* Config_get_configs(void)
{
    return &settings;
}

/* End of GSRAO_Config.cxx */
