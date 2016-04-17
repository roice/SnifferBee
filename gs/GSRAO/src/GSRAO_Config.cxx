/*
 * Configuration file of GSRAO
 *
 * This file contains configuration data and methods of GSRAO.
 * The declarations of the classes, functions and data are written in 
 * file GSRAO_Config.h, which is included by main.cxx and
 * user interface & drawing files.
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-04-16 create this file
 */
#include "GSRAO_Config.h"
// for .ini file reading
#include <boost/property_tree/ptree.hpp>  
#include <boost/property_tree/ini_parser.hpp>

/* Configuration data */
static GSRAO_Config_t settings;

/* Restore settings from configuration file */
void GSRAO_Config_restore(void)
{
    /* check if there exists a config file */
    if(access("settings.cfg", 0))
    {// config file not exist
        GSRAO_Config_init(); // load default config
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
        // source
        settings.source.x = pt.get<float>("Source.x"); // m
        settings.source.y = pt.get<float>("Source.y");
        settings.source.z = pt.get<float>("Source.z");
        settings.source.pps = pt.get<int>("Source.pps"); // parcels per sec
        settings.source.mpp = pt.get<double>("Source.mpp"); // molecules per parcel (10^18)
        // robot
        settings.robot.init_x = pt.get<float>("Robot.init_x");
        settings.robot.init_y = pt.get<float>("Robot.init_y");
        settings.robot.init_z = pt.get<float>("Robot.init_z");
        settings.robot.type = pt.get<std::string>("Robot.type");
    }
}

/* Save settings to configuration file */
void GSRAO_Config_save(void)
{
    /* prepare to write configuration files */
    boost::property_tree::ptree pt;
    // arena size
    pt.put("Arena.width", settings.arena.w);
    pt.put("Arena.length", settings.arena.l);
    pt.put("Arena.height", settings.arena.h);
    // odor source
    pt.put("Source.x", settings.source.x);
    pt.put("Source.y", settings.source.y);
    pt.put("Source.z", settings.source.z);
    pt.put("Source.pps", settings.source.pps);
    pt.put("Source.mpp", settings.source.mpp);
    // robot
    pt.put("Robot.init_x", settings.robot.init_x);
    pt.put("Robot.init_y", settings.robot.init_y);
    pt.put("Robot.init_z", settings.robot.init_z);
    pt.put("Robot.type", settings.robot.type);
    /* write */
    boost::property_tree::ini_parser::write_ini("settings.cfg", pt);
}

/* init settings (obsolete) */
void GSRAO_Config_init(void)
{
    /* init arena settings */
    // arena
    settings.arena.w = 10; // x
    settings.arena.l = 10; // y
    settings.arena.h = 10; // z
    // source
    settings.source.x = 0;
    settings.source.y = 0;
    settings.source.z = 2; // 2 m
    settings.source.pps = 100; // 100 parcels per second
    settings.source.mpp = 2.355; // 2.355*10^18 molecules/parcel, for 100 grams/hr release rate of chlorine when pps = 100
    // robot
    settings.robot.init_x = 1;
    settings.robot.init_y = 0;
    settings.robot.init_z = 2;
    settings.robot.type = "quadrotor";
}

/* get pointer of config data */
GSRAO_Config_t* GSRAO_Config_get_configs(void)
{
    return &settings;
}

/* End of GSRAO_Config.cxx */
