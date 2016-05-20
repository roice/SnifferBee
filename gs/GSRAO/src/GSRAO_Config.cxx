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
        // Motion capture network
        settings.mocap.netcard = pt.get<std::string>("Mocap.netcard");
        settings.mocap.model_name_of_robot[0] = pt.get<std::string>("Mocap.model_name_of_robot_1");
        settings.mocap.model_name_of_robot[1] = pt.get<std::string>("Mocap.model_name_of_robot_2");
        settings.mocap.model_name_of_robot[2] = pt.get<std::string>("Mocap.model_name_of_robot_3");
        settings.mocap.model_name_of_robot[3] = pt.get<std::string>("Mocap.model_name_of_robot_4");
        // Robot
        settings.robot.num_of_robots = pt.get<int>("Robot.num_of_robots");
        settings.robot.ppm_serial_port_path = pt.get<std::string>("Robot.ppm_serial_port_path");
        settings.robot.dnet_serial_port_path = pt.get<std::string>("Robot.dnet_serial_port_path");
        // System
        settings.system.robot_panel_opened = pt.get<bool>("System.robot_panel_opened");
        settings.system.result_panel_opened = pt.get<bool>("System.result_panel_opened");
        settings.system.remoter_panel_opened = pt.get<bool>("System.remoter_panel_opened");
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
    // Mocap network
    pt.put("Mocap.netcard", settings.mocap.netcard);
    pt.put("Mocap.model_name_of_robot_1", settings.mocap.model_name_of_robot[0]);
    pt.put("Mocap.model_name_of_robot_2", settings.mocap.model_name_of_robot[1]);
    pt.put("Mocap.model_name_of_robot_3", settings.mocap.model_name_of_robot[2]);
    pt.put("Mocap.model_name_of_robot_4", settings.mocap.model_name_of_robot[3]);
    // Robot
    pt.put("Robot.num_of_robots", settings.robot.num_of_robots);
    pt.put("Robot.ppm_serial_port_path", settings.robot.ppm_serial_port_path);
    pt.put("Robot.dnet_serial_port_path", settings.robot.dnet_serial_port_path);
    // System
    pt.put("System.robot_panel_opened", settings.system.robot_panel_opened);
    pt.put("System.result_panel_opened", settings.system.result_panel_opened);
    pt.put("System.remoter_panel_opened", settings.system.remoter_panel_opened);
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
    // mocap
    settings.mocap.netcard = "lo IPv4 127.0.0.1"; // netcard name
    // 4 robots max
    settings.mocap.model_name_of_robot[0] = "Rigid Body 1";
    settings.mocap.model_name_of_robot[1] = "Rigid Body 2";
    settings.mocap.model_name_of_robot[2] = "Rigid Body 3";
    settings.mocap.model_name_of_robot[3] = "Rigid Body 4";
    // robot
    settings.robot.num_of_robots = 1;
    settings.robot.ppm_serial_port_path = "/dev/ttyUSB_GSRAO_PPM";
    settings.robot.dnet_serial_port_path = "/dev/ttyUSB_GSRAO_DATA";
    // system
    settings.system.robot_panel_opened = false;
    settings.system.result_panel_opened = false;
    settings.system.remoter_panel_opened = false;
}

/* get pointer of config data */
GSRAO_Config_t* GSRAO_Config_get_configs(void)
{
    return &settings;
}

/* End of GSRAO_Config.cxx */
