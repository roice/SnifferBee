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
    char name[256];

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
        settings.robot.type_of_robot = pt.get<int>("Robot.type_of_robot");
        settings.robot.num_of_robots = pt.get<int>("Robot.num_of_robots");
        settings.robot.ppm_serial_port_path = pt.get<std::string>("Robot.ppm_serial_port_path");
        settings.robot.dnet_serial_port_path[0] = pt.get<std::string>("Robot.dnet_serial_port_path[0]");
settings.robot.dnet_serial_port_path[1] = pt.get<std::string>("Robot.dnet_serial_port_path[1]");
settings.robot.dnet_serial_port_path[2] = pt.get<std::string>("Robot.dnet_serial_port_path[2]");
settings.robot.dnet_serial_port_path[3] = pt.get<std::string>("Robot.dnet_serial_port_path[3]");
        settings.robot.dnet_udp_send_to_robot = pt.get<std::string>("Robot.dnet_udp_send_to_robot");
        settings.robot.dnet_udp_recv_from_robot = pt.get<std::string>("Robot.dnet_udp_recv_from_robot");
        // PID
        settings.robot.pidProfile[0].P[PIDALT] = pt.get<float>("Robot.pidProfile_P_ALT_of_robot_1");
        settings.robot.pidProfile[0].P[PIDVEL] = pt.get<float>("Robot.pidProfile_P_VEL_of_robot_1");
        settings.robot.pidProfile[0].I[PIDVEL] = pt.get<float>("Robot.pidProfile_I_VEL_of_robot_1");
        settings.robot.pidProfile[0].D[PIDVEL] = pt.get<float>("Robot.pidProfile_D_VEL_of_robot_1");
        settings.robot.pidProfile[0].P[PIDPOS] = pt.get<float>("Robot.pidProfile_P_POS_of_robot_1");
        settings.robot.pidProfile[0].P[PIDPOSR] = pt.get<float>("Robot.pidProfile_P_POSR_of_robot_1");
        settings.robot.pidProfile[0].I[PIDPOSR] = pt.get<float>("Robot.pidProfile_I_POSR_of_robot_1");
        settings.robot.pidProfile[0].D[PIDPOSR] = pt.get<float>("Robot.pidProfile_D_POSR_of_robot_1");
        settings.robot.pidProfile[0].P[PIDMAG] = pt.get<float>("Robot.pidProfile_P_MAG_of_robot_1");
        settings.robot.pidProfile[0].I[PIDMAG] = pt.get<float>("Robot.pidProfile_I_MAG_of_robot_1");
        settings.robot.pidProfile[0].D[PIDMAG] = pt.get<float>("Robot.pidProfile_D_MAG_of_robot_1");
 
        settings.robot.pidProfile[1].P[PIDALT] = pt.get<float>("Robot.pidProfile_P_ALT_of_robot_2");
        settings.robot.pidProfile[1].P[PIDVEL] = pt.get<float>("Robot.pidProfile_P_VEL_of_robot_2");
        settings.robot.pidProfile[1].I[PIDVEL] = pt.get<float>("Robot.pidProfile_I_VEL_of_robot_2");
        settings.robot.pidProfile[1].D[PIDVEL] = pt.get<float>("Robot.pidProfile_D_VEL_of_robot_2");
        settings.robot.pidProfile[1].P[PIDPOS] = pt.get<float>("Robot.pidProfile_P_POS_of_robot_2");
        settings.robot.pidProfile[1].P[PIDPOSR] = pt.get<float>("Robot.pidProfile_P_POSR_of_robot_2");
        settings.robot.pidProfile[1].I[PIDPOSR] = pt.get<float>("Robot.pidProfile_I_POSR_of_robot_2");
        settings.robot.pidProfile[1].D[PIDPOSR] = pt.get<float>("Robot.pidProfile_D_POSR_of_robot_2");
        settings.robot.pidProfile[1].P[PIDMAG] = pt.get<float>("Robot.pidProfile_P_MAG_of_robot_2");
        settings.robot.pidProfile[1].I[PIDMAG] = pt.get<float>("Robot.pidProfile_I_MAG_of_robot_2");
        settings.robot.pidProfile[1].D[PIDMAG] = pt.get<float>("Robot.pidProfile_D_MAG_of_robot_2");

        settings.robot.pidProfile[2].P[PIDALT] = pt.get<float>("Robot.pidProfile_P_ALT_of_robot_3");
        settings.robot.pidProfile[2].P[PIDVEL] = pt.get<float>("Robot.pidProfile_P_VEL_of_robot_3");
        settings.robot.pidProfile[2].I[PIDVEL] = pt.get<float>("Robot.pidProfile_I_VEL_of_robot_3");
        settings.robot.pidProfile[2].D[PIDVEL] = pt.get<float>("Robot.pidProfile_D_VEL_of_robot_3");
        settings.robot.pidProfile[2].P[PIDPOS] = pt.get<float>("Robot.pidProfile_P_POS_of_robot_3");
        settings.robot.pidProfile[2].P[PIDPOSR] = pt.get<float>("Robot.pidProfile_P_POSR_of_robot_3");
        settings.robot.pidProfile[2].I[PIDPOSR] = pt.get<float>("Robot.pidProfile_I_POSR_of_robot_3");
        settings.robot.pidProfile[2].D[PIDPOSR] = pt.get<float>("Robot.pidProfile_D_POSR_of_robot_3");
        settings.robot.pidProfile[2].P[PIDMAG] = pt.get<float>("Robot.pidProfile_P_MAG_of_robot_3");
        settings.robot.pidProfile[2].I[PIDMAG] = pt.get<float>("Robot.pidProfile_I_MAG_of_robot_3");
        settings.robot.pidProfile[2].D[PIDMAG] = pt.get<float>("Robot.pidProfile_D_MAG_of_robot_3");
       
        settings.robot.pidProfile[3].P[PIDALT] = pt.get<float>("Robot.pidProfile_P_ALT_of_robot_4");
        settings.robot.pidProfile[3].P[PIDVEL] = pt.get<float>("Robot.pidProfile_P_VEL_of_robot_4");
        settings.robot.pidProfile[3].I[PIDVEL] = pt.get<float>("Robot.pidProfile_I_VEL_of_robot_4");
        settings.robot.pidProfile[3].D[PIDVEL] = pt.get<float>("Robot.pidProfile_D_VEL_of_robot_4");
        settings.robot.pidProfile[3].P[PIDPOS] = pt.get<float>("Robot.pidProfile_P_POS_of_robot_4");
        settings.robot.pidProfile[3].P[PIDPOSR] = pt.get<float>("Robot.pidProfile_P_POSR_of_robot_4");
        settings.robot.pidProfile[3].I[PIDPOSR] = pt.get<float>("Robot.pidProfile_I_POSR_of_robot_4");
        settings.robot.pidProfile[3].D[PIDPOSR] = pt.get<float>("Robot.pidProfile_D_POSR_of_robot_4");
        settings.robot.pidProfile[3].P[PIDMAG] = pt.get<float>("Robot.pidProfile_P_MAG_of_robot_4");
        settings.robot.pidProfile[3].I[PIDMAG] = pt.get<float>("Robot.pidProfile_I_MAG_of_robot_4");
        settings.robot.pidProfile[3].D[PIDMAG] = pt.get<float>("Robot.pidProfile_D_MAG_of_robot_4");
        // ADRC
        settings.robot.adrcProfile[0].w0[ADRCALT] = pt.get<float>("Robot.adrcProfile_w0_ALT_of_robot_1");
        settings.robot.adrcProfile[0].kp[ADRCALT] = pt.get<float>("Robot.adrcProfile_kp_ALT_of_robot_1");
        settings.robot.adrcProfile[0].kd[ADRCALT] = pt.get<float>("Robot.adrcProfile_kd_ALT_of_robot_1");
        settings.robot.adrcProfile[0].w0[ADRCPOS] = pt.get<float>("Robot.adrcProfile_w0_POS_of_robot_1");
        settings.robot.adrcProfile[0].kp[ADRCPOS] = pt.get<float>("Robot.adrcProfile_kp_POS_of_robot_1");
        settings.robot.adrcProfile[0].kd[ADRCPOS] = pt.get<float>("Robot.adrcProfile_kd_POS_of_robot_1");
        settings.robot.adrcProfile[0].w0[ADRCMAG] = pt.get<float>("Robot.adrcProfile_w0_MAG_of_robot_1");
        settings.robot.adrcProfile[0].kp[ADRCMAG] = pt.get<float>("Robot.adrcProfile_kp_MAG_of_robot_1");
        settings.robot.adrcProfile[0].kd[ADRCMAG] = pt.get<float>("Robot.adrcProfile_kd_MAG_of_robot_1");
        // System
        settings.system.robot_panel_opened = pt.get<bool>("System.robot_panel_opened");
        settings.system.result_panel_opened = pt.get<bool>("System.result_panel_opened");
        settings.system.remoter_panel_opened = pt.get<bool>("System.remoter_panel_opened");
        // Miscellaneous
        for (int idx = 0; idx < SERIAL_MAX_ANEMOMETERS; idx++) {
            snprintf(name, sizeof(name), "Miscellaneous.serial_port_path_anemometer_%d", idx+1);
            settings.miscellaneous.anemometer_serial_port_path[idx] = pt.get<std::string>(name);
            snprintf(name, sizeof(name), "Miscellaneous.type_anemometer_%d", idx+1);
            settings.miscellaneous.anemometer_type[idx] = pt.get<std::string>(name);
        }
        settings.miscellaneous.num_of_anemometers = pt.get<int>("Miscellaneous.num_of_anemometers");
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
    pt.put("Robot.type_of_robot", settings.robot.type_of_robot);
    pt.put("Robot.num_of_robots", settings.robot.num_of_robots);
    pt.put("Robot.ppm_serial_port_path", settings.robot.ppm_serial_port_path);
    pt.put("Robot.dnet_serial_port_path[0]", settings.robot.dnet_serial_port_path[0]);
    pt.put("Robot.dnet_serial_port_path[1]", settings.robot.dnet_serial_port_path[1]);
    pt.put("Robot.dnet_serial_port_path[2]", settings.robot.dnet_serial_port_path[2]);
    pt.put("Robot.dnet_serial_port_path[3]", settings.robot.dnet_serial_port_path[3]);
    pt.put("Robot.dnet_udp_send_to_robot", settings.robot.dnet_udp_send_to_robot);
    pt.put("Robot.dnet_udp_recv_from_robot", settings.robot.dnet_udp_recv_from_robot);
    // PID
    pt.put("Robot.pidProfile_P_ALT_of_robot_1", settings.robot.pidProfile[0].P[PIDALT]);
    pt.put("Robot.pidProfile_P_VEL_of_robot_1", settings.robot.pidProfile[0].P[PIDVEL]);
    pt.put("Robot.pidProfile_I_VEL_of_robot_1", settings.robot.pidProfile[0].I[PIDVEL]);
    pt.put("Robot.pidProfile_D_VEL_of_robot_1", settings.robot.pidProfile[0].D[PIDVEL]);
    pt.put("Robot.pidProfile_P_POS_of_robot_1", settings.robot.pidProfile[0].P[PIDPOS]);
    pt.put("Robot.pidProfile_P_POSR_of_robot_1", settings.robot.pidProfile[0].P[PIDPOSR]);
    pt.put("Robot.pidProfile_I_POSR_of_robot_1", settings.robot.pidProfile[0].I[PIDPOSR]);
    pt.put("Robot.pidProfile_D_POSR_of_robot_1", settings.robot.pidProfile[0].D[PIDPOSR]);
    pt.put("Robot.pidProfile_P_MAG_of_robot_1", settings.robot.pidProfile[0].P[PIDMAG]);
    pt.put("Robot.pidProfile_I_MAG_of_robot_1", settings.robot.pidProfile[0].I[PIDMAG]);
    pt.put("Robot.pidProfile_D_MAG_of_robot_1", settings.robot.pidProfile[0].D[PIDMAG]);
    
    pt.put("Robot.pidProfile_P_ALT_of_robot_2", settings.robot.pidProfile[1].P[PIDALT]);
    pt.put("Robot.pidProfile_P_VEL_of_robot_2", settings.robot.pidProfile[1].P[PIDVEL]);
    pt.put("Robot.pidProfile_I_VEL_of_robot_2", settings.robot.pidProfile[1].I[PIDVEL]);
    pt.put("Robot.pidProfile_D_VEL_of_robot_2", settings.robot.pidProfile[1].D[PIDVEL]);
    pt.put("Robot.pidProfile_P_POS_of_robot_2", settings.robot.pidProfile[1].P[PIDPOS]);
    pt.put("Robot.pidProfile_P_POSR_of_robot_2", settings.robot.pidProfile[1].P[PIDPOSR]);
    pt.put("Robot.pidProfile_I_POSR_of_robot_2", settings.robot.pidProfile[1].I[PIDPOSR]);
    pt.put("Robot.pidProfile_D_POSR_of_robot_2", settings.robot.pidProfile[1].D[PIDPOSR]);
    pt.put("Robot.pidProfile_P_MAG_of_robot_2", settings.robot.pidProfile[1].P[PIDMAG]);
    pt.put("Robot.pidProfile_I_MAG_of_robot_2", settings.robot.pidProfile[1].I[PIDMAG]);
    pt.put("Robot.pidProfile_D_MAG_of_robot_2", settings.robot.pidProfile[1].D[PIDMAG]);

    pt.put("Robot.pidProfile_P_ALT_of_robot_3", settings.robot.pidProfile[2].P[PIDALT]);
    pt.put("Robot.pidProfile_P_VEL_of_robot_3", settings.robot.pidProfile[2].P[PIDVEL]);
    pt.put("Robot.pidProfile_I_VEL_of_robot_3", settings.robot.pidProfile[2].I[PIDVEL]);
    pt.put("Robot.pidProfile_D_VEL_of_robot_3", settings.robot.pidProfile[2].D[PIDVEL]);
    pt.put("Robot.pidProfile_P_POS_of_robot_3", settings.robot.pidProfile[2].P[PIDPOS]);
    pt.put("Robot.pidProfile_P_POSR_of_robot_3", settings.robot.pidProfile[2].P[PIDPOSR]);
    pt.put("Robot.pidProfile_I_POSR_of_robot_3", settings.robot.pidProfile[2].I[PIDPOSR]);
    pt.put("Robot.pidProfile_D_POSR_of_robot_3", settings.robot.pidProfile[2].D[PIDPOSR]);
    pt.put("Robot.pidProfile_P_MAG_of_robot_3", settings.robot.pidProfile[2].P[PIDMAG]);
    pt.put("Robot.pidProfile_I_MAG_of_robot_3", settings.robot.pidProfile[2].I[PIDMAG]);
    pt.put("Robot.pidProfile_D_MAG_of_robot_3", settings.robot.pidProfile[2].D[PIDMAG]);

    pt.put("Robot.pidProfile_P_ALT_of_robot_4", settings.robot.pidProfile[3].P[PIDALT]);
    pt.put("Robot.pidProfile_P_VEL_of_robot_4", settings.robot.pidProfile[3].P[PIDVEL]);
    pt.put("Robot.pidProfile_I_VEL_of_robot_4", settings.robot.pidProfile[3].I[PIDVEL]);
    pt.put("Robot.pidProfile_D_VEL_of_robot_4", settings.robot.pidProfile[3].D[PIDVEL]);
    pt.put("Robot.pidProfile_P_POS_of_robot_4", settings.robot.pidProfile[3].P[PIDPOS]);
    pt.put("Robot.pidProfile_P_POSR_of_robot_4", settings.robot.pidProfile[3].P[PIDPOSR]);
    pt.put("Robot.pidProfile_I_POSR_of_robot_4", settings.robot.pidProfile[3].I[PIDPOSR]);
    pt.put("Robot.pidProfile_D_POSR_of_robot_4", settings.robot.pidProfile[3].D[PIDPOSR]);
    pt.put("Robot.pidProfile_P_MAG_of_robot_4", settings.robot.pidProfile[3].P[PIDMAG]);
    pt.put("Robot.pidProfile_I_MAG_of_robot_4", settings.robot.pidProfile[3].I[PIDMAG]);
    pt.put("Robot.pidProfile_D_MAG_of_robot_4", settings.robot.pidProfile[3].D[PIDMAG]);
    // ADRC
    pt.put("Robot.adrcProfile_w0_ALT_of_robot_1", settings.robot.adrcProfile[0].w0[ADRCALT]);
    pt.put("Robot.adrcProfile_kp_ALT_of_robot_1", settings.robot.adrcProfile[0].kp[ADRCALT]);
    pt.put("Robot.adrcProfile_kd_ALT_of_robot_1", settings.robot.adrcProfile[0].kd[ADRCALT]);
    pt.put("Robot.adrcProfile_w0_POS_of_robot_1", settings.robot.adrcProfile[0].w0[ADRCPOS]);
    pt.put("Robot.adrcProfile_kp_POS_of_robot_1", settings.robot.adrcProfile[0].kp[ADRCPOS]);
    pt.put("Robot.adrcProfile_kd_POS_of_robot_1", settings.robot.adrcProfile[0].kd[ADRCPOS]);
    pt.put("Robot.adrcProfile_w0_MAG_of_robot_1", settings.robot.adrcProfile[0].w0[ADRCMAG]);
    pt.put("Robot.adrcProfile_kp_MAG_of_robot_1", settings.robot.adrcProfile[0].kp[ADRCMAG]);
    pt.put("Robot.adrcProfile_kd_MAG_of_robot_1", settings.robot.adrcProfile[0].kd[ADRCMAG]);

    // System
    pt.put("System.robot_panel_opened", settings.system.robot_panel_opened);
    pt.put("System.result_panel_opened", settings.system.result_panel_opened);
    pt.put("System.remoter_panel_opened", settings.system.remoter_panel_opened);
    // Miscellaneous
    char name[256];
    for (int idx = 0; idx < SERIAL_MAX_ANEMOMETERS; idx++) {
        snprintf(name, sizeof(name), "Miscellaneous.serial_port_path_anemometer_%d", idx+1);
        pt.put(name, settings.miscellaneous.anemometer_serial_port_path[idx]);
        snprintf(name, sizeof(name), "Miscellaneous.type_anemometer_%d", idx+1);
        pt.put(name, settings.miscellaneous.anemometer_type[idx]);
    }
    pt.put("Miscellaneous.num_of_anemometers", settings.miscellaneous.num_of_anemometers);
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
    settings.robot.type_of_robot = 1;
    settings.robot.num_of_robots = 1;
    settings.robot.ppm_serial_port_path = "/dev/ttyUSB_GSRAO_PPM";
    settings.robot.dnet_serial_port_path[0] = "/dev/ttyUSB1";
    settings.robot.dnet_serial_port_path[1] = "/dev/ttyUSB2";
    settings.robot.dnet_serial_port_path[2] = "/dev/ttyUSB3";
    settings.robot.dnet_serial_port_path[3] = "/dev/ttyUSB4";
    settings.robot.dnet_udp_send_to_robot = "lo IPv4 127.0.0.1";
    settings.robot.dnet_udp_recv_from_robot = "lo IPv4 127.0.0.1";

    for (char i = 0; i < 4; i++)
    {
        // PID
        settings.robot.pidProfile[i].P[PIDALT] = 1.0;
        settings.robot.pidProfile[i].P[PIDVEL] = 400;
        settings.robot.pidProfile[i].I[PIDVEL] = 10;
        settings.robot.pidProfile[i].D[PIDVEL] = 0.06;
        settings.robot.pidProfile[i].P[PIDPOS] = 0.5;
        settings.robot.pidProfile[i].P[PIDPOSR] = 120;
        settings.robot.pidProfile[i].I[PIDPOSR] = 0.1;
        settings.robot.pidProfile[i].D[PIDPOSR] = 22;
        settings.robot.pidProfile[i].P[PIDMAG] = 30;
        settings.robot.pidProfile[i].I[PIDMAG] = 0.3;
        settings.robot.pidProfile[i].D[PIDMAG] = 0.02;
        // ADRC
        settings.robot.adrcProfile[i].w0[ADRCALT] = 21;
        settings.robot.adrcProfile[i].kp[ADRCALT] = 49;
        settings.robot.adrcProfile[i].kd[ADRCALT] = 14;
        settings.robot.adrcProfile[i].w0[ADRCPOS] = 10;
        settings.robot.adrcProfile[i].kp[ADRCPOS] = 16;
        settings.robot.adrcProfile[i].kd[ADRCPOS] = 8;
        settings.robot.adrcProfile[i].w0[ADRCMAG] = 20;
        settings.robot.adrcProfile[i].kp[ADRCMAG] = 16;
        settings.robot.adrcProfile[i].kd[ADRCMAG] = 16;
    }
    // system
    settings.system.robot_panel_opened = false;
    settings.system.result_panel_opened = false;
    settings.system.remoter_panel_opened = false;
    // miscellaneous
    char name[256];
    for (int i = 0; i < SERIAL_MAX_ANEMOMETERS; i++) {
        snprintf(name, sizeof(name), "/dev/ttyUSB_GSRAO_ANEMOMETER_%d", i+1);
        settings.miscellaneous.anemometer_serial_port_path[i] = name;
        snprintf(name, sizeof(name), "RM Young 3D", i+1);
        settings.miscellaneous.anemometer_type[i] = name;
    }
    settings.miscellaneous.num_of_anemometers = 3;
}

/* get pointer of config data */
GSRAO_Config_t* GSRAO_Config_get_configs(void)
{
    return &settings;
}

/* End of GSRAO_Config.cxx */
