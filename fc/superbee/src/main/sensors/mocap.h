/*
 * This file is part of SuperBee.
 *
 * Motion Capture related.
 *
 * Author       Date        Changelog
 * Roice Luo    2015.07.02  Create
 * Roice Luo    2017.04.05  Modified
 */

#pragma once

/* Mocap data struct */
typedef struct {
    // local ENU, accuracy: 0.1 milimeters
    // example: if up coordinate is -330.6 mm
    //          up == -3306
    int32_t enu[3];
    int32_t vel[3];
    int32_t acc[3];
    int32_t att[3];

    uint32_t time;  // ms
} Mocap_Data_t;

bool mocap_update_data(void);
void mocap_update_state(void);
Mocap_Data_t* mocap_get_data(void);
int32_t mocap_get_alt(void);
int32_t* mocap_get_gpsll(void);
bool mocap_is_alt_ready(void);
void mocap_set_alt_ready_flag(void);
void mocap_clear_alt_ready_flag(void);
bool mocap_is_gps_ready(void);
void mocap_set_gps_ready_flag(void);
void mocap_clear_gps_ready_flag(void);
