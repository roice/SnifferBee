/* This file is part of SuperBee.
 *
 * Motion Capture related.
 *
 * Author       Date        Changelog
 * Roice Luo    2015.07.02  Create
 * Roice Luo    2017.04.05  Modify
 */
#include <stdbool.h>
#include <stdint.h>

#include "sensors/mocap.h"

#include "fc/runtime_config.h"

// Motion Capture Altitude Ready Flag
static bool mocapAltReady = false;
static bool mocapGPSReady = false;
static bool mocapHeadingReady = false;
// Motion Capture Altitude data
static int32_t mocapAlt = 0;    // in mm
// Motion Capture virtual GPS coordinate LL
static int32_t mocapGPS_coord[2];
// Motion Capture heading
static int32_t mocapHeading = 0;

Mocap_Data_t mocap_data = {0};

/* convert Local ENU to LLH, use with caution! */
static void enu2llh(const double *e, double *pos)
{// original position is 0.00000000N 0.00000000E 0.000m

    /* convert LAT */
    pos[0] = e[0]/111.3194*0.001;
    /* convert LON */
    pos[1] = e[1]/110.5741*0.001;
    /* convert HEI */
    // ...
}

bool mocap_update_data(void)
{ 
    double position_e[3], converted_pos[3];
    
/* Altitude */
    mocapAlt = mocap_data.enu[2] / 10;  // mm
/* Lat/Lon */
    /* Local ENU to LLH */
    // Note: as the type of GPS_coord is int32_t (supreme 2.1*10^9), it's
    // not enough for the accuracy of 10^(-8) degree (approx. 1 mm), so
    // the Local LLH coord should limited below 20, in this case, I choose
    // 0.00000000N 0.00000000E 0.000H, which actually lies on the
    // Guinea Bay (oil rich), Africa.
    // So imagine we are flying above the Guinea Bay~
    // Note: As coordinates conversion is a tough task (matrix, LAPACK,...),
    // so here simply uses linearized function
    /* save opt pos to temp array */
    position_e[0] = (double)mocap_data.enu[1] / (double)10000.0f;  // lat
    position_e[1] = (double)mocap_data.enu[0] / (double)10000.0f; // lon
    /* convert local ENU to LLH
     * This function is linearized and the position to be converted is limited
     * to no more than 100 m to the original pos llh 0.0 N 0.0E */
    enu2llh(position_e, converted_pos);
    mocapGPS_coord[0] = (int32_t)(converted_pos[0] * (double)100000000.0f);
    mocapGPS_coord[1] = (int32_t)(converted_pos[1] * (double)100000000.0f);
/* Heading */
    mocapHeading = mocap_data.att[2];

    mocap_data.time = millis(); // save current time

    return true;
}

void mocap_update_state(void) {
    if (millis() - mocap_data.time > 200) {
        DISABLE_FLIGHT_MODE(MOCAP_MODE);
        mocap_clear_alt_ready_flag();
        mocap_clear_gps_ready_flag();
        mocap_clear_heading_ready_flag();
    }
    else {
        ENABLE_FLIGHT_MODE(MOCAP_MODE);
        mocap_set_alt_ready_flag();
        mocap_set_gps_ready_flag();
        mocap_set_heading_ready_flag();
    }
}

/* Functions -- data */
Mocap_Data_t* mocap_get_data(void) {
    return &mocap_data;
}

int32_t mocap_get_alt(void) {
    return mocapAlt;
}

int32_t* mocap_get_gpsll(void) {
    return mocapGPS_coord;
}

int32_t mocap_get_heading(void) {
    return mocapHeading;
}

/* Functions -- Flags */
bool mocap_is_alt_ready(void) {
    return mocapAltReady;
}

void mocap_set_alt_ready_flag(void) {
    mocapAltReady = true;
}

void mocap_clear_alt_ready_flag(void) {
    mocapAltReady = false;
}

bool mocap_is_gps_ready(void) {
    return mocapGPSReady;
}

void mocap_set_gps_ready_flag(void) {
    mocapGPSReady = true;
}

void mocap_clear_gps_ready_flag(void) {
    mocapGPSReady = false;
}

bool mocap_is_heading_ready(void) {
    return mocapHeadingReady;
}

void mocap_set_heading_ready_flag(void) {
    mocapHeadingReady = true;
}

void mocap_clear_heading_ready_flag(void) {
    mocapHeadingReady = false;
}
