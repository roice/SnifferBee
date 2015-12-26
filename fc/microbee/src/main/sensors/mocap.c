/* This file is part of SuperBee.
 *
 * Motion Capture related.
 *
 * Author       Date        Changelog
 * Roice Luo    2015.07.02  Create
 */
#include <stdbool.h>

#include "platform.h"

#include "common/maths.h"
#include "common/axis.h"

#include "drivers/system.h"
#include "drivers/sensor.h"
#include "drivers/accgyro.h"

#include "io/rc_controls.h"
#include "io/beeper.h"
#include "io/gps.h"
#include "io/gimbal.h"

#include "sensors/mocap.h"
#include "sensors/sensors.h"
#include "sensors/acceleration.h"

#include "flight/mixer.h"
#include "flight/pid.h"
#include "flight/navigation.h"
#include "flight/altitudehold.h"
#include "flight/imu.h"

#include "config/config.h"
#include "config/runtime_config.h"
#include "config/config_profile.h"

#ifdef MOCAP

// extern
extern profile_t *currentProfile;

// Motion Capture Altitude Ready Flag
static bool mocapAltReady = false;
static bool mocapGPSReady = false;
// Motion Capture Altitude data
static int32_t mocapAlt = 0;    // in mm
// Motion Capture virtual GPS coordinate LL
static int32_t mocapGPS_coord[2];

/* Global parameters */
struct mocap_enu_t mocap_enu = {0,0,0,0,false};

#ifdef SB_DEBUG
bool sb_debug_applyAltHold = false;
#endif

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

/* Functions -- R/W/Compute... */
void updateMocap(int32_t e, int32_t n, int32_t u)
{
    mocap_enu.east = e;  // in 0.1mm
    mocap_enu.north = n; // in 0.1mm
    mocap_enu.up = u;    // in 0.1mm
    mocap_enu.time = millis(); // save current time
    mocap_enu.fresh = true;
}

bool mocapUpdatePos(void)
{
    double position_e[3], converted_pos[3];
    if (mocap_enu.fresh != true)
    {// no new data
       clearMocapAltReadyFlag(); 
       return false;
    }

    mocap_enu.fresh = false;
    
/* Altitude */
    mocapAlt = mocap_enu.up / 10;  // mm
    setMocapAltReadyFlag();

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
    position_e[0] = (double)mocap_enu.north / (double)10000.0f;  // lat
    position_e[1] = (double)mocap_enu.east / (double)10000.0f; // lon
    /* convert local ENU to LLH
     * This function is linearized and the position to be converted is limited
     * to no more than 100 m to the original pos llh 0.0 N 0.0E */
    enu2llh(position_e, converted_pos);

    mocapGPS_coord[0] = (int32_t)(converted_pos[0] * (double)100000000.0f);
    mocapGPS_coord[1] = (int32_t)(converted_pos[1] * (double)100000000.0f);

    setMocapGPSReadyFlag();
    return true;
}

int32_t mocapReadAltitude(void) {
    return mocapAlt;
}

int32_t mocapReadGPSLL(uint8_t index) {
    if (index == 0 || index == 1)
        return mocapGPS_coord[index];
    else
        return 0;
}

void updateMocapState(void)
{
    static bool navPosHold = false;

    // Mocap activate
    if (!IS_RC_MODE_ACTIVE(BOXMOCAP)) {
        DISABLE_FLIGHT_MODE(MOCAP_MODE);
        return;
    }

    // Mocap flight mode activate & althold gpshold init
    if (!FLIGHT_MODE(MOCAP_MODE)) {
        ENABLE_FLIGHT_MODE(MOCAP_MODE);

        // for Alititude hold
        AltHold = EstAlt;
        initialThrottleHoldrcCommand = rcCommand[THROTTLE];
        initialThrottleHoldrcData = rcData[THROTTLE];
        errorVelocityI = 0;
        altHoldThrottleAdjustment = 0;
    }

    // for Lat/Lon hold
    if (FLIGHT_MODE(MOCAP_MODE))
    {
        if (areSticksInApModePosition(currentProfile->gpsProfile.ap_mode))
        {// stick in middle position
            if (navPosHold == false)
            {// entering Position hold mode
                GPS_hold[LAT] = GPS_coord[LAT];
                GPS_hold[LON] = GPS_coord[LON];
                GPS_set_next_wp(&GPS_hold[LAT], &GPS_hold[LON]);
                nav_mode = NAV_MODE_POSHOLD;
                GPS_reset_nav();
                navPosHold = true;
                // beep
                beeper(BEEPER_ARMING_GPS_FIX);
            }
        }
        else
        {// manually control roll pitch
            nav_mode = NAV_MODE_NONE;
            GPS_reset_nav();
            navPosHold = false;
        }
    } 
}

/* Functions -- Flags */
bool isMocapAltReady(void) {
    return mocapAltReady;
}

void setMocapAltReadyFlag(void) {
    mocapAltReady = true;
}

void clearMocapAltReadyFlag(void) {
    mocapAltReady = false;
}

bool isMocapGPSReady(void) {
    return mocapGPSReady;
}

void setMocapGPSReadyFlag(void) {
    mocapGPSReady = true;
}

void clearMocapGPSReadyFlag(void) {
    mocapGPSReady = false;
}

#endif
