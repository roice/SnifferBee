/*
 * This file is part of SuperBee.
 *
 * Motion Capture related.
 *
 * Author       Date        Changelog
 * Roice Luo    2015.07.02  Create
 */

#pragma once

#ifdef MOCAP

/* position struct, refreshed by SBSP messages */
struct mocap_enu_t
{
    // local ENU, accuracy: 0.1 milimeters
    // example: if up coordinate is -330.6 mm
    //          up == -3306
    int32_t east;
    int32_t north;
    int32_t up;
    // data time
    // this value will be checked in OPT_GPS_New_Data
    // if it exceeds current time a predefined period, the GPS signal
    // must be lost, the reason is probably the plane moves out of
    // OptiTrack range...
    uint32_t time;  // ms
    bool fresh; // when this data is new, fresh == true
};

#ifdef SB_DEBUG
extern bool sb_debug_applyAltHold;
#endif

void updateMocap(int32_t e, int32_t n, int32_t u);
bool mocapUpdatePos(void);
int32_t mocapReadAltitude(void);
int32_t mocapReadGPSLL(uint8_t index);
void updateMocapState(void);
bool isMocapAltReady(void);
void setMocapAltReadyFlag(void);
void clearMocapAltReadyFlag(void);
bool isMocapGPSReady(void);
void setMocapGPSReadyFlag(void);
void clearMocapGPSReadyFlag(void);

#endif
