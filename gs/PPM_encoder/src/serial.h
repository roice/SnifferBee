/**
 * PPM encoder
 *          serial protocol for PPM encoder (spp)
 * This project receives command about RC vaules from PC via serial port and g-
 * enerate PPM signals for every UAV. The signals are connected to TX modules.
 *
 * Author: Roice Luo (Bing Luo)
 * Date:   2016-05-04 create this file
 */

#pragma once

#ifndef SPP_CHANNELS_IN_PPM_SIGNAL // throttle roll pitch yaw
#define SPP_CHANNELS_IN_PPM_SIGNAL 4
#endif

// SPP protocol state
typedef enum {
    IDLE,
    HEADER_START,
    HEADER_P,
    DATA
} sppState_e;

bool sppFrameParsing(uint8_t* frame, uint8_t len);
uint16_t* sppGetChannelData(void);
