/**
 * PPM encoder
 *          serial protocol
 * This project receives command about RC vaules from PC via serial port and g-
 * enerate PPM signals for every UAV. The signals are connected to TX modules.
 *
 * Author: Roice Luo (Bing Luo)
 * Date:   2016-05-04 create this file
 */
#include <stdbool.h>
#include "stm32f1xx_hal.h"
#include "serial.h"

static sppState_e spp_state = IDLE;
static uint8_t spp_checksum;
static uint8_t spp_count;
static uint16_t spp_data[4*SPP_CHANNELS_IN_PPM_SIGNAL];

/*
 * Frame structure:
 * '$P' + XXXXXXXX + XXXXXXXX + XXXXXXXX + XXXXXXXX + parity byte
 *         1st PPM    2nd PPM    3rd PPM    4th PPM
 *                  8 channel each PPM signal
 *                       2 bytes each channel
 * 2x8x4+2+1 = 67 bytes total for 8 channels each PPM signal
 * XXX:
 *     Only 4 channels used at present, so the total bytes for
 *     a PPM frame is 2x4x4+2+1 = 35 bytes
 * XX is a 16-bit unsigned short number, LE
 */
bool sppProcessReceivedData(uint8_t c)
{
    if (spp_state == IDLE) {
        if (c == '$')
            spp_state = HEADER_START;
    }
    else if (spp_state == HEADER_START) {
        if (c == 'P') {
            spp_state = HEADER_P;
            spp_checksum = 0;
            spp_count = 4*SPP_CHANNELS_IN_PPM_SIGNAL*sizeof(uint16_t);
        }
        else
            spp_state = IDLE;
    }
    else if (spp_state == HEADER_P) {
        spp_checksum ^= c;
        spp_count --;
        if (spp_count == 0)
            spp_state = DATA; // received data, need checksum
    }
    else if (spp_state == DATA) {
        spp_state = IDLE;
        if (spp_checksum == c) { // data OK 
            return true;
        } 
    }
    return false;
}

bool sppFrameParsing(uint8_t* frame, uint8_t len)
{
    for (uint8_t i = 0; i < len; i++ )
    {
        if (sppProcessReceivedData( *(frame+i) ))
        {
            // save data
            if (i >= 4*SPP_CHANNELS_IN_PPM_SIGNAL*sizeof(uint16_t))
            {
                for (uint8_t j = 0; j < 4; j++)
                    for (uint8_t k = 0; k < SPP_CHANNELS_IN_PPM_SIGNAL; k++)
                        spp_data[j*SPP_CHANNELS_IN_PPM_SIGNAL+k] = *((uint16_t*)(frame+i-sizeof(uint16_t)*((4-j)*SPP_CHANNELS_IN_PPM_SIGNAL-k)));
                return true;
            }
            else
                break;
        }
    }
    return false;
}

uint16_t* sppGetChannelData(void)
{
    return spp_data;
}
