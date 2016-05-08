/**
 * PPM encoder
 *          serial protocol
 * This project receives command about RC vaules from PC via serial port and g-
 * enerate PPM signals for every UAV. The signals are connected to TX modules.
 *
 * Author: Roice Luo (Bing Luo)
 * Date:   2016-05-04 create this file
 */
#include "serial.h"

static sppState_e spp_state = IDLE;
static unsigned char spp_checksum;
static unsigned char spp_count;
float spp_data[4]; // 4 channels of PPM signal

/*
 * Frame structure:
 * '$P' + XXXX + XXXX + XXXX + XXXX + parity byte
 * 15 bytes total
 * XXXX is binary form of float number, LE
 */
bool sppProcessReceivedData(unsigned char c)
{
    if (spp_state == IDLE) {
        if (c == '$')
            spp_state = HEADER_START;
    }
    else if (spp_state == HEADER_START) {
        if (c == 'P') {
            spp_state = HEADER_P;
            spp_checksum = 0;
            spp_count = 4*sizeof(float);
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
        if (spp_checksum == c) { // data OK
            spp_state = IDLE;
            return true;
        }
    }
    return false;
}

void sppFrameParsing(unsigned char* frame, unsigned int len)
{
}
