/*
 * MicroBee Serial Protocol
 *      for transmitting sensor values to ground station
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.10  Create this file
 */

#ifndef SERIAL_MB_H
#define SERIAL_MB_H

#include "drivers/serial.h"

typedef enum {
    MBSP_IDLE,
    MBSP_START,
    MBSP_B,
    MBSP_ARROW,
    MBSP_SIZE,
    MBSP_CMD,
    MBSP_CMD_RCVD
} mbspState_e;

#define MBSP_PORT_INBUF_SIZE 64

typedef struct mbspPort_s {
    serialPort_t *port; // null when port unused.
    uint8_t offset;
    uint8_t dataSize;
    uint8_t checksum;
    uint8_t indRX;
    uint8_t inBuf[MBSP_PORT_INBUF_SIZE];
    mbspState_e c_state;
    uint8_t cmdMBSP;
} mbspPort_t;

void mbspInit(serialPort_t *serialPort);
void mbspPrint(const char *str);
void mbspPrintf(const char *fmt, ...);

#endif
