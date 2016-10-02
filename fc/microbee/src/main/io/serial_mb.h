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
    uint8_t checksum;
} mbspPort_t;

void mbspInit(void);
void mbspPrint(const char *str);
void mbspPrintf(const char *fmt, ...);
void mbspSendMeasurements(void);
void mbspSendHeartBeat(void);

uint16_t* mb_GetBatteryVoltage(void);

#endif
