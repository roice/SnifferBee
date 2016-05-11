/*
 * MicroBee Serial Protocol
 *      for transmitting sensor values to ground station
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.10  Create this file
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "build_config.h"

#include "drivers/buf_writer.h"

#include "io/serial_mb.h"

#include "common/printf.h"

STATIC_UNIT_TESTED mbspPort_t mbspPort;

static bufWriter_t *mbspWriter;
static uint8_t mbspWriteBuffer[sizeof(*mbspWriter) + 16];

void mbspInit(serialPort_t *serialPort)
{
    memset(&mbspPort, 0x00, sizeof(mbspPort));
    mbspPort.port = serialPort;

    setPrintfSerialPort(mbspPort.port);
    mbspWriter = bufWriterInit(mbspWriteBuffer, sizeof(mbspWriteBuffer), 
            (bufWrite_t)serialWriteBufShim, mbspPort.port);
}

void mbspPrint(const char *str)
{
    // make sure mbsp is initilized
    if (mbspWriter && mbspPort.port) {
        while (*str)
            bufWriterAppend(mbspWriter, *str++);
        bufWriterFlush(mbspWriter);
    }
}

static void mbspPutp(void *p, char ch)
{
    bufWriterAppend(p, ch);
}

void mbspPrintf(const char *fmt, ...)
{
    // make sure mbsp is initilized
    if (mbspWriter && mbspPort.port) {
        va_list va;
        va_start(va, fmt);
        tfp_format(mbspWriter, mbspPutp, fmt, va);
        va_end(va);
    }
}
