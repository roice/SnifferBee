/*
 * MicroBee Serial Protocol
 *      for transmitting sensor values to ground station
 *      
 *      MicroBee use USART1 to control RF
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

#include <platform.h>
#include "build_config.h"

#include "drivers/buf_writer.h"
#include "drivers/adc_mb.h"
#include "drivers/serial.h"

#include "io/serial.h"
#include "io/serial_mb.h"

#include "flight/mixer.h"

#include "common/printf.h"

#include "config/runtime_config.h"


#define MBSP_ADDRESS_GS         0   // address of ground station
#define MBSP_ADDRESS_MB_1       1   // address of Microbee No. 1
#define MBSP_ADDRESS_MB_2       2   // address of Microbee No. 2
#define MBSP_ADDRESS_MB_3       3   // address of Microbee No. 3
#define MBSP_ADDRESS_MB_4       4   // address of Microbee No. 4

#define MBSP_CMD_STATUS         101 // status of MicroBee
#define MBSP_CMD_MEASUREMENTS   102 // readings of (three)gas sensors & motor values (if required)

#define MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE    // send motor values

volatile mbspPort_t   mbspPort;

static bufWriter_t *mbspWriter;
static uint8_t mbspWriteBuffer[sizeof(bufWriter_t) + 30];

static uint16_t battery_volt = 0;

void mbspInit(void)
{
    // FUNCTION_TELEMETRY_FRSKY is used to distinguish with MSP,
    // telemetry is not used
    serialPort_t* serialPort = openSerialPort(SERIAL_PORT_USART1, FUNCTION_TELEMETRY_FRSKY, NULL, 57600, MODE_RXTX, SERIAL_NOT_INVERTED);
    if (!serialPort)
        return;

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

static void serialize8(uint8_t a)
{
    bufWriterAppend(mbspWriter, a);
    mbspPort.checksum ^= a;
}

static void serialize16(uint16_t a)
{
    serialize8((uint8_t)(a >> 0));
    serialize8((uint8_t)(a >> 8));
}
/*
static void serialize32(uint32_t a)
{
    serialize16((uint16_t)(a >> 0));
    serialize16((uint16_t)(a >> 16));
}
*/
void mbspSendMeasurements(void)
{
    static uint16_t measurement_count = 0;
    
    mbspPort.checksum = 0;

    // send addr and channel of ground station RF
    // E53-TTL-100 module, EBYTE Co.Ltd., Chengdu
    // 0x00 0x64 0x82 by default
    bufWriterAppend(mbspWriter, 0x00);
    bufWriterAppend(mbspWriter, 120+MICROBEE_DEVICE_NUMBER);  // address of GS
    bufWriterAppend(mbspWriter, 115+10*(MICROBEE_DEVICE_NUMBER-1));  // 115 + 10*(MICROBEE_DEVICE_NUMBER-1)

    // '$B>' header of mbsp message
    bufWriterAppend(mbspWriter, '$');
    bufWriterAppend(mbspWriter, 'B');

    // address of device which should receive this message
    serialize8(MBSP_ADDRESS_GS);

    // address of device which sent this message (self)
    serialize8(MICROBEE_DEVICE_NUMBER);

    // length of data (bytes)
#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
    serialize8(MB_ADC_CHANNEL_COUNT*sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint8_t) + 4*sizeof(uint16_t));
#else
    serialize8(MB_ADC_CHANNEL_COUNT*sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint8_t));
#endif

    // command, send gas sensor readings
    serialize8(MBSP_CMD_MEASUREMENTS);

    // gas sensor data
    serialize16(mb_adcGetChannel(ADC_GAS_SENSOR_FRONT));
    serialize16(mb_adcGetChannel(ADC_GAS_SENSOR_REAR_LEFT));
    serialize16(mb_adcGetChannel(ADC_GAS_SENSOR_REAR_RIGHT));

    // Battery Voltage
    serialize16(battery_volt);

    // ARM/DISARM info
    if (ARMING_FLAG(ARMED))
        serialize8(1);
    else
        serialize8(0);

    // count number, in case data missing
    serialize16(measurement_count++);

#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
    for (uint16_t i = 0; i < 4; i++)
        serialize16(motor[i]);
#endif 

    // checksum
    bufWriterAppend(mbspWriter, mbspPort.checksum);

    // send message
    bufWriterFlush(mbspWriter);
}

/*
void mbspSendHeartBeat(void)
{
    mbspPort.checksum = 0;

    // send addr and channel of ground station RF
    // E53-TTL-100 module, EBYTE Co.Ltd., Chengdu
    // 0x00 0x64 0x82 by default
    bufWriterAppend(mbspWriter, 0x00);
    bufWriterAppend(mbspWriter, 0x64);
    bufWriterAppend(mbspWriter, 0x82);

    // '$B>' header of mbsp message
    bufWriterAppend(mbspWriter, '$');
    bufWriterAppend(mbspWriter, 'B');
    
    // address of device which should receive this message
    serialize8(MBSP_ADDRESS_GS);

    // address of device which sent this message (self)
    serialize8(MICROBEE_DEVICE_NUMBER);

    // length of data (bytes)
    serialize8(sizeof(uint8_t)+sizeof(uint16_t)); // ARM/DISARM and Battery Voltage

    // command, send microbee status
    serialize8(MBSP_CMD_STATUS);

    // ARM/DISARM info
    if (ARMING_FLAG(ARMED))
        serialize8(1);
    else
        serialize8(0);

    // Battery Voltage
    serialize16(battery_volt);
    
    // checksum
    bufWriterAppend(mbspWriter, mbspPort.checksum);

    // send message
    bufWriterFlush(mbspWriter);
}
*/

// get battery voltage
uint16_t* mb_GetBatteryVoltage(void)
{
    return &battery_volt;
}
