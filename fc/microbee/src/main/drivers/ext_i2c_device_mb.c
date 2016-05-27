/*
 * Driver for communicating with external i2c device I2C port
 *
 * MicroBee use PB10 and PB11 (I2C 2)
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.12  create this file
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <platform.h>
#include "build_config.h"

#include "drivers/bus_i2c.h"

#define MB_I2C_TIMEOUT          7200    // 100 us
#define EXT_DEVICE_I2C_ADDR     0x26    // address of external i2c device
#define MB_I2C_CMD_BAT_VOLT     100     // command of requesting bat volt

static uint16_t adc_value;


// update battery voltage
void mb_BatVoltUpdate(void)
{
    bool ack;
    uint8_t data[2]; // arduino's int is 2 bytes, and ADC is 10 bits
    
        ack = i2cRead(EXT_DEVICE_I2C_ADDR, MB_I2C_CMD_BAT_VOLT, 2, data);
        if (ack) {
            // get adc value 0-1023
            adc_value = 0;
            adc_value |= ((uint16_t)(data[0])) & 0x00FF;
            adc_value |= (((uint16_t)(data[1])) << 8) & 0x0300; // 10 bit ADC
            // convert to battery volt
            float battery_volt = adc_value*5.0f/1024.0f; // ref is 5 V
        }
}


