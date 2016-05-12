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

static uint32_t battery_volt;

// update battery voltage
void mb_BatVoltUpdate(void)
{
    bool ack;
    uint8_t data;

    // send command to the device
    //i2cWrite(EXT_DEVICE_I2C_ADDR, MB_I2C_CMD_BAT_VOLT, MB_I2C_CMD_BAT_VOLT);
    // wait for reply
    
        ack = i2cRead(EXT_DEVICE_I2C_ADDR, MB_I2C_CMD_BAT_VOLT, 1, &data);
        if (ack)
            battery_volt = data;
}

// get battery voltage
uint32_t mb_GetBatteryVoltage(void)
{
    return battery_volt;
}
