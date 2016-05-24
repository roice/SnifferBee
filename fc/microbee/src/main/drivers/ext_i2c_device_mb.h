/*
 * Driver for communicating with external i2c device via I2C port
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.12  create this file
 */
#pragma once

void mb_BatVoltUpdate(void);
float mb_GetBatteryVoltage(void);
