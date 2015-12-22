/*
 * File Name: daq.ino
 * This file implements ADC and I2C to realize a data acquisition device
 * An analog chanel is sampled and trans out data via I2c 
 * 
 * Changelog:
 *          AUTHOR      DATE                LOG
 *          Roice       13 June, 2015       Create
 */

/* Hardware Description
 * Arduino Micro
 * CPU: Atmega32u4
 * Power: 5V
 * Frequency: If the system is powered up by 3.3V, the Freqency must be
 * limited to 8 MHz
 */

#include "MsTimer2.h"           // Need install MsTimer2 lib to {arduino-path}/libraries/, for timer interrupts
#include "Wire.h"               // Wire library, for I2C

#define     DAQ_FREQUENCY   10  // DAQ frequency = 10 Hz

/* global parameters */
int daq_value;

/* setting up */
void setup()
{
    /* ADC setting */
    analogReference(DEFAULT);   // Ref for ADC, DEFAULT INTERNAL EXTERNAL

    /* I2C setting */
    Wire.begin(8);                  // join the I2C bus as a slave with address #8
    Wire.onRequest(requestData);    // register function 'requestData' to be called when a master request data from this device

    /* Timer Interrupt setting, for periodic sampling */
    MsTimer2::set(int(1000/DAQ_FREQUENCY), daq);  // interrupt every 1000/DAQ_FREQUENCY ms and execute function 'daq'
    MsTimer2::start();          // start timing
}

/* main loop */
void loop()
{
    delay(2000);
}

/* function prototypes */
/* daq function to be called when MsTimer2 overflows */
void daq()
{
    daq_value = analogRead(A0);     // Read voltage from port A0

    /* calculate voltage
     * The reference and system voltage is 5V
     */
    //double vol = n * (1.1 / 1024.0*100);
}

/* I2C trans function to be called when a master request data from this device */
void requestData()
{
    /* convert int to char buf */
    char buf[sizeof(int)], i;
    for (i=0; i<sizeof(int); i++)
    {
        buf[i] = daq_value >> (4*i); // little endian
    }

    /* trans buf */
    Wire.write(buf, sizeof(buf));
}





