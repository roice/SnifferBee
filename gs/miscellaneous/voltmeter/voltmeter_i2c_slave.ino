/* Voltage meter, I2C slave device
 *  This routine runs on an ATTINY85 mcu
 *  Setup:
 *  ATtiny PB0    I2C_SDA  
 *  ATtiny PB2    I2C_SCL
 *  ATtiny PB4    Analog input
 *  Current Rx & Tx buffers set at 32 bytes - see usiTwiSlave.h
 *  
 *  Author:
 *      Roice Luo (Bing Luo)
 *  Date:
 *      2016.05.12  create this file
 */

#include "TinyWireS.h"                  // wrapper class for I2C slave routines

// The default buffer size, Can't recall the scope of defines right now
#ifndef TWI_RX_BUFFER_SIZE
#define TWI_RX_BUFFER_SIZE ( 16 )
#endif

#define I2C_SLAVE_ADDR        0x26      // i2c slave address (38)
#define REG_BAT_VOLT          100       // command request battery voltage
#define LED_PIN               1         // ATtiny PB1
#define BAT_VOLT_PIN          4         // ATtiny PB4
#define ADC_CHANNEL_BAT_VOLT  2         // ATtiny PB4, ADC channel 2

bool led_status = false;      // true for ON, false for OFF
byte commandReceived;
int adc_value;        // value of adc channel, 2 bytes

/**
 * This is called for each read request we receive, never put more than one byte of data (with TinyWireS.send) to the 
 * send-buffer when using this callback
 */
void requestEvent()
{  
  if (commandReceived > 0) {
    switch (commandReceived) {
      case REG_BAT_VOLT:
        // get ADC channel value
        adc_value = analogRead(ADC_CHANNEL_BAT_VOLT);
        // transmit this to master
        for (uint8_t i = 0; i < 2; i++) // int is 2 bytes
          TinyWireS.send((adc_value >> 8*i) & 0xFF);
        Toggle(LED_PIN); // indicating that a command has been processed
        break;
      default:
        break;
    }
    commandReceived = 0;
  }
}

/**
 * The I2C data received -handler
 *
 * This needs to complete before the next incoming transaction (start, data, restart/stop) on the bus does
 * so be quick, set flags for long running tasks to be called from the mainloop instead of running them directly,
 */
void receiveEvent(uint8_t bytesReceived)
{
  if (bytesReceived == 1) {// only reg address
    commandReceived = TinyWireS.receive();
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);             // for general DEBUG use
  pinMode(BAT_VOLT_PIN, INPUT);         // for measuring battery voltage
  Blink(LED_PIN, 2);                    // show it's alive
  TinyWireS.begin(I2C_SLAVE_ADDR);      // init I2C Slave mode
  TinyWireS.onReceive(receiveEvent);
  TinyWireS.onRequest(requestEvent);
  commandReceived = 0;
}

void loop() {
  TinyWireS_stop_check();
}

void Toggle(byte led){
  if (led_status){
    digitalWrite(led,LOW);
    led_status = false;
  }
  else{
    digitalWrite(led,HIGH);
    led_status = true;
  }
}

void Blink(byte led, byte times)
{
  for (byte i=0; i< times; i++)
  {
    digitalWrite(led,HIGH);
    led_status = true;
    delay (250);
    digitalWrite(led,LOW);
    led_status = false;
    delay (175);
  }
}
