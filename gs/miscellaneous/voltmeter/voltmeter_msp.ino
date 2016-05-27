/* Voltage meter, send voltage via MSP (protocol)
 *  This routine runs on an ATTINY85 mcu
 *  Setup:
 *  ATtiny PB0    TX  
 *  ATtiny PB2    RX
 *  ATtiny PB4    Analog input
 *  Current Rx & Tx buffers set at 32 bytes - see usiTwiSlave.h
 *  
 *  <SoftSerial> adapted from <SoftwareSerial> for <TinyPinChange> library which allows sharing the Pin Change Interrupt Vector.
 *  Single difference with <SoftwareSerial>: add #include <TinyPinChange.h>  at the top of your sketch.
 *  RC Navy (2012): http://p.loussouarn.free.fr
 *  
 *  Author:
 *      Roice Luo (Bing Luo)
 *  Date:
 *      2016.05.26  create this file
 */

#include <SoftSerial.h>     /* Allows Pin Change Interrupt Vector Sharing */
#include <TinyPinChange.h>  /* Ne pas oublier d'inclure la librairie <TinyPinChange> qui est utilisee par la librairie <RcSeq> */

SoftSerial mySerial(2, 0); // RX, TX

#define SERIAL_BAUD_RATE        9600 //Adjust here the serial rate

#define MSP_BAT_STATUS          230     // command ID for MSP
#define LED_PIN               1         // ATtiny PB1
#define BAT_VOLT_PIN          4         // ATtiny PB4
#define ADC_CHANNEL_BAT_VOLT  2         // ATtiny PB4, ADC channel 2

bool led_status = false;      // true for ON, false for OFF
int adc_value;        // value of adc channel, 2 bytes
byte buf[20];
byte checksum;

void setup() {
  pinMode(LED_PIN, OUTPUT);             // for general DEBUG use
  pinMode(BAT_VOLT_PIN, INPUT);         // for measuring battery voltage
  Blink(LED_PIN, 2);                    // show it's alive
  // Init soft serial
  mySerial.begin(SERIAL_BAUD_RATE);
}

void loop() {
  // get analog channel value
  adc_value = analogRead(ADC_CHANNEL_BAT_VOLT);
  // prepare for MSP frame
  checksum = 0;
  buf[0] = '$';
  buf[1] = 'M';
  buf[2] = '<';
  buf[3] = 2; // 2 bytes of data
  checksum ^= buf[3];
  buf[4] = MSP_BAT_STATUS;
  checksum ^= buf[4];
  buf[5] = (byte)(adc_value & 0x00FF);
  checksum ^= buf[5];
  buf[6] = (byte)((adc_value >> 8) & 0x00FF);
  checksum ^= buf[6];
  buf[7] = checksum;
  // send
  mySerial.write(buf, 8);
  // 2 Hz
  Toggle(LED_PIN);
  delay(500); // 500 ms
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
