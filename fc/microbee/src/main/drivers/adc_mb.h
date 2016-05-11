/*
 * ADC for the measurement of three gas sensors on MicroBee
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.11  create this file
 */
#ifndef ADC_MB_H
#define ADC_MB_H

typedef enum {
    ADC_GAS_SENSOR_FRONT = 0,
    ADC_GAS_SENSOR_REAR_LEFT = 1,
    ADC_GAS_SENSOR_REAR_RIGHT = 2,
    MB_ADC_CHANNEL_MAX = ADC_GAS_SENSOR_REAR_RIGHT
} AdcChannel;

#define MB_ADC_CHANNEL_COUNT (MB_ADC_CHANNEL_MAX + 1)

typedef struct mb_adc_config_s {
    uint8_t adcChannel;         // ADC1_INxx channel number
    uint8_t dmaIndex;           // index into DMA buffer in case of sparse channels
    bool enabled;
    uint8_t sampleTime;
} mb_adc_config_t;

void mb_adcInit(void);
uint16_t mb_adcGetChannel(uint8_t channel);

#endif
