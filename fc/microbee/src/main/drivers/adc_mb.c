/*
 * ADC for the measurement of three gas sensors on MicroBee
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.11  create this file
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <platform.h>
#include "build_config.h"

#include "system.h"

#include "drivers/adc_mb.h"

#ifndef ADC_INSTANCE
#define ADC_INSTANCE                ADC1
#define ADC_ABP2_PERIPHERAL         RCC_APB2Periph_ADC1
#define ADC_AHB_PERIPHERAL          RCC_AHBPeriph_DMA1
#define ADC_DMA_CHANNEL             DMA1_Channel1
#endif

mb_adc_config_t mb_adcConfig[MB_ADC_CHANNEL_COUNT];
volatile uint16_t mb_adcValues[MB_ADC_CHANNEL_COUNT];

void mb_adcInit(void)
{
    uint8_t configuredAdcChannels = 0;

    GPIO_InitTypeDef GPIO_InitStructure;
    GPIO_StructInit(&GPIO_InitStructure);
    GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_AIN;
    
    // GAS SENSOR FRONT
    GPIO_InitStructure.GPIO_Pin = GAS_SENSOR_FRONT_ADC_GPIO_PIN;
    GPIO_Init(GAS_SENSOR_FRONT_ADC_GPIO, &GPIO_InitStructure);
    mb_adcConfig[ADC_GAS_SENSOR_FRONT].adcChannel = GAS_SENSOR_FRONT_ADC_CHANNEL;
    mb_adcConfig[ADC_GAS_SENSOR_FRONT].dmaIndex = configuredAdcChannels++;
    mb_adcConfig[ADC_GAS_SENSOR_FRONT].enabled = true;
    mb_adcConfig[ADC_GAS_SENSOR_FRONT].sampleTime = ADC_SampleTime_239Cycles5;

    // GAS SENSOR REAR LEFT
    GPIO_InitStructure.GPIO_Pin = GAS_SENSOR_REAR_LEFT_ADC_GPIO_PIN;
    GPIO_Init(GAS_SENSOR_REAR_LEFT_ADC_GPIO, &GPIO_InitStructure);
    mb_adcConfig[ADC_GAS_SENSOR_REAR_LEFT].adcChannel = GAS_SENSOR_REAR_LEFT_ADC_CHANNEL;
    mb_adcConfig[ADC_GAS_SENSOR_REAR_LEFT].dmaIndex = configuredAdcChannels++;
    mb_adcConfig[ADC_GAS_SENSOR_REAR_LEFT].enabled = true;
    mb_adcConfig[ADC_GAS_SENSOR_REAR_LEFT].sampleTime = ADC_SampleTime_239Cycles5;

    // GAS SENSOR REAR RIGHT
    GPIO_InitStructure.GPIO_Pin = GAS_SENSOR_REAR_RIGHT_ADC_GPIO_PIN;
    GPIO_Init(GAS_SENSOR_REAR_RIGHT_ADC_GPIO, &GPIO_InitStructure);
    mb_adcConfig[ADC_GAS_SENSOR_REAR_RIGHT].adcChannel = GAS_SENSOR_REAR_RIGHT_ADC_CHANNEL;
    mb_adcConfig[ADC_GAS_SENSOR_REAR_RIGHT].dmaIndex = configuredAdcChannels++;
    mb_adcConfig[ADC_GAS_SENSOR_REAR_RIGHT].enabled = true;
    mb_adcConfig[ADC_GAS_SENSOR_REAR_RIGHT].sampleTime = ADC_SampleTime_239Cycles5;

    RCC_ADCCLKConfig(RCC_PCLK2_Div8);  // 9MHz from 72MHz APB2 clock(HSE), 8MHz from 64MHz (HSI)
    RCC_AHBPeriphClockCmd(ADC_AHB_PERIPHERAL, ENABLE);
    RCC_APB2PeriphClockCmd(ADC_ABP2_PERIPHERAL, ENABLE);

    // FIXME ADC driver assumes all the GPIO was already placed in 'AIN' mode

    DMA_DeInit(ADC_DMA_CHANNEL);
    DMA_InitTypeDef DMA_InitStructure;
    DMA_StructInit(&DMA_InitStructure);
    DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&ADC_INSTANCE->DR;
    DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)mb_adcValues;
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralSRC;
    DMA_InitStructure.DMA_BufferSize = configuredAdcChannels;
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
    DMA_InitStructure.DMA_MemoryInc = configuredAdcChannels > 1 ? DMA_MemoryInc_Enable : DMA_MemoryInc_Disable;
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;
    DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;
    DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
    DMA_Init(ADC_DMA_CHANNEL, &DMA_InitStructure);
    DMA_Cmd(ADC_DMA_CHANNEL, ENABLE);

    ADC_InitTypeDef ADC_InitStructure;
    ADC_StructInit(&ADC_InitStructure);
    ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;
    ADC_InitStructure.ADC_ScanConvMode = configuredAdcChannels > 1 ? ENABLE : DISABLE;
    ADC_InitStructure.ADC_ContinuousConvMode = ENABLE;
    ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;
    ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
    ADC_InitStructure.ADC_NbrOfChannel = configuredAdcChannels;
    ADC_Init(ADC_INSTANCE, &ADC_InitStructure);

    for (uint8_t i = 0; i < configuredAdcChannels; i++)
        ADC_RegularChannelConfig(ADC_INSTANCE, mb_adcConfig[i].adcChannel, i+1, mb_adcConfig[i].sampleTime);

    ADC_DMACmd(ADC_INSTANCE, ENABLE);
    ADC_Cmd(ADC_INSTANCE, ENABLE);

    ADC_ResetCalibration(ADC_INSTANCE);
    while(ADC_GetResetCalibrationStatus(ADC_INSTANCE));
    ADC_StartCalibration(ADC_INSTANCE);
    while(ADC_GetCalibrationStatus(ADC_INSTANCE));

    ADC_SoftwareStartConvCmd(ADC_INSTANCE, ENABLE);
}

uint16_t mb_adcGetChannel(uint8_t channel)
{
    return mb_adcValues[mb_adcConfig[channel].dmaIndex];
}
