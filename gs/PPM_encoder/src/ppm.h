/**
 * PPM encoder
 *          ppm signal producing
 * This project receives command about RC vaules from PC via serial port and g-
 * enerate PPM signals for every UAV. The signals are connected to TX modules.
 *
 * Author: Roice Luo (Bing Luo)
 * Date:   2016-04-20 create this file
 */

#ifndef PPM_H
#define PPM_H

#include "config.h" // configs of PPM_encoder project
#include "stm32f1xx_hal.h" // uint8_t uint16_t uint32_t

#ifndef PPM_CH_NUM
#define PPM_CH_NUM  4
#endif

// PPM frame state
#define PPM_FRAME_SYNC  0
#define PPM_FRAME_CH1   1
#define PPM_FRAME_CH2   2
#define PPM_FRAME_CH3   3
#define PPM_FRAME_CH4   4
#define PPM_FRAME_CH5   5
#define PPM_FRAME_CH6   6
#define PPM_FRAME_CH7   7
#define PPM_FRAME_CH8   8
#define PPM_FRAME_END   9

// PPM state phase
#define PPM_PHASE_PERIOD    0
#define PPM_PHASE_PULSE     1

// PPM constants
#define PPM_SYNC_TIME   4000 // us
#define PPM_STOP_TIME   300 // us

typedef struct {
    uint8_t state; // PPM frame state
    uint8_t phase; // phase in a state
    uint16_t channel[8]; // channel value, 1000~2000 us
    int32_t count; // time left for a frame, (1/9 us)
} PPM_Signal_t;

void PPM_Signal_Init(PPM_Signal_t*);
uint8_t PPM_Signal_Marching(PPM_Signal_t*);
uint16_t PPM_Timer_Period(PPM_Signal_t*);

#endif
