/**
 * PPM encoder
 *          ppm signal producing
 * This project receives command about RC vaules from PC via serial port and g-
 * enerate PPM signals for every UAV. The signals are connected to TX modules.
 *
 * Author: Roice Luo (Bing Luo)
 * Date:   2016-04-20 create this file
 */

#include "ppm.h"
#include "stm32f1xx_hal.h"
#include "config.h" // HIGH/LOW

// PPM signal state init
void PPM_Signal_Init(PPM_Signal_t* signal)
{
    signal->state = PPM_FRAME_END;
    signal->phase = PPM_PHASE_PULSE;
    // channel 1-4
    signal->channel[0] = 1000; // Throttle
    signal->channel[1] = 1500; // Roll
    signal->channel[2] = 1500; // Pitch
    signal->channel[3] = 1500; // Yaw
    for (uint8_t i = 4; i < 8; i++) // channel 5-8
        signal->channel[i] = 1000; // 1000 us
    // init time count, the period of a frame is 20 ms
    signal->count = 20 * 1000 * 9; // (1/9) us
}

// update PPM signal state
// return the signal voltage level H/L
uint8_t PPM_Signal_Marching(PPM_Signal_t* signal)
{
    // state machine
    if (signal->state <= PPM_CH_NUM)
    {
        if (signal->phase == PPM_PHASE_PERIOD)
        {
            signal->phase = PPM_PHASE_PULSE;
            return HIGH;
        }
        else
        {
            // march to next channel/state
            signal->phase = PPM_PHASE_PERIOD; 
            signal->state ++;
            if (signal->state > PPM_CH_NUM)
            {
                signal->state = PPM_FRAME_END; 
            }
            return LOW;
        }
    }
    else 
    // Here the PPM_CH_NUM<state<PPM_FRAME_END is not possible if no direct
    // write to signal->state outside this file
    {
        // judge whether to start a new frame
        if (signal->count == 20 * 1000 *9) // 50 Hz, 20 ms per frame
        {// start a new frame
            signal->state = PPM_FRAME_SYNC;
            signal->phase = PPM_PHASE_PERIOD;
        }
        return LOW;
    } 
}

uint16_t PPM_Timer_Period(PPM_Signal_t* signal)
{
    // calculate timer period value on the base of 1/9 us
    uint16_t timer_period;
    switch (signal->state)
    {
        case PPM_FRAME_SYNC:
            if (signal->phase == PPM_PHASE_PERIOD)
                timer_period = (PPM_SYNC_TIME - PPM_STOP_TIME) * 9;
            else
                timer_period = PPM_STOP_TIME * 9;
            break;
        case PPM_FRAME_END:
            if (signal->count > 60000) // due to 16-bit ARR
                timer_period = 60000;
            else
                timer_period = signal->count;
            break;
        default:
            if (signal->phase == PPM_PHASE_PERIOD)
                timer_period = (signal->channel[signal->state-1] - PPM_STOP_TIME) * 9;
            else
                timer_period = PPM_STOP_TIME * 9;
            break;
    }

    signal->count -= timer_period;
    if (signal->count <= 0)
        signal->count = 20 * 1000 * 9;
    
    return timer_period-1;
}
