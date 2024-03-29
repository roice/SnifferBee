/*
 * MicroBee Robot
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.25
 */

#ifndef MICROBEE_H
#define MICROBEE_H

#define MICROBEE_DENOISE_BEFORE_CONTROL

typedef struct {
    bool linked; // false for unlinked, true for linked to GS
    bool armed; // false for disarm, true for arm
    float bat_volt; // battery voltage
} MicroBee_State_t;

typedef struct {
    float front; // voltage of front sensor
    float left;
    float right;
} MicroBee_Sensors_t;

typedef struct {
    MicroBee_State_t state;
    MicroBee_Sensors_t sensors; // 3 gas sensors
    int motor[4]; // 4 motor values
    int count; // sequence of this frame counted by microbee
    double time; // stores the latest time receiving messages from microbee
} MicroBee_t;

MicroBee_t* microbee_get_states(void);

bool microbee_control_init(int);
void microbee_control_close(int);
bool microbee_state_init(void);
void microbee_state_close(void);

void microbee_switch_to_manual(int);
void microbee_switch_to_auto(int);
void microbee_switch_all_to_manual(void);
void microbee_switch_all_to_auto(void);

#endif
