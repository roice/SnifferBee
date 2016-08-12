#ifndef ROBOT_H
#define ROBOT_H

typedef struct {
    float position[3];
    float attitude[3];
    float wind[3];
} robot_state_t;

robot_state_t* robot_get_state(void);

#endif
