/* Thread communication among threads in GSRAO
 * 
 * Author:  Bing Luo
 * Date:    2017-01-20
 * */

#ifndef THREAD_COMM_H
#define THREAD_COMM_H

typedef struct {
    pthread_mutex_t lock_robot_state; // lock for robot_state
} GSRAO_thread_comm_t;

void GSRAO_init_thread_comm(void);
GSRAO_thread_comm_t* GSRAO_get_thread_comm(void);

#endif
