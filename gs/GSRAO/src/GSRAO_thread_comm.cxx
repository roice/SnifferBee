/* Thread communication among threads in GSRAO
 * 
 * Author:  Bing Luo
 * Date:    2017-01-20
 * */

#include <pthread.h>
#include "GSRAO_thread_comm.h"

GSRAO_thread_comm_t thread_comm;

void GSRAO_init_thread_comm(void)
{
    pthread_mutex_init(&(thread_comm.lock_robot_state), NULL);
}

GSRAO_thread_comm_t* GSRAO_get_thread_comm (void)
{
    return &thread_comm;
}
