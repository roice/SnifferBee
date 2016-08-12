/* 
 * Playing thread
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.08.10
 */

#ifndef PLAY_THREAD_H
#define PLAY_THREAD_H

bool play_thread_init(void);

void play_thread_stop(void);

void play_thread_set_file_path(const char*);

void* play_thread_get_data(void);

#endif
