/*
 * Send udp frames to pioneer robot
 *
 * Written by Roice, Luo
 *
 * Date: 2017.02.26 create this file
 */

#ifndef UDP_PIONEER_SEND_H
#define UDP_PIONEER_SEND_H

bool pioneer_send_init(const char* local_address);
void pioneer_send_vw(float, float);
void pioneer_send_close(void);

#endif
