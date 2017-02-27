/*
 * Receive udp frames from odor compass device
 *
 * Written by Roice, Luo
 *
 * Date: 2017.02.26 create this file
 */

#ifndef UDP_OCDEV_RECEIVE_H
#define UDP_OCDEV_RECEIVE_H

bool ocdev_receive_init(const char* local_address);
void ocdev_receive_close(void);

#endif
