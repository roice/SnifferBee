/*
 * Receive udp frames from odor compass device
 *
 * Written by Roice, Luo
 *
 * Date: 2017.02.26 create this file
 */

#ifndef UDP_OCDEV_RECEIVE_H
#define UDP_OCDEV_RECEIVE_H

extern float pool_ocdev_samples[]; // 3 sensors, 100 samples each

bool ocdev_receive_init(const char* local_address);
void ocdev_receive_close(void);

#endif
