/*
 * Serial PPM protocal for encoder
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.18
 */

#include <stdio.h>
#include <pthread.h>
#include <time.h> // nanosleep()
#include "io/serial.h"

static int fd; // file descriptor for the port connecting PPM encoder
static pthread_t spp_thread_handle;
static bool exit_spp_thread = false;
static SPP_RC_DATA_t spp_rc_data[4];
static char spp_frame[100];

static void* spp_loop(void*);

bool spp_init(const char* port)
{
    // open serial port
    fd = serial_open(port);
    if (fd == -1)
        return false;

    // setup serial port
    if (!serial_setup(fd, 115200)) // N81
        return false;

    // init RC channels
    for (char i = 0; i < 4; i++) // 4 robots max
    {
        spp_rc_data[i].throttle = 1000;
        spp_rc_data[i].roll = 1500;
        spp_rc_data[i].pitch = 1500;
        spp_rc_data[i].yaw = 1500;
    }

    // create thread for PPM sending
    exit_spp_thread = false;
    if (pthread_create(&spp_thread_handle, NULL, &spp_loop, (void*)&exit_spp_thread) != 0)
        return false;

    return true;
}

static void* spp_loop(void* exit)
{
    struct timespec req, rem;
    req.tv_sec = 0;
    req.tv_nsec = 20000000L; // 20 ms

    unsigned short temp;
    unsigned char checksum;

    while (!*((bool*)exit))
    {
        // prepare PPM frame, 35 bytes
        // RAET (yaw roll pitch throttle)
        spp_frame[0] = '$';
        spp_frame[1] = 'P';
        for (char i = 0; i < 4; i++) // 4 robots max
        {
            // yaw
            if (spp_rc_data[i].yaw >= 1000 && spp_rc_data[i].yaw <= 2000)
                temp = spp_rc_data[i].yaw;
            else
                temp = 1000;
            spp_frame[2+i*4*2+0] = (char)temp;
            spp_frame[2+i*4*2+1] = (char)(temp >> 8);
            // roll
            if (spp_rc_data[i].roll >= 1000 && spp_rc_data[i].roll <= 2000)
                temp = spp_rc_data[i].roll;
            else
                temp = 1500;
            spp_frame[2+i*4*2+2] = (char)temp;
            spp_frame[2+i*4*2+3] = (char)(temp >> 8);
            // pitch
            if (spp_rc_data[i].pitch >= 1000 && spp_rc_data[i].pitch <= 2000)
                temp = spp_rc_data[i].pitch;
            else
                temp = 1500;
            spp_frame[2+i*4*2+4] = (char)temp;
            spp_frame[2+i*4*2+5] = (char)(temp >> 8);
            // throttle
            if (spp_rc_data[i].throttle >= 1000 && spp_rc_data[i].throttle <= 2000)
                temp = spp_rc_data[i].throttle;
            else
                temp = 1500;
            spp_frame[2+i*4*2+6] = (char)temp;
            spp_frame[2+i*4*2+7] = (char)(temp >> 8);       
        }
        checksum = 0;
        for (char i = 0; i < 32; i++)
            checksum ^= spp_frame[2+i];
        spp_frame[2+32] = checksum;

        // send PPM frame to encoder
        serial_write(fd, spp_frame, 35);

        // 50 Hz
        nanosleep(&req, &rem); // 20 ms
    }
}

void spp_close(void)
{
    // exit spp thread
    exit_spp_thread = true;
    pthread_join(spp_thread_handle, NULL);

    // close serial port
    serial_close(fd);
}

SPP_RC_DATA_t* spp_get_rc_data(void)
{
    return spp_rc_data;
}
