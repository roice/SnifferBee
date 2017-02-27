/*
 * Pioneer Robot
 *         
 *
 * Author: Roice (LUO Bing)
 * Date: 2017-02-26 create this file
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
/* thread */
#include <pthread.h>
/* GSRAO */
#include "mocap/packet_client.h"
#include "robot/microbee.h"
#include "robot/robot.h"
#include "io/serial.h"
#include "common/vector_rotation.h"
#include "GSRAO_Config.h"
#include "GSRAO_thread_comm.h"
/* CBLAS */
#include "cblas.h"
/* Liquid */
#include "liquid.h"

static pthread_t pioneer_control_thread_handle;
static bool exit_pioneer_control_thread = false;

static void* pioneer_control_loop(void*);

bool pioneer_control_init(void)
{
    exit_pioneer_control_thread = false;
    if (pthread_create(&pioneer_control_thread_handle, NULL, &pioneer_control_loop, (void*)&exit_pioneer_control_thread) != 0)
        return false;

    return true;
}

void pioneer_control_close(void)
{
    if (!exit_pioneer_control_thread)
    {
        exit_pioneer_control_thread = true;
        pthread_join(pioneer_control_thread_handle, NULL);
        printf("pioneer control thread terminated\n");
    }
}

static void* pioneer_control_loop(void* exit)
{
    struct timespec req, rem;

    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = 500000000L; // 500 ms

    while (!*((bool*)exit))
    {
        // send reference position to the robot

        // 2 Hz
        nanosleep(&req, &rem); // 500 ms
    }
}

