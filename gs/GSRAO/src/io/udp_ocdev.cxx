/*
 * Receive udp frames from odor compass device
 *
 * Written by Roice, Luo
 *
 * Date: 2017.02.26 create this file
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
/* socket */
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
/* thread */
#include <pthread.h>
/* GSRAO */
#include "mocap/packet_client.h"
#include "robot/robot.h"
#include "GSRAO_thread_comm.h"

#define MAX_PACKETSIZE			4096	// max size of packet (actual packet size is dynamic)
#define MULTICAST_ADDRESS		"224.1.1.1" // IANA, local network
#define PORT_DATA  			    5005

static int fd_sock;
static struct sockaddr_in addr;
static pthread_t ocdev_receive_thread_handle;
static bool exit_ocdev_receive_thread = false;
static void* ocdev_receive_loop(void* exit);

float pool_ocdev_samples[100*3];

bool ocdev_receive_init(const char* local_address)
{
    struct ip_mreq mreq;

    // create UDP socket
    fd_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (fd_sock < 0)
        return false;
    
    memset((char *)&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(PORT_DATA);

    if (bind(fd_sock, (struct sockaddr *) &addr, sizeof(addr)) < 0)
        return false;

    mreq.imr_multiaddr.s_addr = inet_addr(MULTICAST_ADDRESS);         
    mreq.imr_interface.s_addr = inet_addr(local_address);         
    if (setsockopt(fd_sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0)
        return false;

    // setup timeout
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 500000; // 0.5 s
    if (setsockopt(fd_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
        return false;

    /* create thread to receive data */
    exit_ocdev_receive_thread = false;
    if (pthread_create(&ocdev_receive_thread_handle, NULL, &ocdev_receive_loop, (void*)&exit_ocdev_receive_thread) != 0)
        return false;

    return true;
}

static void* ocdev_receive_loop(void* exit)
{
    int addrlen = sizeof(addr), cnt;
    char message[MAX_PACKETSIZE];
    char value_ascii[20];
    int idx_value_ascii = 0; // index in value_ascii
    int idx_sample = 0; // index in pool_ocdev_samples

    GSRAO_thread_comm_t* tc = GSRAO_get_thread_comm();
    static int rcv_count = 0;

    while (!*((bool*)exit))
    {
        cnt = recvfrom(fd_sock, message, sizeof(message), 0, (struct sockaddr *) &addr, (socklen_t *) &addrlen);
        if (cnt > 0) // normal receiving
        {
            pthread_mutex_lock(&(tc->lock_ocdev_data)); // keep other threads from visiting
            // save samples to data
            idx_sample = 0;
            for (int i = 0; i < cnt; i++) {
                switch (message[i]) {
                    case ',':
                        value_ascii[idx_value_ascii] = 0x00;
                        idx_value_ascii = 0;
                        pool_ocdev_samples[(idx_sample%3)*100+idx_sample/3] = 3.0*atof(value_ascii);
                        idx_sample++;
                        break;
                    default:
                        idx_value_ascii = idx_value_ascii < 6 ? idx_value_ascii : 0; // 0.xxxx
                        value_ascii[idx_value_ascii++] = message[i];
                        break;
                }
            }
            // add robot state to robot record
            Robot_Record_t record = {0};
            std::vector<Robot_Record_t>* robot_rec = robot_get_record();
            MocapData_t* data = mocap_get_data();
            Robot_State_t* robot_state = robot_get_state();
            memcpy(record.enu, data->robot[0].enu, 3*sizeof(float));
            memcpy(record.att, data->robot[0].att, 3*sizeof(float));
            record.sensor[0] = pool_ocdev_samples[0];
            record.sensor[1] = pool_ocdev_samples[100];
            record.sensor[2] = pool_ocdev_samples[200];
            record.count = rcv_count++;
            record.time = rcv_count*0.1; // s
            robot_rec[0].push_back(record);
            pthread_mutex_unlock(&(tc->lock_ocdev_data));
            pthread_cond_signal(&(tc->cond_ocdev_data));
        }
        else if (cnt == 0) { // link is terminated
            break;
        }
        else    // timeout error
        {
            //printf("timeout\n");
        }
    }
}

void ocdev_receive_close(void)
{
    if (!exit_ocdev_receive_thread) // if still running
    {
        // exit ocdev receive thread
        exit_ocdev_receive_thread = true;
        pthread_join(ocdev_receive_thread_handle, NULL);
        // close socket
        close(fd_sock);
        printf("Odor compass device receive thread terminated\n");
    }
}
