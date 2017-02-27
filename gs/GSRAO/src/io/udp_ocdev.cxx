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

#define MAX_PACKETSIZE			4096	// max size of packet (actual packet size is dynamic)
#define PORT_DATA  			    1511


static int fd_sock;
static struct sockaddr_in addr;
static pthread_t ocdev_receive_thread_handle;
static bool exit_ocdev_receive_thread = false;
static void* ocdev_receive_loop(void* exit);

bool ocdev_receive_init(const char* local_address)
{
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

    while (!*((bool*)exit))
    {
        cnt = recvfrom(fd_sock, message, sizeof(message), 0, (struct sockaddr *) &addr, (socklen_t *) &addrlen);
        if (cnt > 0) // normal receiving
        {
            printf("%d\n", message[0]);
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
