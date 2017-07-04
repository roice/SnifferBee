/*
 * Receive data from bear (pioneer robot)
 *
 * Author:  Roice Luo
 * Date:    2017.07.03
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
/* */
#include "piobear_data_parser.h"

#define MAX_PACKETSIZE			4096	// max size of packet (actual packet size is dynamic)
#define MULTICAST_ADDRESS		"224.1.1.1" // IANA, local network
#define PORT_DATA  			    5005

static int fd_sock;
static struct sockaddr_in addr;
static pthread_t udp_receive_thread_handle;
static bool exit_udp_receive_thread = true;
static void* udp_receive_loop(void* exit);

bool net_receive_bear_data_init(const char* local_address)
{
    /* TODO: local_address */

    struct ip_mreq mreq;

    // create UDP socket
    fd_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (fd_sock < 0) {
        perror("PioBear GS Error: in net_receive_bear_data_init(), socket create failed.\n");
        return false;
    }
    
    memset((char *)&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(PORT_DATA);

    if (bind(fd_sock, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("PioBear GS Error: in net_receive_bear_data_init(), socket bind failed.\n");
        return false;
    }

    mreq.imr_multiaddr.s_addr = inet_addr(MULTICAST_ADDRESS);         
    //mreq.imr_interface.s_addr = inet_addr(local_address);         
    if (setsockopt(fd_sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        perror("PioBear GS Error: in net_receive_bear_data_init(), setsockopt failed.\n");
        return false;
    }

    // setup timeout
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 500000; // 0.5 s
    if (setsockopt(fd_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("PioBear GS Error: in net_receive_bear_data_init(), setsockopt failed.\n");
        return false;
    }

    /* create thread to receive data */
    exit_udp_receive_thread = false;
    if (pthread_create(&udp_receive_thread_handle, NULL, &udp_receive_loop, (void*)&exit_udp_receive_thread) != 0)
        return false;

    return true;
}

static void* udp_receive_loop(void* exit)
{
    int addrlen = sizeof(addr), cnt;
    char message[MAX_PACKETSIZE];

    while (!*((bool*)exit))
    {
        cnt = recvfrom(fd_sock, message, sizeof(message), 0, (struct sockaddr *) &addr, (socklen_t *) &addrlen);
        if (cnt > 0) // normal receiving
        {
            piobear_parse_data(message, cnt);
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

void net_receive_bear_data_terminate(void)
{
    if (!exit_udp_receive_thread) // if still running
    {
        // exit udp receive thread
        exit_udp_receive_thread = true;
        pthread_join(udp_receive_thread_handle, NULL);
        // close socket
        close(fd_sock);
        printf("PioBear UDP receive thread terminated.\n");
    }
}
