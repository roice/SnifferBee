/*
 * Send udp frames to pioneer robot
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
#define IP_PIONEER              "192.168.1.110"
#define PORT_DATA  			    7070

static int fd_sock;
static struct sockaddr_in addr;

bool pioneer_send_init(const char* local_address)
{
    // create UDP socket
    fd_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (fd_sock < 0)
        return false;
    
    memset((char *)&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(IP_PIONEER);
    addr.sin_port = htons(PORT_DATA);

    return true;
}

void pioneer_send_vw(float v, float w)
{
    char buf[20] = {0};
    buf[0] = 'g';
    buf[1] = 2; // not using pan
    memcpy(&buf[2], &v, sizeof(float));
    memcpy(&buf[6], &w, sizeof(float));
    if (sendto(fd_sock, buf, sizeof(buf), 0, (struct sockaddr*)&addr, sizeof(addr)) < 0)
        printf("Send command to pioneer via UDP failed.\n");
}

void pioneer_send_close(void)
{
    // send stop to robot
    pioneer_send_vw(0, 0);

    // close socket
    close(fd_sock);
    printf("Pioneer UDP send terminated\n");
}
