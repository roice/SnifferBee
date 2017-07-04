/*
 * Send udp frames to pioneer robot
 *
 * Written by Roice, Luo
 *
 * Date: 2017.02.26 create this file
 *       2017.07.02 modify
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

#define MAX_PACKETSIZE			1024	// max size of packet (actual packet size is dynamic)
#define IP_PIONEER              "192.168.30.81"
#define PORT_DATA  			    7070

/* Commands */
#define     PIOBEAR_CMD_GOTO    100
#define     PIOBEAR_CMD_STOP    200

static char message[MAX_PACKETSIZE];

static int fd_sock;
static struct sockaddr_in addr;

bool net_send_bear_cmd_init(const char* local_address)
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

/*
 * send command 'goto' to the bear
 * the bear then execute ArActionGoto
 *
 * Arguments:
 * dist_right:      distance to move right from here (mm)
 * dist_front:      distance to move front from here (mm)
 * heading:         robot's heading to change to
 */
void net_send_bear_cmd_goto(float dist_right, float dist_front, float heading)
{
    char checksum = 0;
    unsigned char idx = 0;

    message[0] = '$';               // header 1
    message[1] = 'R';               // header 2
    message[2] = PIOBEAR_CMD_GOTO;  // command goto
    message[3] = 3*sizeof(float);   // data bytes, 3*sizeof(float)
    idx = 4;                        // already 4 bytes
    memcpy(&message[idx], (char*)&dist_right, sizeof(float));
    idx += sizeof(float);
    memcpy(&message[idx], (char*)&dist_front, sizeof(float));
    idx += sizeof(float);
    memcpy(&message[idx], (char*)&heading, sizeof(float));
    idx += sizeof(float);
    for (int i = 2; i < idx; i++)   // checksum from command
        checksum ^= message[i];
    message[idx++] = checksum;      // now idx is also the num of bytes
    
    if (sendto(fd_sock, message, idx, 0, (struct sockaddr*)&addr, sizeof(addr)) < 0)
        printf("PioBear Error: Send command to pioneer via UDP failed. function is net_send_bear_cmd_goto() \n");
}

void net_send_bear_cmd_terminate(void)
{
    // close socket
    close(fd_sock);
    printf("PioBear net (UDP) command send terminated.\n");
}
