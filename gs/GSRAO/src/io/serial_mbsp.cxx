/*
 * MicroBee Serial Protocal
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.05.24
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h> // nanosleep()
#include "io/serial.h"
#include "robot/robot.h"
#include "robot/microbee.h"

#define MBSP_ADDRESS_GS         0   // address of ground station
#define MBSP_ADDRESS_MB_1       1   // address of Microbee No. 1
#define MBSP_ADDRESS_MB_2       2   // address of Microbee No. 2
#define MBSP_ADDRESS_MB_3       3   // address of Microbee No. 3
#define MBSP_ADDRESS_MB_4       4   // address of Microbee No. 4

#define MBSP_CMD_STATUS         101 // status of MicroBee
#define MBSP_CMD_GAS_SENSORS    102 // readings of (three)gas sensors

typedef enum {
    MBSP_IDLE,
    MBSP_START, // '$'
    MBSP_B,     // 'B'
    MBSP_TO,    // device to receive
    MBSP_FROM,  // device which sent this message
    MBSP_LEN,   // data length (bytes)
    MBSP_COMMAND,   // command ID
    MBSP_DATA,  // data
} mbspState_e;

typedef struct {
    char to;
    char from;
    unsigned char len;
    char command;
    char data[256];
} mbspData_t;

static int fd; // file descriptor for the port receiving MBSP messages
static pthread_t mbsp_thread_handle;
static bool exit_mbsp_thread = false;
static char mbsp_frame[256];
static mbspData_t mbsp_data;

static void* mbsp_loop(void*);

bool mbsp_init(const char* port)
{
    // open serial port
    fd = serial_open(port); // blocking
    if (fd == -1)
        return false;

    // setup serial port
    if (!serial_setup(fd, 115200)) // N81
        return false;

    // create thread for PPM sending
    exit_mbsp_thread = false;
    if (pthread_create(&mbsp_thread_handle, NULL, &mbsp_loop, (void*)&exit_mbsp_thread) != 0)
        return false;

    return true;
}

void mbsp_close(void)
{
    if (!exit_mbsp_thread) // if still running
    {
        // exit mbsp thread
        exit_mbsp_thread = true;
        pthread_join(mbsp_thread_handle, NULL);
        // close serial port
        serial_close(fd);
        printf("MicroBee Serial Protocol thread terminated\n");
    }
}



static void mbspEvaluateData(void)
{
    struct timespec time;

    MicroBee_t* mb;

    // this program is GS, 4 robots max
    if (mbsp_data.to == MBSP_ADDRESS_GS 
            && mbsp_data.from <= MBSP_ADDRESS_MB_4
            && mbsp_data.from >= MBSP_ADDRESS_MB_1)
    {
        switch (mbsp_data.command) {
            case MBSP_CMD_STATUS: 
                if (mbsp_data.len != 5) // 1(char type)+4(float type)
                    break;
                mb = microbee_get_states();
                mb[mbsp_data.from-1].state.armed = mbsp_data.data[0]>0? true:false;
                mb[mbsp_data.from-1].state.bat_volt = *(float*)(&(mbsp_data.data[1]));
                printf("arm/disarm is %d, bat volt is %f\n", mb[mbsp_data.from-1].state.armed, mb[mbsp_data.from-1].state.bat_volt);
                clock_gettime(CLOCK_REALTIME, &time);
                mb[mbsp_data.from-1].time = time.tv_sec + time.tv_nsec/1.0e9;
                break;
            case MBSP_CMD_GAS_SENSORS:
                if (mbsp_data.len != 6) // 3*2(uint16_t)
                    break;
                mb = microbee_get_states();
                int front, left, right;
                front = *(short*)(&(mbsp_data.data[0]));
                left = *(short*)(&(mbsp_data.data[2]));
                right = *(short*)(&(mbsp_data.data[4]));
                mb[mbsp_data.from-1].sensors.front = front*3.3/4096.0;
                mb[mbsp_data.from-1].sensors.left = left*3.3/4096.0;
                mb[mbsp_data.from-1].sensors.right = right*3.3/4096.0;
                printf("front %f, left %f, right %f\n", mb[mbsp_data.from-1].sensors.front, mb[mbsp_data.from-1].sensors.left, mb[mbsp_data.from-1].sensors.right);
                clock_gettime(CLOCK_REALTIME, &time);
                mb[mbsp_data.from-1].time = time.tv_sec + time.tv_nsec/1.0e9;
            default:
                break;
        }
    }
}

static void mbspProcessByte(char c)
{
    static mbspState_e mbsp_state = MBSP_IDLE;
    static char mbsp_checksum;
    static int mbsp_count;
    static int data_index;

    if (mbsp_state == MBSP_IDLE) {
        if (c == '$')
            mbsp_state = MBSP_START;
    }
    else if (mbsp_state == MBSP_START) {
        if (c == 'B') {
            mbsp_state = MBSP_B;
            mbsp_checksum = 0;
        }
        else
            mbsp_state = MBSP_IDLE;
    }
    else if (mbsp_state == MBSP_B) {
        mbsp_checksum ^= c;
        mbsp_data.to = c;
        if (c <= 4) // 4 robots max
            mbsp_state = MBSP_TO;
        else
            mbsp_state = MBSP_IDLE;
    }
    else if (mbsp_state == MBSP_TO) {
        mbsp_checksum ^= c;
        mbsp_data.from = c;
        if (c <= 4) // 4 robots max
            mbsp_state = MBSP_FROM;
        else
            mbsp_state = MBSP_IDLE;
    }
    else if (mbsp_state == MBSP_FROM) {
        if (c != 0) {// require at least one byte of data
            mbsp_checksum ^= c;
            mbsp_data.len = c;
            mbsp_count = c;
            data_index = 0;
            mbsp_state = MBSP_LEN;
        }
        else
            mbsp_state = MBSP_IDLE;
    }
    else if (mbsp_state == MBSP_LEN) {
        mbsp_checksum ^= c;
        mbsp_data.command = c;
        mbsp_state = MBSP_COMMAND;
    }
    else if (mbsp_state == MBSP_COMMAND) {
        mbsp_checksum ^= c; 
        mbsp_data.data[data_index] = c;
        data_index ++;
        mbsp_count --;
        if (mbsp_count == 0)
            mbsp_state = MBSP_DATA; // received data, need checksum
    }
    else if (mbsp_state == MBSP_DATA) {
        if (mbsp_checksum == c) { // data OK 
            // evaluate MBSP DATA
            mbspEvaluateData();
        }
        else
        { 
            printf("checksum failed\n");
        }
        mbsp_state = MBSP_IDLE;
    }
}

static void mbspProcessFrame(char* buf, int len)
{
    char c;

    for (int i = 0; i < len; i++)
    {
        c = buf[i];
        mbspProcessByte(c);
    }
}

static void* mbsp_loop(void* exit)
{
    int nbytes;

    while (!*((bool*)exit))
    {
        nbytes = serial_read(fd, mbsp_frame, 256);
        if (nbytes > 0)
            mbspProcessFrame(mbsp_frame, nbytes);
    }
}
