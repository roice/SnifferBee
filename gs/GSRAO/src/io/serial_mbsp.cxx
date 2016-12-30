/*
 * MicroBee Serial Protocal
 *
 * One MicroBee <--> One Ground Receiver
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
#include <vector>
#include "io/serial.h"
#include "mocap/packet_client.h"
#include "robot/robot.h"
#include "robot/microbee.h"

#define MBSP_ADDRESS_GS         0   // address of ground station
#define MBSP_ADDRESS_MB_1       1   // address of Microbee No. 1
#define MBSP_ADDRESS_MB_2       2   // address of Microbee No. 2
#define MBSP_ADDRESS_MB_3       3   // address of Microbee No. 3
#define MBSP_ADDRESS_MB_4       4   // address of Microbee No. 4

// command of MBSP start from 101
#define MBSP_CMD_STATUS         101 // status of MicroBee
#define MBSP_CMD_MEASUREMENTS   102 // readings of (three)gas sensors & motor values(if required)

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

static int fd[4]; // file descriptor for the port receiving MBSP messages
static pthread_t mbsp_thread_handle[4];
static Thread_Arguments_t  thread_args[4];
static bool exit_mbsp_thread = false;
static mbspData_t mbsp_data[4];

static int num_gs = 0;

static void* mbsp_loop(void*);

bool mbsp_init(std::string* ports, int num_robots)
{
    if (num_robots <= 0 or num_robots > 4)
        return false;

    // open serial port
    for (int i = 0; i < num_robots; i++) {
        fd[i] = serial_open(ports[i].c_str()); // blocking
        if (fd[i] == -1)
            return false;
    }

    // setup serial port
    for (int i = 0; i < num_robots; i++)
        if (!serial_setup(fd[i], 57600)) // N81
            return false;

    exit_mbsp_thread = false;
    for (int i = 0; i < num_robots; i++) {
        thread_args[i].arg = &exit_mbsp_thread;
        thread_args[i].index = i;
        if (pthread_create(&mbsp_thread_handle[i], NULL, &mbsp_loop, (void*)&thread_args[i]) != 0)
            return false;
    }

    num_gs = num_robots;

    return true;
}

void mbsp_close(void)
{
    if (!exit_mbsp_thread and num_gs) // if still running
    {
        // exit mbsp thread
        exit_mbsp_thread = true;
        for (int i = 0; i < num_gs; i++)
            pthread_join(mbsp_thread_handle[i], NULL);
        // close serial port
        for (int i = 0; i < num_gs; i++)
            serial_close(fd[i]);
        printf("MicroBee Serial Protocol thread terminated.\n");
    }
}

static void mbspEvaluateData(int index_mb)
{
    struct timespec time;

    MicroBee_t* mb;

    // this program is GS, 4 robots max
    if (mbsp_data[index_mb].to == MBSP_ADDRESS_GS 
            && mbsp_data[index_mb].from <= MBSP_ADDRESS_MB_4
            && mbsp_data[index_mb].from >= MBSP_ADDRESS_MB_1)
    {
        switch (mbsp_data[index_mb].command) {
            case MBSP_CMD_STATUS:
            {
                if (mbsp_data[index_mb].len != 3) // 1(char type)+2(short int type)
                    break;
                mb = microbee_get_states();
                mb[mbsp_data[index_mb].from-1].state.armed = mbsp_data[index_mb].data[0]>0? true:false;
                mb[mbsp_data[index_mb].from-1].state.bat_volt = ((*((short*)(&mbsp_data[index_mb].data[1]))) & 0x03FF)*5.0f/1024.0f;
                clock_gettime(CLOCK_REALTIME, &time);
                mb[mbsp_data[index_mb].from-1].time = time.tv_sec + time.tv_nsec/1.0e9;
                break;
            }
            case MBSP_CMD_MEASUREMENTS:
            {
#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
                if (mbsp_data[index_mb].len != 3*2+2+4*2+2+1)
#else
                if (mbsp_data[index_mb].len != 3*2+2+2+1) // 3*sizeof((uint16_t)) + (sizeof(uint16_t)) + (sizeof(uint16_t)) + (sizeof(uint8_t))
#endif
                    break;
                mb = microbee_get_states();
                int front, left, right;
                front = *(short*)(&(mbsp_data[index_mb].data[0]));
                left = *(short*)(&(mbsp_data[index_mb].data[2]));
                right = *(short*)(&(mbsp_data[index_mb].data[4]));
                unsigned short count;
                count = *(short*)(&(mbsp_data[index_mb].data[9]));
#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
                int motor[4];
                for (int i = 0; i < 4; i++)
                    motor[i] = *(short*)(&(mbsp_data[index_mb].data[11+2*i]));
#endif 
                mb[mbsp_data[index_mb].from-1].sensors.front = front*3.3/4096.0;
                mb[mbsp_data[index_mb].from-1].sensors.left = left*3.3/4096.0;
                mb[mbsp_data[index_mb].from-1].sensors.right = right*3.3/4096.0;
                mb[mbsp_data[index_mb].from-1].count = count;
                mb[mbsp_data[index_mb].from-1].state.armed = mbsp_data[index_mb].data[8]>0? true:false;
                mb[mbsp_data[index_mb].from-1].state.bat_volt = ((*((short*)(&mbsp_data[index_mb].data[6]))) & 0x03FF)*5.0f/1024.0f;
#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
                for (int i = 0; i < 4; i++)
                    mb[mbsp_data[index_mb].from-1].motor[i] = motor[i] & 0x0000FFFF;
#endif
                //printf("front %f, left %f, right %f\n", mb[mbsp_data[index_mb].from-1].sensors.front, mb[mbsp_data[index_mb].from-1].sensors.left, mb[mbsp_data[index_mb].from-1].sensors.right);
                //printf("motor: [ %d, %d, %d, %d ]\n", mb[mbsp_data[index_mb].from-1].motor[0], mb[mbsp_data[index_mb].from-1].motor[1], mb[mbsp_data[index_mb].from-1].motor[2], mb[mbsp_data[index_mb].from-1].motor[3]);
                clock_gettime(CLOCK_REALTIME, &time);
                mb[mbsp_data[index_mb].from-1].time = time.tv_sec + time.tv_nsec/1.0e9;
                // record
                Robot_Record_t record = {0};
                std::vector<Robot_Record_t>* robot_rec = robot_get_record();
                MocapData_t* data = mocap_get_data();
                Robot_State_t* robot_state = robot_get_state();
                memcpy(record.enu, data->robot[mbsp_data[index_mb].from-1].enu, 3*sizeof(float));
                memcpy(record.att, data->robot[mbsp_data[index_mb].from-1].att, 3*sizeof(float));
                memcpy(record.sensor, &(mb[mbsp_data[index_mb].from-1].sensors.front), 3*sizeof(float));
                memcpy(&(record.count), &(mb[mbsp_data[index_mb].from-1].count), sizeof(int));
#ifdef MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE
                memcpy(record.motor, mb[mbsp_data[index_mb].from-1].motor, 4*sizeof(int));
#endif
                memcpy(&(record.bat_volt), &(mb[mbsp_data[index_mb].from-1].state.bat_volt), sizeof(float));
                record.time = mb[mbsp_data[index_mb].from-1].time;
                memcpy(record.wind, robot_state[mbsp_data[index_mb].from-1].wind, 3*sizeof(float));
                robot_rec[mbsp_data[index_mb].from-1].push_back(record);
                break;
            }
            default:
                break;
        }
    }
}

static void mbspProcessByte(char c, int index_mb)
{
    static mbspState_e mbsp_state[4] = {MBSP_IDLE, MBSP_IDLE, MBSP_IDLE, MBSP_IDLE};
    static char mbsp_checksum[4];
    static int mbsp_count[4];
    static int data_index[4];

    if (mbsp_state[index_mb] == MBSP_IDLE) {
        if (c == '$')
            mbsp_state[index_mb] = MBSP_START;
    }
    else if (mbsp_state[index_mb] == MBSP_START) {
        if (c == 'B') {
            mbsp_state[index_mb] = MBSP_B;
            mbsp_checksum[index_mb] = 0;
        }
        else
            mbsp_state[index_mb] = MBSP_IDLE;
    }
    else if (mbsp_state[index_mb] == MBSP_B) {
        mbsp_checksum[index_mb] ^= c;
        mbsp_data[index_mb].to = c;
        if (c <= 4) // 4 robots max
            mbsp_state[index_mb] = MBSP_TO;
        else
            mbsp_state[index_mb] = MBSP_IDLE;
    }
    else if (mbsp_state[index_mb] == MBSP_TO) {
        mbsp_checksum[index_mb] ^= c;
        mbsp_data[index_mb].from = c;
        if (c <= 4) // 4 robots max
            mbsp_state[index_mb] = MBSP_FROM;
        else
            mbsp_state[index_mb] = MBSP_IDLE;
    }
    else if (mbsp_state[index_mb] == MBSP_FROM) {
        if (c != 0) {// require at least one byte of data
            mbsp_checksum[index_mb] ^= c;
            mbsp_data[index_mb].len = c;
            mbsp_count[index_mb] = c;
            data_index[index_mb] = 0;
            mbsp_state[index_mb] = MBSP_LEN;
        }
        else
            mbsp_state[index_mb] = MBSP_IDLE;
    }
    else if (mbsp_state[index_mb] == MBSP_LEN) {
        mbsp_checksum[index_mb] ^= c;
        mbsp_data[index_mb].command = c;
        if (c == 101 || c == 102)
            mbsp_state[index_mb] = MBSP_COMMAND;
        else
            mbsp_state[index_mb] = MBSP_IDLE;
    }
    else if (mbsp_state[index_mb] == MBSP_COMMAND) {
        mbsp_checksum[index_mb] ^= c; 
        mbsp_data[index_mb].data[data_index[index_mb]] = c;
        data_index[index_mb] ++;
        mbsp_count[index_mb] --;
        if (mbsp_count[index_mb] == 0)
            mbsp_state[index_mb] = MBSP_DATA; // received data, need checksum
    }
    else if (mbsp_state[index_mb] == MBSP_DATA) {
        if (mbsp_checksum[index_mb] == c) { // data OK 
            // evaluate MBSP DATA
            mbspEvaluateData(index_mb);
        }
        else
        {
            printf("checksum failed\n");
            printf("mbsp_data[index_mb].to = %x, from = %x, len = %x, command = %x, data = ", mbsp_data[index_mb].to, mbsp_data[index_mb].from, mbsp_data[index_mb].len, mbsp_data[index_mb].command);
            for (int i = 0; i < mbsp_data[index_mb].len; i++)
                printf("%x ", mbsp_data[index_mb].data[i]);
            printf("checksum = %x, while received %x\n", mbsp_checksum[index_mb], c);
        }
        mbsp_state[index_mb] = MBSP_IDLE;
    }
}

static void mbspProcessFrame(char* buf, int len, int index_mb)
{
    char c;

    for (int i = 0; i < len; i++)
    {
        c = buf[i];
        mbspProcessByte(c, index_mb);
    }
}

static void* mbsp_loop(void* args)
{
    int nbytes;
    char mbsp_frame[4][256]; // 4 robots

    while (!*((bool*)(((Thread_Arguments_t*)args)->arg)))
    {
        nbytes = serial_read(fd[((Thread_Arguments_t*)args)->index], mbsp_frame[((Thread_Arguments_t*)args)->index], 256);
        if (nbytes > 0) {
            /*
            printf("Received %d bytes: ", nbytes);
            for (int i = 0; i < nbytes; i++)
                printf("%x ", mbsp_frame[((Thread_Arguments_t*)args)->index][i]);
            printf("\n");
            */
            mbspProcessFrame(mbsp_frame[((Thread_Arguments_t*)args)->index], nbytes, ((Thread_Arguments_t*)args)->index);
        }
    }
}
