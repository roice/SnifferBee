/*
 * Serial Protocal for R.M. Young sonic wind sensor
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.09.05
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h> // nanosleep()
#include <vector>
#include "io/serial.h"

typedef struct {
    int index;
    void* arg;
} Thread_Arguments_t;

typedef struct {
    // frame
    char frame[18];     // 18 bytes
    // pointer
    int pointer;
    // parsed value
    signed short u;     // U vector cm/s
    signed short v;     // V vector cm/s
    signed short w;     // W vector cm/s
    unsigned short T;   // sonic temperature K*100
    unsigned short V1;  // V1 input 5V Full Scale, 0-4000
    unsigned short V2;  // V2 input 5V Full Scale, 0-4000
    unsigned short V3;  // V3 input 1V Full Scale, 0-4000
    unsigned short V4;  // V4 input 1V Full Scale, 0-4000
    char    status;     // status, byte, non-zero=error
    char    checksum;   // byte, XOR of all chars, hex val
} Young_Binary_Frame_t;

static int num_ports = 1;
static int fd[SERIAL_YOUNG_MAX_ANEMOMETERS]; // max number of sensors supported
static pthread_t    young_read_thread_handle[SERIAL_YOUNG_MAX_ANEMOMETERS];
static pthread_t    young_write_thread_handle;
static bool     exit_young_thread = false;
Thread_Arguments_t  young_thread_args[SERIAL_YOUNG_MAX_ANEMOMETERS];
Anemometer_Data_t   wind_data[SERIAL_YOUNG_MAX_ANEMOMETERS];
std::vector<Anemometer_Data_t> wind_record[SERIAL_YOUNG_MAX_ANEMOMETERS];
std::string path_ports[SERIAL_YOUNG_MAX_ANEMOMETERS];
static Young_Binary_Frame_t young_frame[SERIAL_YOUNG_MAX_ANEMOMETERS];

static void* young_write_loop(void*);
static void* young_read_loop(void*);

bool sonic_anemometer_young_init(int n_ports = 1, std::string* ports = NULL)
{
    if (n_ports < 1 or n_ports > SERIAL_YOUNG_MAX_ANEMOMETERS)
        return false;

    if (!ports) return false;

    // open serial port
    for (int i = 0; i < n_ports; i++) {
        fd[i] = serial_open(ports[i].c_str()); // blocking
        if (fd[i] == -1)
            return false;
    }

    // setup serial port
    for (int i = 0; i < n_ports; i++) {
        if (!serial_setup(fd[i], 38400)) // N81
            return false;
    }

    // clear young frame
    memset(young_frame, 0, sizeof(young_frame));

    // create thread for receiving anemometer measurements
    exit_young_thread = false;
    num_ports = n_ports;

printf("num_ports = %d\n", num_ports);

    for (int i = 0; i < n_ports; i++) {
        young_thread_args[i].arg = &exit_young_thread;
        young_thread_args[i].index = i;
        if (pthread_create(&young_read_thread_handle[i], NULL, &young_read_loop, (void*)&young_thread_args[i]) != 0)
            return false;
    }
    if (pthread_create(&young_write_thread_handle, NULL, &young_write_loop, (void*)&exit_young_thread) != 0)
            return false;

    return true;
}

void sonic_anemometer_young_close(void)
{
    if (!exit_young_thread) // if still running
    {
        // exit young thread
        exit_young_thread = true;
        pthread_join(young_write_thread_handle, NULL);
        for (int i = 0; i < num_ports; i++)
            pthread_join(young_read_thread_handle[i], NULL);
        // close serial port
        for (int i = 0; i < num_ports; i++)
            serial_close(fd[i]);
        printf("Young anemometer serial thread terminated.\n");
    }
}

static void youngProcessFrame(char* buf, int len, int index)
{ 
    if (index < 0 or index >= SERIAL_YOUNG_MAX_ANEMOMETERS)
        return;

    for (int i = 0; i < len; i++) {
        if (young_frame[index].pointer < 18)
            young_frame[index].frame[young_frame[index].pointer++] = buf[i];
        if (young_frame[index].pointer >= 18) {
            for (int i = 0; i < 17; i++)
                young_frame[index].checksum ^= young_frame[index].frame[i];
            // check if this frame is valid
            if (young_frame[index].frame[16] == 0 and young_frame[index].frame[17] == young_frame[index].checksum) {
                // parse raw data
                young_frame[index].u = (((unsigned short)(young_frame[index].frame[0]) << 8) & 0xff00) | ((unsigned short)(young_frame[index].frame[1]) & 0x00ff);
                young_frame[index].v = (((unsigned short)(young_frame[index].frame[2]) << 8) & 0xff00) | ((unsigned short)(young_frame[index].frame[3]) & 0x00ff);
                young_frame[index].w = (((unsigned short)(young_frame[index].frame[4]) << 8) & 0xff00) | ((unsigned short)(young_frame[index].frame[5]) & 0x00ff);
                young_frame[index].T = (((unsigned short)(young_frame[index].frame[6]) << 8) & 0xff00) | ((unsigned short)(young_frame[index].frame[7]) & 0x00ff);
                young_frame[index].V1 = (((unsigned short)(young_frame[index].frame[8]) << 8) & 0xff00) | ((unsigned short)(young_frame[index].frame[9]) & 0x00ff);
                young_frame[index].V2 = (((unsigned short)(young_frame[index].frame[10]) << 8) & 0xff00) | ((unsigned short)(young_frame[index].frame[11]) & 0x00ff);
                young_frame[index].V3 = (((unsigned short)(young_frame[index].frame[12]) << 8) & 0xff00) | ((unsigned short)(young_frame[index].frame[13]) & 0x00ff);
                young_frame[index].V4 = (((unsigned short)(young_frame[index].frame[14]) << 8) & 0xff00) | ((unsigned short)(young_frame[index].frame[15]) & 0x00ff);
                // save data
                wind_data[index].speed[0] = (float)young_frame[index].u / 100.0;   // convert to m/s
                wind_data[index].speed[1] = (float)young_frame[index].v / 100.0;
                wind_data[index].speed[2] = (float)young_frame[index].w / 100.0;
                wind_data[index].temperature = (float)young_frame[index].T / 100.0 - 273.15; // convert to degree centigrade
                // save record
                wind_record[index].push_back(wind_data[index]);

//printf("anemometer %d , speed = [ %f, %f, %f ], temperature = %f.\n", index, wind_data[index].speed[0], wind_data[index].speed[1], wind_data[index].speed[2], wind_data[index].temperature);

            }
            young_frame[index].pointer = 0;
            young_frame[index].checksum = 0;
        }
    }
}

static void* young_write_loop(void* exit)
{
    struct timespec req, rem;
    // loop interval
    req.tv_sec = 0;
    req.tv_nsec = (int)(1000000000.0 / 32.0); // 32 Hz
   
    char command[3] = {0x4D, 0x41, 0x21};

    while (!*((bool*)exit))
    {
        for (int i = 0; i < num_ports; i++)
            serial_write(fd[i], command, 3);
        nanosleep(&req, &rem);
    }
}

static void* young_read_loop(void* args)
{
    int nbytes;
    char frame[512];
    
    while (!*((bool*)(((Thread_Arguments_t*)args)->arg)))
    {
        nbytes = serial_read(fd[((Thread_Arguments_t*)args)->index], frame, 512);
        if (nbytes > 0)
            youngProcessFrame(frame, nbytes, ((Thread_Arguments_t*)args)->index);
    }
}

std::string* sonic_anemometer_get_port_paths(void)
{
    return path_ports;
}

std::vector<Anemometer_Data_t>* sonic_anemometer_get_wind_record(void)
{
    return wind_record;
}

Anemometer_Data_t* sonic_anemometer_get_wind_data(void)
{
    return wind_data;
}
