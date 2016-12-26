/*
 * Serial Protocal for Anemometers
 * Support List:
 *      R.M. Young 3D sonic wind sensor
 *      Gill 3D sonic wind sensor
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.09.05      create this file
 *      2016.12.10      add support for Gill 3D
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h> // nanosleep()
#include <vector>
#include <cmath>
#include "io/serial.h"

// RM Young Binary Frame Type
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

// Gill ASCII Frame Type
typedef struct {
    // protocol
    char protocol[50] = {0x02, 'X', 'X', ',', 'X', 'X', ',', 'S', 'X', 'X', '.', 'X', 'X', ',', 'S', 'X', 'X', '.', 'X', 'X', ',', 'S', 'X', 'X', '.', 'X', 'X', ',', 'S', 'X', 'X', '.', 'X', 'X', ',',0x03};
    // frame
    char frame[100];
    // pointer
    int pointer = 0;
    // parsed value
    short    StaA; // status address
    short    StaD; // status data
    float  u;  // U vector cm/s
    float  v;  // V vector cm/s
    float  w;  // W vector cm/s
    float  T;  // absolute temperature
} Gill_ASCII_Frame_t;

static int num_ports = 0;
static int fd[SERIAL_MAX_ANEMOMETERS]; // max number of sensors supported
static pthread_t    read_thread_handle[SERIAL_MAX_ANEMOMETERS];
static pthread_t    write_thread_handle;
static bool     exit_thread = false;

static Thread_Arguments_t  thread_args[SERIAL_MAX_ANEMOMETERS];
Anemometer_Data_t   wind_data[SERIAL_MAX_ANEMOMETERS];
std::vector<Anemometer_Data_t> wind_record[SERIAL_MAX_ANEMOMETERS];
std::string anemometer_port_path[SERIAL_MAX_ANEMOMETERS];
std::string anemometer_type[SERIAL_MAX_ANEMOMETERS];
static Young_Binary_Frame_t young_frame[SERIAL_MAX_ANEMOMETERS];

// RM Young
static void* young_write_loop(void*);
static void* young_read_loop(void*);
static void* gill_read_loop(void*);

bool sonic_anemometer_init(int n_ports = 1, std::string* ports = NULL, std::string* types = NULL)
{
    if (n_ports < 1 or n_ports > SERIAL_MAX_ANEMOMETERS)
        return false;

    if (!ports or !types) return false;

    // open serial port
    for (int i = 0; i < n_ports; i++) {
        fd[i] = serial_open(ports[i].c_str()); // blocking
        if (fd[i] == -1)
            return false;
        if (types[i] == "RM Young 3D") {
            if (!serial_setup(fd[i], 38400)) // N81
                return false;
        }
        else if (types[i] == "Gill 3D") {
            if (!serial_setup(fd[i], 115200)) // N81
                return false;
        }
        else
            return false; // type not recognized
    }

    // clear anemometer frames
    memset(young_frame, 0, sizeof(young_frame));

    // create thread for receiving anemometer measurements
    exit_thread = false;
    num_ports = n_ports;

    for (int i = 0; i < n_ports; i++) {
        thread_args[i].arg = &exit_thread;
        thread_args[i].index = i;
        if (types[i] == "RM Young 3D") {
            if (pthread_create(&read_thread_handle[i], NULL, &young_read_loop, (void*)&thread_args[i]) != 0)
                return false;
        }
        else if (types[i] == "Gill 3D") {
            if (pthread_create(&read_thread_handle[i], NULL, &gill_read_loop, (void*)&thread_args[i]) != 0)
                return false;
        }
    }

    if (pthread_create(&write_thread_handle, NULL, &young_write_loop, (void*)&exit_thread) != 0)
            return false;

    return true;
}

void sonic_anemometer_close(void)
{
    if (!exit_thread and num_ports) // if still running
    {
        // exit threads
        exit_thread = true;
        pthread_join(write_thread_handle, NULL);
        for (int i = 0; i < num_ports; i++)
            pthread_join(read_thread_handle[i], NULL);
        // close serial port
        for (int i = 0; i < num_ports; i++) 
            serial_close(fd[i]);
        printf("Anemometer serial thread terminated.\n");
    }
}

static void youngProcessFrame(char* buf, int len, int index)
{ 
    if (index < 0 or index >= SERIAL_MAX_ANEMOMETERS)
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

if (index == 0) {
//    printf("anemometer %d , speed = [ %f, %f, %f ], temperature = %f.\n", index, wind_data[index].speed[0], wind_data[index].speed[1], wind_data[index].speed[2], wind_data[index].temperature);
    printf("speed = %f\n", std::sqrt(wind_data[index].speed[0]*wind_data[index].speed[0]+wind_data[index].speed[1]*wind_data[index].speed[1]+wind_data[index].speed[2]*wind_data[index].speed[2]));
}

            }
            young_frame[index].pointer = 0;
            young_frame[index].checksum = 0;
        }
    }
}

static void gillProcessFrame(char* buf, int len, int index)
{
    static Gill_ASCII_Frame_t gill_frame;

    if (index < 0 or index >= SERIAL_MAX_ANEMOMETERS)
        return;

    if (len <= 0)
        return;

    for (int i = 0; i < len; i++) {
        switch (gill_frame.protocol[gill_frame.pointer]) {
            case 0x02: // start frame
                if (buf[i] == 0x02)
                    gill_frame.frame[gill_frame.pointer++] = buf[i];
                break;
            case 'X': // number, not ',' or '.', or '+', or '-'
                if (buf[i] != ',' and buf[i] != '.' and buf[i] != '+' and buf[i] != '-')
                    gill_frame.frame[gill_frame.pointer++] = buf[i];
                else
                    gill_frame.pointer = 0;
                break;
            case 'S': // '+' or '-'
                if (buf[i] == '+' or buf[i] == '-')
                    gill_frame.frame[gill_frame.pointer++] = buf[i];
                else
                    gill_frame.pointer = 0;
                break;
            case ',':
                if (buf[i] == ',')
                    gill_frame.frame[gill_frame.pointer++] = buf[i];
                else
                    gill_frame.pointer = 0;
                break;
            case '.':
                if (buf[i] == '.')
                    gill_frame.frame[gill_frame.pointer++] = buf[i];
                else
                    gill_frame.pointer = 0;
                break;
            case 0x03: // end frame
                if (buf[i] == 0x03) { // received a complete frame
                    gill_frame.frame[gill_frame.pointer++] = buf[i]; 
// Debug
    //gill_frame.frame[gill_frame.pointer] = 0x0;
    //printf("gill_frame = %s\n", gill_frame.frame);

                    gill_frame.StaA = ((short)gill_frame.frame[1] << 8) & ((short)gill_frame.frame[2]);
                    gill_frame.StaD = ((short)gill_frame.frame[4] << 8) & ((short)gill_frame.frame[5]);
                    char temp_f[10] = {0};
                    memcpy(temp_f, &gill_frame.frame[7], 6*sizeof(char));
                    gill_frame.u = atof(temp_f);
                    memcpy(temp_f, &gill_frame.frame[14], 6*sizeof(char));
                    gill_frame.v = atof(temp_f);
                    memcpy(temp_f, &gill_frame.frame[21], 6*sizeof(char));
                    gill_frame.w = atof(temp_f);
                    memcpy(temp_f, &gill_frame.frame[28], 6*sizeof(char));
                    gill_frame.T = atof(temp_f);

// Debug
    //printf("wind = [%f, %f, %f], T = %f\n", gill_frame.u, gill_frame.v, gill_frame.w, gill_frame.T);

                    // save data
                    wind_data[index].speed[0] = (float)gill_frame.u;
                    wind_data[index].speed[1] = (float)gill_frame.v;
                    wind_data[index].speed[2] = (float)gill_frame.w;
                    wind_data[index].temperature = (float)gill_frame.T;
                    // save record
                    wind_record[index].push_back(wind_data[index]);

                    gill_frame.pointer = 0; // clear pointer
                }
                break;
            default:
                break;
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
        for (int i = 0; i < num_ports; i++) {
            if (anemometer_type[i] == "RM Young 3D")
                serial_write(fd[i], command, 3);
        }
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

static void* gill_read_loop(void* args)
{
    int nbytes;
    char frame[512];
    
    while (!*((bool*)(((Thread_Arguments_t*)args)->arg)))
    {
        nbytes = serial_read(fd[((Thread_Arguments_t*)args)->index], frame, 512);
        if (nbytes > 0)
            gillProcessFrame(frame, nbytes, ((Thread_Arguments_t*)args)->index);
    }
}

std::string* sonic_anemometer_get_port_paths(void)
{
    return anemometer_port_path;
}

std::string* sonic_anemometer_get_types(void)
{
    return anemometer_type;
}

std::vector<Anemometer_Data_t>* sonic_anemometer_get_wind_record(void)
{
    return wind_record;
}

Anemometer_Data_t* sonic_anemometer_get_wind_data(void)
{
    return wind_data;
}
