/*
 * Parse data received from the bear
 *
 * Author:  Roice Luo
 * Date:    2017.07.03
 */

#include <stdio.h>
//#include "robot/robot.h"

#define MAX_PACKETSIZE      4096  //Max length of buffer

/* Data Type */
#define     PIOBEAR_DATA_HEARTBEAT    100

typedef enum {
    PARSE_IDLE,
    PARSE_START,    // '$'
    PARSE_R,        // 'R'
    PARSE_CMD,      // command
    PARSE_LEN,      // length of data
    PARSE_DATA,     // data
} parserState_e;

typedef struct {
    char command;
    unsigned char len_data; // cannot be longer than 256
    char data[MAX_PACKETSIZE];
} parserData_t;

static char parser_checksum = 0;
static char parser_count = 0;
static char parser_state = PARSE_IDLE;
static parserData_t parser_data;

static void validate_command(parserData_t   &data)
{
    switch (data.command) {
        case PIOBEAR_DATA_HEARTBEAT:
        {
            if (data.len_data != sizeof(float))
                break;
            float bat_volt; int idx = 0;
            bat_volt = *((float*)&data.data[idx]);
            idx += sizeof(float);
            if (bat_volt == bat_volt) // to prevent generatin nan
                printf("Bear's battery voltage is %02.1f V.\n", bat_volt);
        }
            break;
        default:
            break;
    }
}

static void parser_reinit(void)
{
    parser_checksum = 0;
    parser_count = 0;
    parser_state = PARSE_IDLE;
}

static void parse_byte(char c)
{
    static unsigned char count = 0;

    if (parser_state == PARSE_IDLE) {
        if (c == '$')
            parser_state = PARSE_START;
    }
    else if (parser_state == PARSE_START) {
        if (c == 'R') {
            parser_state = PARSE_R;
            parser_checksum = 0;
        }
        else
            parser_state = PARSE_IDLE;
    }
    else if (parser_state == PARSE_R) {
        parser_data.command = c;
        parser_checksum ^= c;
        parser_state = PARSE_CMD;
        count = 0; // count receiving of len_data (4 bytes)
    }
    else if (parser_state == PARSE_CMD) {
        parser_data.len_data = c;
        parser_checksum ^= c;
        parser_state = PARSE_DATA;
        count = 0;
    }
    else if (parser_state == PARSE_DATA) {
        if (count == parser_data.len_data) {
            // this case include len_data > 0 and len_data = 0
            // validate the checksum
            if (parser_checksum == c)
                validate_command(parser_data);  // execute command
            else
                parser_state = PARSE_IDLE;
        }
        else {// len_data > 0
            parser_data.data[count++] = c;
            parser_checksum ^= c;
        }
    }
    else
        parser_reinit();
}

bool piobear_parse_data(char *buf, int size)
{
    parser_reinit();
    for (int i = 0; i < size; i++)
        parse_byte(buf[i]);
}
