/**
 * PPM encoder
 *          serial protocol for PPM encoder (spp)
 * This project receives command about RC vaules from PC via serial port and g-
 * enerate PPM signals for every UAV. The signals are connected to TX modules.
 *
 * Author: Roice Luo (Bing Luo)
 * Date:   2016-05-04 create this file
 */

// SPP protocol state
typedef enum {
    IDLE,
    HEADER_START,
    HEADER_P,
    DATA
} sppState_e;

bool sppFrameParsing(unsigned char* frame, unsigned int len);
