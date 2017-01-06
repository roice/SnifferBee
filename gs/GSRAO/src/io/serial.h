#ifndef SERIAL_H
#define SERIAL_H

#include <string>
#include <vector>

/* serial.cxx */
int serial_open(const char* port);
bool serial_setup(int fd, int baud);
bool serial_write(int, char*, int);
int serial_read(int, char*, int);
void serial_close(int fd);

/* serial_spp.cxx */
typedef struct {
    int throttle;
    int roll;
    int pitch;
    int yaw;
} SPP_RC_DATA_t;
bool spp_init(const char*);
void spp_close(void);
SPP_RC_DATA_t* spp_get_rc_data(void);

typedef struct {
    int index;
    void* arg;
} Thread_Arguments_t;

/* serial_mbsp.cxx */
#define MB_MEASUREMENTS_INCLUDE_MOTOR_VALUE    // measurements include motor value
bool mbsp_init(std::string*, int);
void mbsp_close(void);

/* serial_anemometer.cxx */
#define SERIAL_MAX_ANEMOMETERS    10

typedef struct {
    float speed[3];
    float temperature;
} Anemometer_Data_t;

bool sonic_anemometer_init(int, std::string*, std::string*);
void sonic_anemometer_close(void);
std::string* sonic_anemometer_get_port_paths(void);
std::string* sonic_anemometer_get_types(void);
Anemometer_Data_t* sonic_anemometer_get_wind_data(void);
std::vector<Anemometer_Data_t>* sonic_anemometer_get_wind_record(void);

#endif
