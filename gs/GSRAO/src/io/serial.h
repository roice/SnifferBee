#ifndef SERIAL_H
#define SERIAL_H

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
/* serial_mbsp.cxx */
bool mbsp_init(const char*);
void mbsp_close(void);

#endif
