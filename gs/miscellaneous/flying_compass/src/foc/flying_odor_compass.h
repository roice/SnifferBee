/*
 * Flying Odor Compass
 *
 * This technique digs information from three MOX gas sensors which are
 * equally spaced under propellers of a quadrotor
 *
 * Author:
 *      Roice Luo
 * Date:
 *      2016.06.17
 */

#ifndef FLYING_ODOR_COMPASS
#define FLYING_ODOR_COMPASS

#include <vector>

#define FOC_NUM_SENSORS         3
#define FOC_RADIUS              0.1     // meter
#define FOC_WIND_MIN            0.05    // m/s
#define FOC_WIND_MAX            2.0     // m/s
#define FOC_DELAY               5       // seconds, int
#define FOC_MOX_DAQ_FREQ        10      // Hz, int
#define FOC_MOX_INTERP_FACTOR   10      // samples/symbol, > 4, int
#define FOC_RECORD_LEN          600     // seconds of history recording, int

typedef struct {
    float mox_reading[FOC_NUM_SENSORS];
    float position[3];
    float attitude[3];
    double time;
} FOC_Input_t; // data type input to FOC

typedef struct {
    float reading[FOC_NUM_SENSORS];
    double time;
} FOC_Reading_t; // mox reading data type processed in FOC

typedef struct {
    float dir[3];
} FOC_Wind_t;

typedef struct {
    float toa[FOC_NUM_SENSORS]; // time of arrival
    float std[FOC_NUM_SENSORS]; // standard deviation
} FOC_Delta_t; // delta time/varince (feature extracted from mox reading)

typedef struct {
    float wind_speed_xy[2]; // plane coord, x/y
    float wind_speed_en[2]; // global coord, e/n
    float wind_speed_filtered_xy[2];
    bool valid; // this result is valid or not
} FOC_Estimation_t;

class Flying_Odor_Compass
{
    public:
        Flying_Odor_Compass(void);
        bool update(FOC_Input_t&);
        // data
        std::vector<FOC_Wind_t> data_wind;
        std::vector<FOC_Input_t> data_raw;
        std::vector<FOC_Reading_t> data_denoise;
        std::vector<FOC_Reading_t> data_interp;
        std::vector<FOC_Reading_t> data_smooth;
        std::vector<FOC_Reading_t> data_diff;
        std::vector<FOC_Delta_t> data_delta;
        std::vector<FOC_Estimation_t> data_est;
    private:
        // unscented kalman filters
        
};

#endif