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
#define FOC_DELAY               20      // seconds
#define FOC_MOX_DAQ_FREQ        10      // Hz
#define FOC_MOX_INTERP_FACTOR   10      // samples/symbol
#define FOC_RECORD_LEN          600     // seconds of history recording

typedef struct {
    float mox_reading[FOC_NUM_SENSORS];
    float attitude[3];
    double time;
} FOC_Input_t;

typedef struct {
    float reading[FOC_NUM_SENSORS];
    double time;
} FOC_Reading_t;

class Flying_Odor_Compass
{
    public:
        Flying_Odor_Compass(void);
        bool update(FOC_Input_t&);
        // data
        std::vector<FOC_Input_t> data_raw;
        std::vector<FOC_Reading_t> data_denoise;
        std::vector<FOC_Reading_t> data_interp;
        std::vector<FOC_Reading_t> data_diff;
        std::vector<double> data_peak_time[FOC_NUM_SENSORS];
    private:
        // unscented kalman filters
        
};

#endif
