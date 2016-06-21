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

#define FOC_NUM_SENSORS     3
#define FOC_RADIUS          0.1 // meter
#define FOC_MOX_DAQ_FREQ    10 // 10 Hz
#define FOC_RECORD_LEN      1000

typedef struct {
    float mox_reading[FOC_NUM_SENSORS];
    float attitude[3];
    double time;
} FOC_Input_t;

typedef struct {
    float smoothed_mox_reading[FOC_NUM_SENSORS];
    double time;
} FOC_State_t;

class Flying_Odor_Compass
{
    public:
        Flying_Odor_Compass(void);
        void update(FOC_Input_t&);
        // data
        std::vector<FOC_Input_t> foc_input;
        std::vector<FOC_State_t> foc_state;
    private:
        // unscented kalman filters
        float sensor_reading_var_process_noise;
        float sensor_reading_var_measurement_noise;
        void* sensor_reading_filter[FOC_NUM_SENSORS]; // ukf filters
        void* sensor_reading_state[FOC_NUM_SENSORS]; // state vectors
        void* sensor_reading_sys[FOC_NUM_SENSORS]; // system model
        void* sensor_reading_mm[FOC_NUM_SENSORS]; // measurement model
        void* sensor_reading_z[FOC_NUM_SENSORS]; // measurement
};

#endif
