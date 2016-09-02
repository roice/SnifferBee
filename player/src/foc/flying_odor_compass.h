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

#define FOC_NUM_SENSORS             3
#define FOC_RADIUS                  0.1     // meter
#define FOC_WIND_MIN                0.05    // m/s
#define FOC_WIND_MAX                5.0     // m/s
#define FOC_SIGNAL_DELAY            2       // seconds, int
#define FOC_TDOA_DELAY              1       // seconds, int
#define FOC_TIME_RECENT_INFO        1       // seconds, int
#define FOC_MOX_DAQ_FREQ            25      // Hz, int
#define FOC_MOX_INTERP_FACTOR       10      // samples/symbol, > 4, int
#define FOC_DIFF_LAYERS_PER_GROUP   3       // layers of difference per group, 2 <= layers
#define FOC_DIFF_GROUPS             6       // groups of difference
#define FOC_MAX_PARTICLES           100     // max number of particles
#define FOC_RECORD_LEN              600     // seconds of history recording, int

// display state of foc estimation
#define FOC_ESTIMATE_DEBUG

typedef struct {
    float mox_reading[FOC_NUM_SENSORS];
    float position[3];
    float attitude[3];
    float wind[3];
    int count;
    double time;
} FOC_Input_t; // data type input to FOC

typedef struct {
    float x;
    float y;
    float z;
} FOC_Vector_t;

typedef struct {
    float reading[FOC_NUM_SENSORS];
    double time;
} FOC_Reading_t; // mox reading data type processed in FOC

typedef struct {
    float wind[3];      // e/n/u earth coordinate
    float wind_p[3];    // plane coordinate
    float wind_filtered[3];
} FOC_Wind_t;

typedef struct {
    float toa[FOC_NUM_SENSORS]; // time of arrival
    float abs[FOC_NUM_SENSORS]; // absolute, to calculate belief later
    int index; // index of first sensor in edge sequence
    float dt;
} FOC_TDOA_t; // delta time/varince (feature extracted from mox reading)

typedef struct {
    float std[FOC_NUM_SENSORS];
} FOC_STD_t;

typedef struct {
    float pos[3];
    float r;
} FOC_Puff_t;

typedef struct {
    float pos_r[3]; // relative position from particle to robot
    float weight;
    std::vector<FOC_Puff_t>* plume; // virtual plume
    FOC_TDOA_t  tdoa;
    FOC_STD_t   std;
} FOC_Particle_t;

typedef struct {
    std::vector<FOC_Particle_t>* particles; // particles, virtual sources
    float wind_p[3];    // plane coord, x/y
    float wind[3];      // global coord, e/n
    float direction[3]; // direction of gas source
    float clustering;   // degree of aggregation
    float std[FOC_NUM_SENSORS];
    float dt;
    bool valid; // this result is valid or not
#ifdef FOC_ESTIMATE_DEBUG
    float radius_particle_to_robot;
#endif
} FOC_Estimation_t;

typedef struct {
    float   reading;
    int     index_time;
    int     index_sensor;
} FOC_Edge_t;

typedef struct {
    int index[FOC_NUM_SENSORS]; // indices of change points
    int disp;
} FOC_ChangePoints_t;

class Flying_Odor_Compass
{
    public:
        Flying_Odor_Compass(void);
        bool update(FOC_Input_t&);
        // data
        std::vector<FOC_Wind_t>         data_wind;
        std::vector<FOC_Input_t>        data_raw;
        std::vector<FOC_Reading_t>      data_denoise;
        std::vector<FOC_Reading_t>      data_interp;
        std::vector<FOC_Reading_t>      data_smooth[FOC_DIFF_GROUPS*(FOC_DIFF_LAYERS_PER_GROUP+1)];
        std::vector<FOC_Reading_t>      data_diff[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_Reading_t>      data_edge_max[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_Reading_t>      data_edge_min[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_ChangePoints_t> data_cp_max[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_ChangePoints_t> data_cp_min[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_TDOA_t>         data_tdoa[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_STD_t>          data_std[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_Estimation_t>   data_est;
    private:
        // unscented kalman filters
        
};

#endif
