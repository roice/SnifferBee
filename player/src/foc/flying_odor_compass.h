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
#define FOC_WIND_MIN                0.1    // m/s
#define FOC_WIND_MAX                5.0     // m/s
#define FOC_SIGNAL_DELAY            2       // seconds, int
#define FOC_TDOA_DELAY              1       // seconds, int
#define FOC_MOX_DAQ_FREQ            20      // Hz, int
#define FOC_MOX_INTERP_FACTOR       10      // samples/symbol, > 4, int
#define FOC_LEN_RECENT_INFO         (15*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR)    // approx. 15 s
#define FOC_LEN_WAVELET             (3*FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR)    // approx. 3 s
#define FOC_RECENT_TIME_TO_EST      20.0    // seconds
#define FOC_WT_LEVELS               100       // wavelet transform levels
#define FOC_MAX_PARTICLES           1000     // max number of particles
#define FOC_MAX_HIST_PARTICLES      100     // max history particles
#define FOC_RECORD_LEN              1000    // seconds of history recording, int

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
    FOC_TDOA_t tdoa;
    FOC_STD_t std;
} FOC_Particle_t;

typedef struct {
    std::vector<FOC_Particle_t>* particles; // particles, virtual sources
    std::vector<FOC_Particle_t>* hist_particles;
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

typedef struct {
    int   t;      // time of this maxima occurs
    float   value;  // value of this maxima point
    int     level;  // this maxima point belongs to which level
} FOC_ModMax_t;

typedef struct {
    int levels; // the number of levels this maxline grows through
    float value[FOC_WT_LEVELS]; // value of modmax
    int t[FOC_WT_LEVELS]; // time of modmax
} FOC_Maxline_t;

typedef struct {
// foc_feature.cxx
    int type; // sign
    int idx_ml[FOC_NUM_SENSORS]; // index of data_maxline this feature extracts from
    float toa[FOC_NUM_SENSORS]; // second
    float abs_top_level_wt_value[FOC_NUM_SENSORS];
    float sum_abs_top_level_wt_value; // sum of the abs of top level wt value
    float sum_abs_tdoa; // sum of abs(tdoa), s
    float sum_llh_mls_t; // sum of likelihood of time of maxlines, -FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)  ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float sum_llh_mls_value; // sum of likelihood of value of maxlines, -FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!) ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float sum_llh_mls_levels; // sum of diff of levels of maxlines, 0. ~ FOC_NUM_SENSORS!/(2!(FOC_NUM_SENSORS-2)!)
    float belief_llh; // belief of likelihood of this comb
    float credit; // contribution of this feature to all features list
// foc_estimate.cxx
    bool valid_to_infer_direction;
    float direction[3]; // earth coord (ENU)
    float direction_p[3]; // aircraft coord
} FOC_Feature_t;

class Flying_Odor_Compass
{
    public:
        Flying_Odor_Compass(void);
        bool update(FOC_Input_t&);
        // data
        std::vector<FOC_Wind_t>         data_wind;
        std::vector<FOC_Input_t>        data_raw;
        std::vector<FOC_Reading_t>      data_denoise;
        std::vector<float>              data_interp[FOC_NUM_SENSORS];

#if 0 // Scale space method. Smooth+Diff+Edges+TDOA
        std::vector<FOC_Reading_t>      data_smooth[FOC_DIFF_GROUPS*(FOC_DIFF_LAYERS_PER_GROUP+1)];
        std::vector<FOC_Reading_t>      data_diff[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_Reading_t>      data_edge_max[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_Reading_t>      data_edge_min[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_ChangePoints_t> data_cp_max[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_ChangePoints_t> data_cp_min[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_TDOA_t>         data_tdoa[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
        std::vector<FOC_STD_t>          data_std[FOC_DIFF_GROUPS*FOC_DIFF_LAYERS_PER_GROUP];
#endif
        // memory allocated in foc_wavelet.cu
        float                           data_wvs[FOC_WT_LEVELS*FOC_LEN_WAVELET]; // wavelets of multi-scales
        std::vector<int>                data_wvs_idx; // index of every scale wavelet in data_wvs
        std::vector<float>              data_wt_out[FOC_NUM_SENSORS][FOC_WT_LEVELS]; // wavelet transform of signals
        std::vector<FOC_ModMax_t>       data_modmax[FOC_NUM_SENSORS][FOC_WT_LEVELS][2]; // modulus maxima points, points are sequentially placed in the order of level 0, 1, 2, ..., FOC_WT_LEVELS-1
        std::vector<FOC_Maxline_t>      data_maxline[FOC_NUM_SENSORS][2]; // maxima lines
        std::vector<FOC_Feature_t>      data_feature;
        std::vector<FOC_Estimation_t>   data_est;
    private:
        // unscented kalman filters
        
};

#endif
