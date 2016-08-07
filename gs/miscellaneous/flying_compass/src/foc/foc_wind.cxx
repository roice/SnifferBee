#include <string.h>
#include <vector>
#include "flying_odor_compass.h"
#include "liquid.h"

static firfilt_rrrf f_w[3]; // xyz, filter for wind

void foc_wind_smooth_init(std::vector<FOC_Wind_t>& out)
{
/* create FIR filter for Phase 0: wind estimation */
    for (int i = 0; i < 3; i++)
        f_w[i] = firfilt_rrrf_create_kaiser(FOC_SIGNAL_DELAY*FOC_MOX_DAQ_FREQ, 1.0f/FOC_MOX_DAQ_FREQ*2, 60.0, 0.0);
    out.clear();
}

void foc_wind_smooth_update(FOC_Input_t& new_in, std::vector<FOC_Wind_t>& out)
{
    FOC_Wind_t new_out;
    // TODO: 3D
    for (int i = 0; i < 2; i++) {
        firfilt_rrrf_push(f_w[i], new_in.wind[i]);
        firfilt_rrrf_execute(f_w[i], &new_out.wind_filtered[i]);
    }

    // save result
    memcpy(new_out.wind, new_in.wind, 3*sizeof(float));
    out.push_back(new_out);
}
