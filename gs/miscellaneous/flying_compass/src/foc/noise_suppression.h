#ifndef NOISE_SUPPRESSION_H
#define NOISE_SUPPRESSION_H

#include "foc/flying_odor_compass.h"

void noise_suppression_ukf_init(void);
FOC_Reading_t noise_suppression_ukf_update(FOC_Input_t&);
bool noise_suppression_gaussian_filter(std::vector<FOC_Reading_t>* input,
        std::vector<FOC_Reading_t>* output, int order, float backtrace_time);

#endif
