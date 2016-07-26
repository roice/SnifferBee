#ifndef FOC_NOISE_REDUCTION_H
#define FOC_NOISE_REDUCTION_H

#include "foc/flying_odor_compass.h"

void foc_noise_reduction_ukf_init(void);
FOC_Reading_t foc_noise_reduction_ukf_update(FOC_Input_t&);
bool foc_noise_reduction_gaussian_filter(std::vector<FOC_Reading_t>* input,
        std::vector<FOC_Reading_t>* output, int order, float backtrace_time);

#endif
