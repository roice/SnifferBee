#ifndef FOC_TDOA_H
#define FOC_TDOA_H

#include "foc/flying_odor_compass.h"

void foc_tdoa_init(std::vector<FOC_ChangePoints_t>*, std::vector<FOC_ChangePoints_t>*, std::vector<FOC_TDOA_t>*);
bool foc_tdoa_update(std::vector<FOC_Reading_t>*, std::vector<FOC_Reading_t>*, std::vector<FOC_Reading_t>*, std::vector<FOC_ChangePoints_t>*, std::vector<FOC_ChangePoints_t>*, std::vector<FOC_TDOA_t>*);

#endif
