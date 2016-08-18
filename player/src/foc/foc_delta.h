#ifndef FOC_DELTA_H
#define FOC_DELTA_H

#include "foc/flying_odor_compass.h"



#if defined(FOC_DELTA_METHOD_CROSS_CORRELATION)
void foc_delta_init(std::vector<FOC_Delta_t>&);
bool foc_delta_update(std::vector<FOC_Reading_t>&, std::vector<FOC_Delta_t>&);
#elif defined(FOC_DELTA_METHOD_EDGE_DETECTION)
void foc_delta_init(std::vector<FOC_ChangePoints_t>&, std::vector<FOC_ChangePoints_t>&, std::vector<FOC_Delta_t>&);
bool foc_delta_update(std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&, std::vector<FOC_ChangePoints_t>&, std::vector<FOC_ChangePoints_t>&, std::vector<FOC_Delta_t>&);
#endif

#endif
