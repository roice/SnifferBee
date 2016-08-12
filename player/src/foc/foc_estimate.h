#ifndef FOC_ESTIMATE_H
#define FOC_ESTIMATE_H


void foc_estimate_source_direction_init(std::vector<FOC_Estimation_t>&);

bool foc_estimate_source_direction_update(std::vector<FOC_Input_t>&, std::vector<FOC_Delta_t>&, std::vector<FOC_Wind_t>&, std::vector<FOC_Estimation_t>&);

#endif
