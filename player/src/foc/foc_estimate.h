#ifndef FOC_ESTIMATE_H
#define FOC_ESTIMATE_H


void foc_estimate_source_init(std::vector<FOC_Estimation_t>&);

bool foc_estimate_source_update(std::vector<FOC_Feature_t>&, std::vector<FOC_Estimation_t>&);

#endif
