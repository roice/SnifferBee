#ifndef FOC_DELTA_H
#define FOC_DELTA_H

void foc_delta_init(std::vector<FOC_Delta_t>&);
bool foc_delta_update(std::vector<FOC_Reading_t>&, std::vector<FOC_Delta_t>&);
#endif
