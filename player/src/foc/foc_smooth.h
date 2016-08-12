#ifndef FOC_SMOOTH_H
#define FOC_SMOOTH_H

void foc_smooth_init(std::vector<FOC_Reading_t>&, int, float, float, float);
bool foc_smooth_update(std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&);
#endif
