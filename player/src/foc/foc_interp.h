#ifndef FOC_INTERP_H
#define FOC_INTERP_H

void foc_interp_init(std::vector<FOC_Reading_t>&, int, int, float);
bool foc_interp_update(FOC_Reading_t&, std::vector<FOC_Reading_t>&);
#endif
