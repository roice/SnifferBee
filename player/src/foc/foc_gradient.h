#ifndef FOC_GRADIENT_H
#define FOC_GRADIENT_H

void foc_gradient_init(std::vector<FOC_Reading_t>&);
bool foc_gradient_update(std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&);

#endif
