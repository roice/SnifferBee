#ifndef FOC_STD_H
#define FOC_STD_H

void foc_std_init(std::vector<FOC_STD_t>&);
bool foc_std_update(std::vector<FOC_Reading_t>*, std::vector<FOC_STD_t>&);
#endif
