#ifndef FOC_INTERP_H
#define FOC_INTERP_H

void foc_interp_init(std::vector<float>*, int, int, float);
bool foc_interp_update(float*, std::vector<float>*);
#endif
