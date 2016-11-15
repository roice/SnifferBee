#ifndef FOC_INTERP_H
#define FOC_INTERP_H

void foc_interp_init(std::vector<double>*, int, int, float);
bool foc_interp_update(float*, std::vector<double>*);
#endif
