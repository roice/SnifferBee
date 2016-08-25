#ifndef FOC_DIFF_H
#define FOC_DIFF_H

void foc_diff_init(std::vector<FOC_Reading_t>*);

bool foc_diff_update(std::vector<FOC_Reading_t>*, std::vector<FOC_Reading_t>*);

#if 0
bool foc_diff_update(std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>*);
#endif

#endif
