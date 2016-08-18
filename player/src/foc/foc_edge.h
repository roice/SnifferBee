#ifndef FOC_EDGE_H
#define FOC_EDGE_H

void foc_edge_init(std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&);
bool foc_edge_update(std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&);

#endif
