#ifndef VIRTUAL_PLUME_H
#define VIRTUAL_PLUME_H

#define N_PUFFS     1000  // number of puffs per virtual plume

void release_virtual_plume(float*, float*, float*, float*, std::vector<FOC_Puff_t>*);
void calculate_virtual_tdoa_and_std(std::vector<FOC_Puff_t>*, float*, float*, std::vector<FOC_Particle_t>&);
bool estimate_horizontal_direction_according_to_tdoa(FOC_TDOA_t& delta, float* out);

#endif
