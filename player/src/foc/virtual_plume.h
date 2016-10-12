#ifndef VIRTUAL_PLUME_H
#define VIRTUAL_PLUME_H

#define N_PUFFS     200  // number of puffs per virtual plume

// CPU version
void release_virtual_plume(float*, float*, float*, float*, std::vector<FOC_Puff_t>*);
void calculate_virtual_tdoa_and_std(std::vector<FOC_Puff_t>*, float*, float*, FOC_Particle_t&);
// GPU verstion
void release_virtual_plumes_and_calculate_virtual_tdoa_std(std::vector<FOC_Particle_t>*, FOC_Input_t&, float*);

bool estimate_horizontal_direction_according_to_tdoa(FOC_TDOA_t& delta, float* out);

#endif
