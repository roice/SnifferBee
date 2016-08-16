#ifndef VIRTUAL_PLUME_H
#define VIRTUAL_PLUME_H

#define N_PUFFS     50  // number of puffs per virtual plume

void release_virtual_plume(float*, float*, float*, float*, std::vector<FOC_Puff_t>*);
void calculate_virtual_mox_reading(std::vector<FOC_Puff_t>*, std::vector<FOC_Reading_t>*, float*, float*);
void calculate_virtual_delta(std::vector<FOC_Reading_t>*, FOC_Delta_t&);
bool calculate_likelihood_of_virtual_delta(FOC_Delta_t&, std::vector<FOC_Particle_t>*);
bool estimate_horizontal_direction_according_to_tdoa(FOC_Delta_t& delta, float* out);

#endif
