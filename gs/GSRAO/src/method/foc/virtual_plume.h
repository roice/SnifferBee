#ifndef VIRTUAL_PLUME_H
#define VIRTUAL_PLUME_H

#include <vector>
#include "method/foc/flying_odor_compass.h"

#define N_PUFFS     600  // number of puffs per virtual plume
#define     VIRTUAL_PLUME_DT    0.01   // second

#ifndef GPU_COMPUTING
#define GPU_COMPUTING
#endif

#ifndef GPU_COMPUTING
// CPU version
void release_virtual_plume(float*, float*, float*, float*, std::vector<FOC_Puff_t>*);
void calculate_virtual_tdoa_and_std(std::vector<FOC_Puff_t>*, float*, float*, FOC_Particle_t&);
#else
// GPU verstion
void release_virtual_plumes_and_calculate_weights(std::vector<FOC_Particle_t>*, float*, float*, float*, float*);
#endif

#endif
