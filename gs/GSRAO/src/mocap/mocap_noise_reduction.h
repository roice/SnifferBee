#ifndef MOCAP_NOISE_REDUCTION_H
#define MOCAP_NOISE_REDUCTION_H

typedef struct {
    float v[3];
} Mocap_3D_Vector_t;

void mocap_noise_reduction_ukf_init(int);
Mocap_3D_Vector_t mocap_noise_reduction_ukf_update(int, Mocap_3D_Vector_t&);

#endif
