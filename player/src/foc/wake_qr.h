#ifndef WAKE_QR_H
#define WAKE_QR_H

#include <vector>

#define QR_MOTOR_DISTANCE       0.22     // m
#define QR_PROP_RADIUS          0.0725   // m
#define QR_WAKE_RINGS           1        // 4 wake rings

typedef struct {
    float pos[4][3];
    float att[3];
} Wake_QR_ring_t;

#ifndef GPU_COMPUTING
void wake_qr_calculate_velocity(float*, float*, float*, float*, float*);
std::vector<Wake_QR_ring_t>* wake_qr_get_info_vortex_rings(void);
#endif

#endif
