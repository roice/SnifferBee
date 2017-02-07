#include "foc/virtual_plume.h"
#include "foc/wake_qr.h"
#include "foc/error_cuda.h"

#ifdef GPU_COMPUTING

#define PI 3.14159265358979323846

/* GPU version */
__device__ void RotateVector(float* vector, float* out, float yaw, float pitch, float roll)
{
    // calculate rotation matrix
    float sin_yaw = sinf(yaw);
    float cos_yaw = cosf(yaw);
    float sin_pitch = sinf(pitch);
    float cos_pitch = cosf(pitch);
    float sin_roll = sinf(roll);
    float cos_roll = cosf(roll);
    float R_zyx[9];
    R_zyx[0] = cos_yaw*cos_pitch;
    R_zyx[1] = cos_yaw*sin_pitch*sin_roll-sin_yaw*cos_roll;
    R_zyx[2] = cos_yaw*sin_pitch*cos_roll+sin_yaw*sin_roll;
    R_zyx[3] = sin_yaw*cos_pitch;
    R_zyx[4] = sin_yaw*sin_pitch*sin_roll+cos_yaw*cos_roll;
    R_zyx[5] = sin_yaw*sin_pitch*cos_roll-cos_yaw*sin_roll;
    R_zyx[6] = -sin_pitch;
    R_zyx[7] = cos_pitch*sin_roll;
    R_zyx[8] = cos_pitch*cos_roll;
    // rotate
    out[0] = R_zyx[0]*vector[0]+R_zyx[1]*vector[1]+R_zyx[2]*vector[2];
    out[1] = R_zyx[3]*vector[0]+R_zyx[4]*vector[1]+R_zyx[5]*vector[2];
    out[2] = R_zyx[6]*vector[0]+R_zyx[7]*vector[1]+R_zyx[8]*vector[2];
}

__device__ float CompleteEllipticIntFirst(float k)
{
    return PI/2.0*(1.0 + 0.5*0.5*k*k + 0.5*0.5*0.75*0.75*(k*k*k*k));
}

__device__ float CompleteEllipticIntSecond(float k)
{
    return PI/2.0*(1.0 - 0.5*0.5*k*k - 0.5*0.5*0.75*0.75*(k*k*k*k)/3.0);
}

__device__ void InducedVelocityVortexRing(float* center_ring, float radius_ring, float Gamma_ring, float core_radius_ring, float* att_ring, float*pos, float* vel)
{
    float op[3], op_z, op_r, db_op_z, db_op_r;
    float u_z, u_r, m, a, b;

    /* Step 1: convert absolute position to relative position */
    // relative position, center_ring as origin
    for (int i = 0; i < 3; i++)
        op[i] = pos[i] - center_ring[i];

    /* Step 2: calculate relative velocity */
    float norm_vel; // norm of velocity
    float unit_vector[3] = {0}; // unit vector of the ring axis
    float db_radius = radius_ring*radius_ring;          // radius^2
    float db_delta = core_radius_ring*core_radius_ring; // core^2
   
    // calculate unit vector
    float temp_v[3] = {0.0, 0.0, 1.0};
    RotateVector(temp_v, unit_vector, att_ring[2], att_ring[1], att_ring[0]);

    if (op[0]==0.0f && op[1]==0.0f && op[2]==0.0f) { // P is at center
        norm_vel = Gamma_ring/(2*radius_ring); // norm of velocity
        for (int i = 0; i < 3; i++)
            vel[i] += -unit_vector[i]*norm_vel;
    }
    else {
        // op_z, cylindrical coord
        op_z = op[0]*unit_vector[0]+op[1]*unit_vector[1]+op[2]*unit_vector[2];
        db_op_z = op_z*op_z;
        // op_r, cylindrical coord
        db_op_r = op[0]*op[0]+op[1]*op[1]+op[2]*op[2]-db_op_z;
        if (db_op_r < 0) {
            db_op_r = 0;
            op_r = 0;
        }
        else
            op_r = sqrtf(db_op_r);
        // a, A, m
        a = sqrtf((op_r+radius_ring)*(op_r+radius_ring)+db_op_z+db_delta);
        b = (op_r-radius_ring)*(op_r-radius_ring)+db_op_z+db_delta;
        m = 4*op_r*radius_ring/(a*a);
        // u_z, cylindrical coord 
        u_z = Gamma_ring/(2*PI*a) * ((-(db_op_r-db_radius+db_op_z+db_delta)/b)
            *CompleteEllipticIntSecond(m) + CompleteEllipticIntFirst(m));
        // u_r, cylindrical coord
        u_r = Gamma_ring*op_z/(2*PI*op_r*a) * (((db_op_r+db_radius+db_op_z+db_delta)/b)
            *CompleteEllipticIntSecond(m) - CompleteEllipticIntFirst(m));
        // map u_z, u_r to cartesian coord
        norm_vel = sqrtf(u_z*u_z + u_r*u_r);
        for (int i = 0; i < 3; i++)
            vel[i] += -unit_vector[i]*norm_vel;
    }
    /* Step 3: convert relative velocity to absolute velocity */
    // as the velocity of quad-rotor is ommited, this step is skipped
}

__device__ void WakeQRCalculateVelocity(float* pos_qr, float* att_qr, float* pos, float* wind, float* vel)
{
    /* vortex ring method */
    float c_ring[4][3] = {{0}, {0}, {0}, {0}}; // center of 4 motor vortex ring
    // calculate centers of motors
    float temp_c[4][3] = {
        {QR_MOTOR_DISTANCE/2.0/1.414, QR_MOTOR_DISTANCE/2.0/1.414, 0.0},
        {-QR_MOTOR_DISTANCE/2.0/1.414, QR_MOTOR_DISTANCE/2.0/1.414, 0.0}, 
        {-QR_MOTOR_DISTANCE/2.0/1.414, -QR_MOTOR_DISTANCE/2.0/1.414, 0.0}, 
        {QR_MOTOR_DISTANCE/2.0/1.414, -QR_MOTOR_DISTANCE/2.0/1.414, 0.0}
    };
    for (int i = 0; i < 4; i++) {
        RotateVector(temp_c[i], c_ring[i], att_qr[2], att_qr[1], att_qr[0]);
        for (int j = 0; j < 3; j++)
            c_ring[i][j] += pos_qr[j];
    }

    Wake_QR_ring_t wake_qr_rings[4];
    float temp_v[3] = {0.0, 0.0, -1.0};
    float unit[3] = {0};
    RotateVector(temp_v, unit, att_qr[2], att_qr[1], att_qr[0]);
    for (int i = 0; i < QR_WAKE_RINGS; i++) { // num of rings
        for (int j = 0; j < 4; j++) { // 4 rotors
            for (int k = 0; k < 3; k++) // xyz
                wake_qr_rings[i].pos[j][k] = c_ring[j][k] + i*unit[k]*0.01; // 0.01 m gap
        }
        wake_qr_rings[i].att[0] = att_qr[0];
        wake_qr_rings[i].att[1] = att_qr[1];
        wake_qr_rings[i].att[2] = att_qr[2];
    }

    // calculate induced velocity
    vel[0] = 0; vel[1] = 0; vel[2] = 0;
    for (int i = 0; i < QR_WAKE_RINGS; i++)
        for (int j = 0; j < 4; j++)
            //InducedVelocityVortexRing(wake_qr_rings[i].pos[j], QR_PROP_RADIUS, 0.134125, 0.005, wake_qr_rings[i].att, pos, vel);
            InducedVelocityVortexRing(wake_qr_rings[i].pos[j], QR_PROP_RADIUS, 0.134125, 0.005, wake_qr_rings[i].att, pos, vel);

}

__global__ void CalculateVirtualPlumes(FOC_Particle_t* particles, FOC_Puff_t* puffs, FOC_Input_t* raw, float* wind, int num_plumes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float vel[3], pos[3];
    if (tid < num_plumes) {
        pos[0] = particles[tid].pos_r[0]+raw->position[0];
        pos[1] = particles[tid].pos_r[1]+raw->position[1];
        pos[2] = particles[tid].pos_r[2]+raw->position[2];
        for (int i = 0; i < N_PUFFS; i++) {
            // calculate puff pos and radius
            WakeQRCalculateVelocity(raw->position, raw->attitude, pos, wind, vel);
            // update puff position
            for (int j = 0; j < 3; j++) {
                pos[j] += (vel[j]+wind[j])*VIRTUAL_PLUME_DT;
                //pos[j] += (vel[j])*VIRTUAL_PLUME_DT;
                puffs[tid*N_PUFFS+i].pos[j] = pos[j];
            }
        }
    }
}

__global__ void CalculateVirtualTDOAandSTD(FOC_Particle_t* particles, FOC_Puff_t* puffs, FOC_Input_t* raw, int num_plumes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_plumes)   return;

    float* pos = raw->position;
    float* att = raw->attitude;

    // calculate position of sensors
    float pos_s[3][3] = {{0}, {0}, {0}};
    float temp_s[3][3] = { {0, FOC_RADIUS, 0},
        {FOC_RADIUS*(-0.8660254), FOC_RADIUS*(-0.5), 0},
        {FOC_RADIUS*0.8660254, FOC_RADIUS*(-0.5), 0} };
    for (int i = 0; i < 3; i++) // relative position of sensors
        RotateVector(temp_s[i], pos_s[i], att[2], att[1], att[0]);
    for (int i = 0; i < 3; i++) // absolute position of sensors
        for (int j = 0; j < 3; j++)
            pos_s[i][j] += pos[j];
    
    // calculate tdoa
    double temp_dis[3] = {
        sqrtf((puffs[tid*N_PUFFS].pos[0]-pos_s[0][0])*(puffs[tid*N_PUFFS].pos[0]-pos_s[0][0]) + (puffs[tid*N_PUFFS].pos[1]-pos_s[0][1])*(puffs[tid*N_PUFFS].pos[1]-pos_s[0][1]) + (puffs[tid*N_PUFFS].pos[2]-pos_s[0][2])*(puffs[tid*N_PUFFS].pos[2]-pos_s[0][2])), 
        sqrtf((puffs[tid*N_PUFFS].pos[0]-pos_s[1][0])*(puffs[tid*N_PUFFS].pos[0]-pos_s[1][0]) + (puffs[tid*N_PUFFS].pos[1]-pos_s[1][1])*(puffs[tid*N_PUFFS].pos[1]-pos_s[1][1]) + (puffs[tid*N_PUFFS].pos[2]-pos_s[1][2])*(puffs[tid*N_PUFFS].pos[2]-pos_s[1][2])),
        sqrtf((puffs[tid*N_PUFFS].pos[0]-pos_s[2][0])*(puffs[tid*N_PUFFS].pos[0]-pos_s[2][0]) + (puffs[tid*N_PUFFS].pos[1]-pos_s[2][1])*(puffs[tid*N_PUFFS].pos[1]-pos_s[2][1]) + (puffs[tid*N_PUFFS].pos[2]-pos_s[2][2])*(puffs[tid*N_PUFFS].pos[2]-pos_s[2][2])) }; // FOC_NUM_SENSORS = 3
    double temp_distance;
    int temp_idx[3] = {0};
    for (int i = 0; i < N_PUFFS; i++) {
        for (int j = 0; j < 3; j++) { // FOC_NUM_SENSORS = 3
            temp_distance = sqrtf((puffs[tid*N_PUFFS+i].pos[0]-pos_s[j][0])*(puffs[tid*N_PUFFS+i].pos[0]-pos_s[j][0]) + (puffs[tid*N_PUFFS+i].pos[1]-pos_s[j][1])*(puffs[tid*N_PUFFS+i].pos[1]-pos_s[j][1]) + (puffs[tid*N_PUFFS+i].pos[2]-pos_s[j][2])*(puffs[tid*N_PUFFS+i].pos[2]-pos_s[j][2]));
            if (temp_distance < temp_dis[j]) {
                temp_dis[j] = temp_distance;
                temp_idx[j] = i;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        particles[tid].tdoa.toa[i] = (float)(temp_idx[i])*VIRTUAL_PLUME_DT;
    }

    // calculate standard deviation
    double distance[3] = {temp_dis[0], temp_dis[1], temp_dis[2]};
    for (int i = 0; i < 3; i++)
        //particles[tid].std.std[i] = 10.0/exp(distance[i]*distance[i]);
        particles[tid].std.std[i] = 1.0/(distance[i]*distance[i]);
        //particles[tid].std.std[i] = temp_dis[0];
        //particles[tid].std.std[i] = temp_idx[i];
}

#define NUM_DIST_PROJECTION  20
__global__ void CalculateWeights(FOC_Particle_t* particles, FOC_Input_t* raw, float* wind, float* std, int num_plumes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_plumes)   return;

    float* pos = raw->position;
    float* att = raw->attitude;

    // calculate position of sensors
    float pos_s[3][3] = {{0}, {0}, {0}};
    float temp_s[3][3] = { {0, FOC_RADIUS, 0},
        {FOC_RADIUS*(-0.8660254), FOC_RADIUS*(-0.5), 0},
        {FOC_RADIUS*0.8660254, FOC_RADIUS*(-0.5), 0} };
    for (int i = 0; i < 3; i++) // relative position of sensors
        RotateVector(temp_s[i], pos_s[i], att[2], att[1], att[0]);
    for (int i = 0; i < 3; i++) // absolute position of sensors
        for (int j = 0; j < 3; j++)
            pos_s[i][j] += pos[j];

    // calculate e_wind
    float e_wind[2];
    float nrm_wind = sqrtf(wind[0]*wind[0]+wind[1]*wind[1]);
    if (nrm_wind == 0) {
        particles[tid].weight = 0;
        return;
    }
    for (int i = 0; i < 2; i++)
        e_wind[i] = wind[i]/nrm_wind;

    // distance from sensor to points on e_wind
    float dist[3][NUM_DIST_PROJECTION]; // FOC_NUM_SENSORS = 3
    float temp_pos[3] = {0};
    for (int i = 0; i < NUM_DIST_PROJECTION; i++) {
        for (int j = 0; j < 2; j++) {
            temp_pos[j] = e_wind[j]*((float)i*(2.0*FOC_RADIUS)/(float)NUM_DIST_PROJECTION-FOC_RADIUS);
            temp_pos[j] += pos[j];
        }
        temp_pos[2] = pos[2];
        for (int j = 0; j < 3; j++)
            dist[j][i] = sqrtf((pos_s[j][0]-temp_pos[0])*(pos_s[j][0]-temp_pos[0]) + (pos_s[j][1]-temp_pos[1])*(pos_s[j][1]-temp_pos[1]) + (pos_s[j][2]-temp_pos[2])*(pos_s[j][2]-temp_pos[2]));
    }

    // sequence
    float virt_seq[NUM_DIST_PROJECTION] = {0}, real_seq[NUM_DIST_PROJECTION] = {0};
    for (int i = 0; i < NUM_DIST_PROJECTION; i++) {
        for (int j = 0; j < 3; j++) {
            //virt_seq[i] += particles[tid].std.std[j]*1.0/exp(dist[j][i]*dist[j][i]);//*std::exp(dist[j][i]*dist[j][i]);
            //real_seq[i] += std[j]*1.0/exp(dist[j][i]*dist[j][i]);//*std::exp(dist[j][i]*dist[j][i]);
            virt_seq[i] += particles[tid].std.std[j]*1.0/(dist[j][i]*dist[j][i]);//*std::exp(dist[j][i]*dist[j][i]);
            real_seq[i] += std[j]*1.0/(dist[j][i]*dist[j][i]);//*std::exp(dist[j][i]*dist[j][i]);
        }
    }
    
    // normalize
    double sum_virt_seq = 0, sum_real_seq = 0;
    double mean_virt_seq, mean_real_seq;
    for (int i = 0; i < NUM_DIST_PROJECTION; i++) {
        sum_virt_seq += virt_seq[i];
        sum_real_seq += real_seq[i];
    }
    if (sum_virt_seq == 0 or sum_real_seq == 0) {
        particles[tid].weight = 0;
        return;
    }
    mean_virt_seq = sum_virt_seq / (float)NUM_DIST_PROJECTION;
    mean_real_seq = sum_real_seq / (float)NUM_DIST_PROJECTION;
    for (int i = 0; i < NUM_DIST_PROJECTION; i++) {
        virt_seq[i] -= mean_virt_seq;
        real_seq[i] -= mean_real_seq;
    }
    sum_virt_seq = 0; sum_real_seq = 0;
    float sum_virt_real_seq = 0;
    for (int i = 0; i < NUM_DIST_PROJECTION; i++) {
        sum_virt_seq += virt_seq[i]*virt_seq[i];
        sum_real_seq += real_seq[i]*real_seq[i];
        sum_virt_real_seq += virt_seq[i]*real_seq[i];
    }
    float corr = sum_virt_real_seq / sqrt(sum_virt_seq*sum_real_seq);
    if (corr < 0) {
        particles[tid].weight = 0;
        return;
    }
    
    // correlation
    //float corr = 0;
    //for (int i = 0; i < NUM_DIST_PROJECTION; i++) {
    //    corr += virt_seq[i]*real_seq[i];
    //}
    particles[tid].weight = corr;
    //particles[tid].weight = real_seq[tid];
}

void release_virtual_plumes_and_calculate_weights(std::vector<FOC_Particle_t>* particles, float* position, float* attitude, float* est_wind, float* raw_std)
{
    /* get the properties of all the graphic cards this machine has */
    cudaDeviceProp prop;
    int count; // number of devices
    HANDLE_ERROR( cudaGetDeviceCount(&count) );
    
    if (count <= 0) {// no graphic card found
        return;
    }
    else    // default the 1st
        HANDLE_ERROR( cudaGetDeviceProperties(&prop, 0) );

    // save position and attitude to raw
    FOC_Input_t raw;
    memcpy(raw.position, position, 3*sizeof(float));
    memcpy(raw.attitude, attitude, 3*sizeof(float));

    // puffs
    FOC_Puff_t*     host_puffs;
    FOC_Particle_t* host_particles;
    FOC_Puff_t*     dev_puffs;
    FOC_Particle_t* dev_particles;
    FOC_Input_t*    dev_raw;    // contains pos, att...
    float*          dev_wind;
    float*          dev_std;
    
    int num_plumes = particles->size();

    // Phase 1: allocate memory
    // allocate host memory for host_puffs
    HANDLE_ERROR( cudaHostAlloc((void**)&host_puffs, 
        num_plumes*N_PUFFS*sizeof(*host_puffs), cudaHostAllocDefault) );
    // allocate host memory for host_particles
    HANDLE_ERROR( cudaHostAlloc((void**)&host_particles, 
        num_plumes*sizeof(*host_particles), cudaHostAllocDefault) );
    // allocate device memory for dev_puffs
    HANDLE_ERROR( cudaMalloc((void**)&dev_puffs,
        num_plumes*N_PUFFS*sizeof(*dev_puffs)) );
    // allocate device memory for dev_particles
    HANDLE_ERROR( cudaMalloc((void**)&dev_particles,
        num_plumes*sizeof(*dev_particles)) );
    // allocate device memory for dev_raw
    HANDLE_ERROR( cudaMalloc((void**)&dev_raw, sizeof(*dev_raw)) );
    // allocate device memory for dev_wind
    HANDLE_ERROR( cudaMalloc((void**)&dev_wind, 3*sizeof(*dev_wind)) );
    // allocate device memory for dev_std
    HANDLE_ERROR( cudaMalloc((void**)&dev_std, FOC_NUM_SENSORS*sizeof(*dev_std)) );

    // Phase 2: copy info to GPU's memory
    for (int i = 0; i < num_plumes; i++)
        memcpy(&(host_particles[i]), &(particles->at(i)), sizeof(FOC_Particle_t));
    HANDLE_ERROR( cudaMemcpy(dev_particles, host_particles, 
        num_plumes*sizeof(*host_particles), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_raw, &raw, 
                sizeof(FOC_Input_t), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_wind, est_wind, 
                3*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_std, raw_std, 
                3*sizeof(float), cudaMemcpyHostToDevice) );

    // Phase 3: release virtual plumes & calculate tdoa/std
    int threads = std::min(prop.warpSize, prop.maxThreadsPerBlock);
    int blocks = (num_plumes + threads -1)/threads;
    CalculateVirtualPlumes<<<blocks, threads>>>(dev_particles, dev_puffs, dev_raw, dev_wind, num_plumes);
    CalculateVirtualTDOAandSTD<<<blocks, threads>>>(dev_particles, dev_puffs, dev_raw, num_plumes);
    CalculateWeights<<<blocks, threads>>>(dev_particles, dev_raw, dev_wind, dev_std, num_plumes);

    // Phase 4: copy cuda memory to CPU memory
    HANDLE_ERROR( cudaMemcpy(host_particles, dev_particles, 
                num_plumes*sizeof(*host_particles), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(host_puffs, dev_puffs, 
                num_plumes*N_PUFFS*sizeof(*host_puffs), cudaMemcpyDeviceToHost) );

    // Phase 5: save puffs, tdoa/std and weights info
    for (int i = 0; i < num_plumes; i++) {
        particles->at(i).plume->clear();
        for (int j = 0; j < N_PUFFS; j++)
            particles->at(i).plume->push_back(host_puffs[i*N_PUFFS+j]);
        memcpy(&(particles->at(i).tdoa), &(host_particles[i].tdoa), sizeof(FOC_TDOA_t));
        memcpy(&(particles->at(i).std), &(host_particles[i].std), sizeof(FOC_STD_t));
        particles->at(i).weight = host_particles[i].weight;
    }

    // Phase 6: free memory
    // free device memory
    HANDLE_ERROR( cudaFree(dev_puffs) );
    HANDLE_ERROR( cudaFree(dev_raw) );
    HANDLE_ERROR( cudaFree(dev_wind) );
    // free host memory
    HANDLE_ERROR( cudaFreeHost(host_puffs) );
    HANDLE_ERROR( cudaFreeHost(host_particles) );
}

#endif
