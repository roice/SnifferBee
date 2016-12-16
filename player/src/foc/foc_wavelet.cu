#include <vector>
#include <string>
#include <cmath>
#include "foc/error_cuda.h"
#include "flying_odor_compass.h"

// data pointers for GPU computation
static float *data_wt_in;
static float *dev_data_wt_in;
static float *dev_data_wt_out;
static float *data_wavelets;
static float *dev_data_wavelets;

/* 1st order derivative of gaussian wavelet
   f'(x)=-x/(sigma^3*sqrt(2*pi))*e^(-x^2/(2*sigma^2))
 */
float wavelet_gauss_d1_psi(float x, float sigma)
{
    return -x*std::exp(-0.5*std::pow(x,2)/std::pow(sigma,2))/(std::pow(sigma,3)*std::sqrt(2*M_PI));
}

/* Make wavelets
    wvs             array containing wavelets, len(wvs)=len*num_levels
    len             sample window of wavelets, M
    num_levels      number of levels
 */
bool sample_wavelets(std::string wavelet_name, float* wvs, int len, int num_levels)
{
    if (wavelet_name != "gauss_d1" or wavelet_name != "gauss_d2")
        return false;
    if (!wvs)
        return false;
    if (len <= 0 or num_levels <= 0)
        return false;

    // sample wavelets
    float scale; // scale = (num_levels-level_idx/num_levels)*10
    float sum_h; // for normalization
    memset(wvs, 0, len*FOC_WT_LEVELS*sizeof(float));
    for (int i = 0; i < num_levels; i++) {
        //scale = (float)(num_levels-i)/(float)num_levels*50.0; // linear
        scale = (std::pow(2, (float)(num_levels-i)/(float)num_levels) - 1)*50.0;
        for (int j = 0; j < len; j++) {
            if (wavelet_name == "gauss_d1")
                wvs[i*len+j] = wavelet_gauss_d1_psi(scale*(float)(j-len/2)/(float)(FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR), 0.5);
        }
        // normalize
        sum_h = 0;
        for (int j = 0; j < len; j++)
            sum_h += std::pow(wvs[i*len+j],2);
        for (int j = 0; j < len; j++)
            wvs[i*len+j] /= sum_h;
    }

    return true;
}

/* Continuous Wavelet Transform
 */
void foc_cwt_init(float **addr_data_wvs, std::vector<int>& data_wvs_idx, float **data_wt_out, std::vector<int>& data_wt_idx)
{
    // index of data_wt_out
    data_wt_idx.reserve(FOC_WT_LEVELS);
    data_wt_idx.clear();
    for (int i = 0; i < FOC_WT_LEVELS; i++)
        data_wt_idx.push_back(i*N);
    // index of data_wavelets, for the debugging purpose
    data_wvs_idx.reserve(FOC_WT_LEVELS);
    data_wvs_idx.clear();
    for (int i = 0; i < FOC_WT_LEVELS; i++)
        data_wvs_idx.push_back(i*FOC_LEN_WAVELET);

/* allocate memory for wavelet tranformation of signals */
    // allocate a page-locked host memory & device memory containing all of the output wavelet levels
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        HANDLE_ERROR( cudaHostAlloc((void**)&(data_wt_out[i]), N*FOC_WT_LEVELS*sizeof(float), cudaHostAllocDefault) ); 
    }
    HANDLE_ERROR( cudaMalloc((void**)&dev_data_wt_out, L*FOC_WT_LEVELS*sizeof(float)) );
    // allocate a page-locked host memory & device memory containing signals
    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        HANDLE_ERROR( cudaHostAlloc((void**)&data_wt_in[i], L*FOC_WT_LEVELS*sizeof(float), cudaHostAllocDefault) );
        HANDLE_ERROR( cudaMalloc((void**)&dev_data_wt_in[i], L*FOC_WT_LEVELS*sizeof(float)) );
    }
    // allocate host memory for multi-level wavelets, for the debugging purpose
    HANDLE_ERROR( cudaHostAlloc((void**)addr_data_wvs, FOC_LEN_WAVELET*FOC_WT_LEVELS*sizeof(float), cudaHostAllocDefault) );

/* prepare for the FFT of wavelets */
    cufftHandle plan_wavelets;
    float *data_wavelets, *dev_data_wavelets;
    HANDLE_ERROR( cudaHostAlloc((void**)&data_wavelets, L*FOC_WT_LEVELS*sizeof(float), cudaHostAllocDefault) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_data_wavelets, L*FOC_WT_LEVELS*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&fft_of_wavelets, sizeof(cufftComplex)*L*FOC_WT_LEVELS) );
    HANDLE_ERROR( cudaMalloc((void**)&fft_of_signals, sizeof(cufftComplex)*L*FOC_WT_LEVELS) );
    HANDLE_ERROR( cudaMalloc((void**)&fft_of_results, sizeof(cufftComplex)*L*FOC_WT_LEVELS) );
    if (!sample_wavelets("gauss_d1", *addr_data_wvs, FOC_LEN_WAVELET, data_wavelets, L, FOC_WT_LEVELS)) // make wavelets
        printf("Sample wavelets error: L smaller than M\n");
    HANDLE_ERROR( cudaMemcpy(dev_data_wavelets, data_wavelets, L*FOC_WT_LEVELS*sizeof(float), cudaMemcpyHostToDevice) ); // copy memory to device
    // calculate FFT of wavelets
    cufftPlan1d(&plan_wavelets, L, CUFFT_R2C, FOC_WT_LEVELS);
    cufftExecR2C(plan_wavelets, dev_data_wavelets, fft_of_wavelets);
    cufftDestroy(plan_wavelets);
    HANDLE_ERROR( cudaFreeHost(data_wavelets) );
    HANDLE_ERROR( cudaFree(dev_data_wavelets) );

    // FFT plans
    cufftPlan1d(&plan_signals, L, CUFFT_R2C, FOC_WT_LEVELS); // FFT
    cufftPlan1d(&plan_results, L, CUFFT_C2R, FOC_WT_LEVELS); // IFFT
}

/* Complex multiplication
    c[i] = a[i]*b[i]*scale
 */
__global__ void ComplexPointWiseScalingMul(cufftComplex *a, cufftComplex *b, cufftComplex *c, int size, float scale)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    cufftComplex temp;

    if (tid < size) {
        temp = cuCmulf(a[tid], b[tid]);
        c[tid] = make_cuFloatComplex(scale*cuCrealf(temp), scale*cuCimagf(temp));
    }
}

/* Continuous Wavelet Transform
 */
bool foc_cwt_update(std::vector<float> *signal, float **data_wt_out, std::vector<int>& data_wt_idx)
{
    if (signal[0].size() < FOC_LEN_RECENT_INFO) // signal[0..FOC_NUM_SENSORS-1] have the same length
        return false;

    // prepare for the data to batch transform
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        memset(data_wt_in[idx], 0, sizeof(float)*L*FOC_WT_LEVELS);
        for (int i = 0; i < FOC_WT_LEVELS; i++)
            std::copy(signal[idx].end()-FOC_LEN_RECENT_INFO, signal[idx].end(), &(data_wt_in[idx][i*L]));
    }

/* Wavele Transform */  
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        // CWT Phase 0: prepare for data
        HANDLE_ERROR( cudaMemcpy(dev_data_wt_in[idx], data_wt_in[idx], L*FOC_WT_LEVELS*sizeof(float), cudaMemcpyHostToDevice) );
        // CWT Phase 1: FFT of signals (FFT of wavelets had been calculated in init)
        cufftExecR2C(plan_signals, dev_data_wt_in[idx], fft_of_signals);
        // CWT Phase 2: Complex multiply of fft_signals and fft_wavelets
        //              cuFFT performs un-normalized FFTs, so the user should scale either transform (FFT or IFFT) by the reciprocal of the size of the data set
        ComplexPointWiseScalingMul<<<(L*FOC_WT_LEVELS+512-1)/512, 512>>>(fft_of_signals, fft_of_wavelets, fft_of_results, L*FOC_WT_LEVELS, 1.0/L);
        // CWT Phase 3: IFFT (top FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1 data valid)
        cufftExecC2R(plan_results, fft_of_results, dev_data_wt_out);
        for (int i = 0; i < FOC_WT_LEVELS; i++)
            HANDLE_ERROR( cudaMemcpy(&(data_wt_out[idx][data_wt_idx[i]]), &dev_data_wt_out[i*L], (FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)*sizeof(float), cudaMemcpyDeviceToHost) );
    }

    return true;
}

/* Identify modmax
    a level of a signal is processed in a thread
 */
__global__ void IdentifyModMax(FOC_ModMax_t *modmax, int *modmax_num, float *wt_out)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid < FOC_NUM_SENSORS*FOC_WT_LEVELS) {
        int i, idx_modmax, offset = tid*(FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1);
        // Non-maximum suppression
        // traverse every points to find the local maxima
        modmax_num[tid] = 0;
        i = offset + 1; // +1 for comparison
        idx_modmax = offset;
        while (i < (offset+FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)-1) // -1 for comparison
        {
            if (wt_out[i] > wt_out[i+1]) {
                if (wt_out[i] >= wt_out[i-1]) {
                    modmax[idx_modmax].t = i - offset;
                    modmax[idx_modmax].value = wt_out[i];
                    modmax[idx_modmax].level = tid%FOC_WT_LEVELS;
                    idx_modmax++;
                }
            }
            else {
                i++;
                while (i < (offset+FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)-1 and wt_out[i] <= wt_out[i+1])
                    i++;
                if (i < (offset+FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)-1) {
                    modmax[idx_modmax].t = i - offset;
                    modmax[idx_modmax].value = wt_out[i];
                    modmax[idx_modmax].level = tid%FOC_WT_LEVELS;
                    idx_modmax++;
                }
            }
            i=i+2;
        }
        modmax_num[tid] = idx_modmax - offset;
    }
}

bool foc_identify_modmax(std::vector<FOC_ModMax_t>* data_modmax, int *data_modmax_num, float **data_wt_out, std::vector<int>& data_wt_idx)
{
    if (data_wt_idx.size() == 0)
        return false;

    // re-calculate modulus maxima
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        data_modmax[i].clear();

#if 0
/* Calculate using GPU
   memcpy to vectors costs most of the time
 */
    FOC_ModMax_t *modmax, *dev_modmax;
    int *dev_modmax_num;
    float *dev_wt_out;
    HANDLE_ERROR( cudaHostAlloc((void**)&modmax, (FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)*FOC_WT_LEVELS*FOC_NUM_SENSORS*sizeof(FOC_ModMax_t), cudaHostAllocDefault) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_modmax, sizeof(FOC_ModMax_t)*(FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)*FOC_WT_LEVELS*FOC_NUM_SENSORS) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_modmax_num, sizeof(int)*FOC_NUM_SENSORS*FOC_WT_LEVELS) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_wt_out, sizeof(float)*FOC_NUM_SENSORS*FOC_WT_LEVELS*(FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)) );

    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        HANDLE_ERROR( cudaMemcpy(&(dev_wt_out[idx*FOC_WT_LEVELS*(FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)]), data_wt_out[idx], (FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)*FOC_WT_LEVELS*sizeof(float), cudaMemcpyHostToDevice) );

    IdentifyModMax<<<(FOC_NUM_SENSORS*FOC_WT_LEVELS+512-1)/512, 512>>>(dev_modmax, dev_modmax_num, dev_data_wt_out);

    HANDLE_ERROR( cudaMemcpy(modmax, dev_modmax, (FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)*FOC_WT_LEVELS*FOC_NUM_SENSORS*sizeof(FOC_ModMax_t), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(data_modmax_num, dev_modmax_num, FOC_WT_LEVELS*FOC_NUM_SENSORS*sizeof(int), cudaMemcpyDeviceToHost) );

    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        for (int level = 0; level < FOC_WT_LEVELS; level++)
            std::copy(&(modmax[(idx*FOC_WT_LEVELS+level)*(FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)]), &(modmax[(idx+1)*(FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1)*FOC_WT_LEVELS+data_modmax_num[idx*FOC_WT_LEVELS+level]]), std::back_inserter(data_modmax[idx]));

    HANDLE_ERROR( cudaFree(dev_wt_out) );
    HANDLE_ERROR( cudaFree(dev_modmax_num) );
    HANDLE_ERROR( cudaFree(dev_modmax) );
    HANDLE_ERROR( cudaFreeHost(modmax) );

#else
/* Calculate using CPU */
    int i;
    FOC_ModMax_t local_max;
    for (int idx_s = 0; idx_s < FOC_NUM_SENSORS; idx_s++) {
        // Non-maximum suppression
        for (int idx_wt = 0; idx_wt < data_wt_idx.size(); idx_wt++) {
            // traverse every points to find the local maxima
            data_modmax_num[idx_s*FOC_WT_LEVELS+idx_wt] = 0;
            i = data_wt_idx.at(idx_wt) + 1; // +1 for comparison
            while (i < data_wt_idx.at(idx_wt)+FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1-1) // -1 for comparison
            {
                if (data_wt_out[idx_s][i] > data_wt_out[idx_s][i+1]) {
                    if (data_wt_out[idx_s][i] >= data_wt_out[idx_s][i-1]) {
                        // threshold
                        if (data_wt_out[idx_s][i] > 0.01) {
                            local_max.t = i - data_wt_idx.at(idx_wt);
                            local_max.value = data_wt_out[idx_s][i];
                            local_max.level = idx_wt;
                            data_modmax[idx_s].push_back(local_max);
                            data_modmax_num[idx_s*FOC_WT_LEVELS+idx_wt]++;
                        }
                    }
                }
                else {
                    i++;
                    while (i < data_wt_idx.at(idx_wt)+FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1-1 and data_wt_out[idx_s][i] <= data_wt_out[idx_s][i+1])
                        i++;
                    if (i < data_wt_idx.at(idx_wt)+FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1-1) {
                        // threshold
                        if (data_wt_out[idx_s][i] > 0.01) {
                            local_max.t = i - data_wt_idx.at(idx_wt);
                            local_max.value = data_wt_out[idx_s][i];
                            local_max.level = idx_wt;
                            data_modmax[idx_s].push_back(local_max);
                            data_modmax_num[idx_s*FOC_WT_LEVELS+idx_wt]++;
                        }
                    }
                }
                i=i+2;
            }
        } // end for idx_wt
    } // end for idx_s
#endif

    return true;
}

/* Propogate maxima lines
 */
__global__ void PropagateMaximaLines(FOC_ModMax_t *modmax, int *modmax_num, int *modmax_idx, FOC_ModMax_t *maxline, int *maxline_size)
{
    float distance;
    int temp_idx;

    int tid = threadIdx.x + blockIdx.x*blockDim.x;

#if 1
    if (tid < modmax_num[0]) { // number of maxima lines 
        maxline[tid*FOC_WT_LEVELS+0].t = modmax[modmax_idx[0]+tid].t;
        maxline[tid*FOC_WT_LEVELS+0].value = modmax[modmax_idx[0]+tid].value;
        maxline[tid*FOC_WT_LEVELS+0].level = modmax[modmax_idx[0]+tid].level;
        maxline_size[tid] = 1;
        for (int level = 1; level < FOC_WT_LEVELS; level++) { // for every levels
            if (modmax_num[level] > 0) {
                distance = FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1; // init distance to a very large number
                for (int j = 0; j < modmax_num[level]; j++) { // find its coarser scale modmax point
                    if (fabsf(modmax[modmax_idx[level]+j].t - maxline[tid*FOC_WT_LEVELS+level-1].t) < distance) {
                        distance = fabsf(modmax[modmax_idx[level]+j].t - maxline[tid*FOC_WT_LEVELS+level-1].t);
                        temp_idx = modmax_idx[level]+j;
                    }
                }
                if (distance > FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR/5.0)
                    break;
                else { // find maxima belong to this line
                    maxline[tid*FOC_WT_LEVELS+level].t = modmax[temp_idx].t;
                    maxline[tid*FOC_WT_LEVELS+level].value = modmax[temp_idx].value;
                    maxline[tid*FOC_WT_LEVELS+level].level = modmax[temp_idx].level;
                    maxline_size[tid]++;
                }
            }
            else // end propogation
                break;
        }
    }
#else
    if (tid < modmax_num[0]) { // tid is the index of maxima lines here
        maxline[tid*FOC_WT_LEVELS+0].t = modmax[modmax_idx[0]+tid].t;
        maxline[tid*FOC_WT_LEVELS+0].value = modmax[modmax_idx[0]+tid].value;
        maxline[tid*FOC_WT_LEVELS+0].level = modmax[modmax_idx[0]+tid].level;
        maxline_size[tid] = 1;
    }
    __syncthreads();

    for (int level = 1; level < FOC_WT_LEVELS; level++) { // for every levels
        if (modmax_num[level] > 0 and tid < modmax_num[level]) { // tid is the index of modmax in the level
            distance = FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1; // init distance to a very large number
            for (int j = 0; j < modmax_num[0]; j++) { // traverse every incomplete maxlines, j is the index of maxlines
                if (maxline[j*FOC_WT_LEVELS+maxline_size[j]-1].level == level-1) { // last point of this maxline
                    if (fabsf(modmax[modmax_idx[level]+tid].t - maxline[j*FOC_WT_LEVELS+maxline_size[j]-1].t) < distance) {
                        distance = fabsf(modmax[modmax_idx[level]+tid].t - maxline[j*FOC_WT_LEVELS+maxline_size[j]-1].t);
                        temp_idx = j; // index of maxline
                    }
                }
            }
        }
        __syncthreads();

        if (modmax_num[level] > 0 and tid < modmax_num[level]) { // tid is the index of modmax in the level
            if (distance < FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR/5.0) {
                // found maxline this maxima belongs to
                maxline[temp_idx*FOC_WT_LEVELS+level].t = modmax[modmax_idx[level]+tid].t;
                maxline[temp_idx*FOC_WT_LEVELS+level].value = modmax[modmax_idx[level]+tid].value;
                maxline[temp_idx*FOC_WT_LEVELS+level].level = modmax[modmax_idx[level]+tid].level;
                maxline_size[temp_idx]++;
            }
        }
        __syncthreads();
    }
#endif
}

bool foc_chain_maxline(std::vector<FOC_ModMax_t>* data_modmax, int* data_modmax_num, std::vector<FOC_ModMax_t>*** data_maxline)
{
    static unsigned int previous_num_maxline[FOC_NUM_SENSORS] = {0};

    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        if (data_modmax[i].size() <= 0 or data_modmax_num[i*FOC_WT_LEVELS] <= 0)
            return false;
    }

    // clean memory of data_maxline
    // traverse every vector and free the mem
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        if (data_maxline[idx] != NULL) {
            for (int i = 0; i < previous_num_maxline[idx]; i++)
                std::vector<FOC_ModMax_t>().swap(*(data_maxline[idx][i]));
            // free memory containing the pointers
            free(data_maxline[idx]);
            data_maxline[idx] = NULL;
        }
    }

    // allocate new memory for pointers of vectors
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        data_maxline[idx] = (std::vector<FOC_ModMax_t>**)malloc(data_modmax_num[idx*FOC_WT_LEVELS+0]*sizeof(std::vector<FOC_ModMax_t>*));
        for (int i = 0; i < data_modmax_num[idx*FOC_WT_LEVELS+0]; i++) {
            data_maxline[idx][i] = new std::vector<FOC_ModMax_t>;
            data_maxline[idx][i]->reserve(FOC_WT_LEVELS);
        }
        previous_num_maxline[idx] = data_modmax_num[idx*FOC_WT_LEVELS+0];
    }

    // calculate index for modulus maxima
    int data_modmax_idx[FOC_NUM_SENSORS][FOC_WT_LEVELS];
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        data_modmax_idx[idx][0] = 0;
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        for (int level = 1; level < FOC_WT_LEVELS; level++)
            data_modmax_idx[idx][level] = data_modmax_idx[idx][level-1] + data_modmax_num[idx*FOC_WT_LEVELS+level-1];
    
    // start propogate maxima lines from the most fine scale
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        for (int i = 0; i < data_modmax_num[idx*FOC_WT_LEVELS+0]; i++)
            data_maxline[idx][i]->push_back(data_modmax[idx].at(i));
#if 1
    /* GPU */
    FOC_ModMax_t* dev_modmax[FOC_NUM_SENSORS];
    FOC_ModMax_t* maxline[FOC_NUM_SENSORS];
    FOC_ModMax_t* dev_maxline[FOC_NUM_SENSORS];
    int*          maxline_size[FOC_NUM_SENSORS];
    int*          dev_maxline_size[FOC_NUM_SENSORS];
    int*          dev_modmax_num[FOC_NUM_SENSORS];
    int*          dev_modmax_idx[FOC_NUM_SENSORS];
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        HANDLE_ERROR( cudaMalloc((void**)&dev_modmax[idx], sizeof(FOC_ModMax_t)*data_modmax[idx].size()) );
        HANDLE_ERROR( cudaMemcpy(dev_modmax[idx], &(data_modmax[idx].at(0)), data_modmax[idx].size()*sizeof(FOC_ModMax_t), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaHostAlloc((void**)&maxline[idx], FOC_WT_LEVELS*data_modmax_num[idx*FOC_WT_LEVELS+0]*sizeof(FOC_ModMax_t), cudaHostAllocDefault) ); 
        HANDLE_ERROR( cudaMalloc((void**)&dev_maxline[idx], sizeof(FOC_ModMax_t)*FOC_WT_LEVELS*data_modmax_num[idx*FOC_WT_LEVELS+0]) );
        HANDLE_ERROR( cudaHostAlloc((void**)&maxline_size[idx], data_modmax_num[idx*FOC_WT_LEVELS+0]*sizeof(int), cudaHostAllocDefault) ); 
        HANDLE_ERROR( cudaMalloc((void**)&dev_maxline_size[idx], data_modmax_num[idx*FOC_WT_LEVELS+0]*sizeof(int)) );
        HANDLE_ERROR( cudaMalloc((void**)&dev_modmax_num[idx], FOC_WT_LEVELS*sizeof(int)) );
        HANDLE_ERROR( cudaMemcpy(dev_modmax_num[idx], &data_modmax_num[idx*FOC_WT_LEVELS+0], FOC_WT_LEVELS*sizeof(int), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMalloc((void**)&dev_modmax_idx[idx], FOC_WT_LEVELS*sizeof(int)) );
        HANDLE_ERROR( cudaMemcpy(dev_modmax_idx[idx], data_modmax_idx[idx], FOC_WT_LEVELS*sizeof(int), cudaMemcpyHostToDevice) );
    }

// Debug
//printf("num = [%d, %d, %d]\n", data_modmax_num[0*FOC_WT_LEVELS+0], data_modmax_num[1*FOC_WT_LEVELS+0], data_modmax_num[2*FOC_WT_LEVELS+0]);

    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        PropagateMaximaLines<<<(data_modmax_num[idx*FOC_WT_LEVELS+0]+128-1)/128, 128>>>(dev_modmax[idx], dev_modmax_num[idx], dev_modmax_idx[idx], dev_maxline[idx], dev_maxline_size[idx]);

    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        HANDLE_ERROR( cudaMemcpy(maxline[idx], dev_maxline[idx], FOC_WT_LEVELS*data_modmax_num[idx*FOC_WT_LEVELS+0]*sizeof(FOC_ModMax_t), cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(maxline_size[idx], dev_maxline_size[idx], data_modmax_num[idx*FOC_WT_LEVELS+0]*sizeof(int), cudaMemcpyDeviceToHost) );
        for (int j = 0; j < data_modmax_num[idx*FOC_WT_LEVELS+0]; j++)
            std::copy(&(maxline[idx][j*FOC_WT_LEVELS]), &(maxline[idx][j*FOC_WT_LEVELS+maxline_size[idx][j]]), std::back_inserter(*data_maxline[idx][j]));
    }

    for (int i = 0; i < FOC_NUM_SENSORS; i++) {
        HANDLE_ERROR( cudaFree(dev_modmax[i]) );
        HANDLE_ERROR( cudaFree(dev_maxline[i]) );
        HANDLE_ERROR( cudaFree(dev_maxline_size[i]) );
        HANDLE_ERROR( cudaFree(dev_modmax_num[i]) );
        HANDLE_ERROR( cudaFree(dev_modmax_idx[i]) );
        HANDLE_ERROR( cudaFreeHost(maxline[i]) );
        HANDLE_ERROR( cudaFreeHost(maxline_size[i]) );
    }

#else
    /* CPU */
    // propogate from fine scales to coarse scales
    float distance;
    int temp_idx;
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        for (int i = 0; i < data_modmax_num[idx*FOC_WT_LEVELS+0]; i++) { // for every maxima line
            for (int level = 1; level < FOC_WT_LEVELS; level++) { // for every levels
                if (data_modmax_num[idx*FOC_WT_LEVELS+level] > 0) {
                    distance = FOC_LEN_RECENT_INFO+FOC_LEN_WAVELET-1; // init distance to a very large number
                    for (int j = 0; j < data_modmax_num[idx*FOC_WT_LEVELS+level]; j++) { // find its coarser scale modmax point
                        if (std::abs(data_modmax[idx].at(data_modmax_idx[idx][level]+j).t - data_maxline[idx][i]->at(level-1).t) < distance) {
                            distance = std::abs(data_modmax[idx].at(data_modmax_idx[idx][level]+j).t - data_maxline[idx][i]->at(level-1).t);
                            temp_idx = data_modmax_idx[idx][level]+j;
                        }
                    }
                    if (distance > FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR)
                        break;
                    else // find maxima belong to this line
                        data_maxline[idx][i]->push_back(data_modmax[idx].at(temp_idx));

                }
                else // end propogation
                    break;
            }
        }
    }
#endif
    return true;
}
