#include <cufft.h>
#include <string>
#include <cmath>
#include "foc/error_cuda.h"
#include "flying_odor_compass.h"

// data pointers for GPU computation
static float *data_wt_in[FOC_NUM_SENSORS];
static float *dev_data_wt_in[FOC_NUM_SENSORS];
static float *dev_data_wt_out;

// L, length of convolution out
static int L;

// FFT handles
cufftHandle plan_signals, plan_results;

// FFT of wavelets
cufftComplex *fft_of_wavelets;

/* 1st order derivative of gaussian wavelet
   f'(x)=-x/(sigma^3*sqrt(2*pi))*e^(-x^2/(2*sigma^2))
 */
float wavelet_gauss_d1_psi(float x, float sigma)
{
    return -x*std::exp(-0.5*std::pow(x,2)/std::pow(sigma,2))/(std::pow(sigma,3)*std::sqrt(2*M_PI));
}

/* Make wavelets
    wvs             array containing wavelets, len(wvs)=len*num_levels
    wvs_conv        extended (insert zeros), len(wvs_conv)=len_conv*num_levels
    len             sample window of wavelets, M
    len_conv        L
    num_levels      number of levels
 */
bool sample_wavelets(std::string wavelet_name, float* wvs, int len, float *wvs_conv, int len_conv, int num_levels)
{
    if (len > len_conv)
        return false; 

    // sample wavelets
    float scale; // scale = (num_levels-level_idx/num_levels)*10
    float sum_h; // for normalization
    memset(wvs, 0, len*FOC_WT_LEVELS*sizeof(float));
    for (int i = 0; i < num_levels; i++) {
        scale = (float)(num_levels-i)/(float)num_levels*50.0;
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

    // extend wavelets from length M to length L
    memset(wvs_conv, 0, len_conv*FOC_WT_LEVELS*sizeof(float));
    for (int i = 0; i < num_levels; i++)
        std::copy(wvs+i*len, wvs+(i+1)*len, wvs_conv+i*len_conv);

    return true;
}

/* Continuous Wavelet Transform
 */
void foc_cwt_init(float **addr_data_wvs, std::vector<int>& data_wvs_idx, float **data_wt_out, std::vector<int>& data_wt_idx)
{
    // calculate L
    int N = FOC_LEN_RECENT_INFO + FOC_LEN_WAVELET - 1;
    L = pow(2, static_cast<int>(log2(static_cast<double>(N))+1));
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
    cufftComplex *fft_of_signals, *fft_of_results; // FFT intermediate
    HANDLE_ERROR( cudaMalloc((void**)&fft_of_signals, sizeof(cufftComplex)*L*FOC_WT_LEVELS) );
    HANDLE_ERROR( cudaMalloc((void**)&fft_of_results, sizeof(cufftComplex)*L*FOC_WT_LEVELS) );
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
    HANDLE_ERROR( cudaFree(fft_of_signals) );
    HANDLE_ERROR( cudaFree(fft_of_results) );

    return true;
}

bool foc_calculate_maximalines(std::vector<float>* data_maxline, float **data_wt_out, std::vector<int>& data_wt_idx)
{
    if (data_wt_idx.size() == 0)
        return false;

    // propogate from fine scales to coarse scales
    for (int idx_wt = 0; idx_wt < data_wt_idx.size(); idx_wt++) {

    }

    return true;
}
