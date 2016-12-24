#include <vector>
#include <string>
#include <cmath>
#include "foc/error_cuda.h"
#include "flying_odor_compass.h"

// cwt
float *dev_wavelets;

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
    if (!(wavelet_name == "gauss_d1") and !(wavelet_name == "gauss_d2"))
        return false;

    if (wvs == NULL)
        return false;
    if (len <= 0 or num_levels <= 0)
        return false;

    // sample wavelets
    float scale; // scale = (num_levels-level_idx/num_levels)*10
    float sum_h; // for normalization
    for (int i = 0; i < num_levels; i++) {
        scale = 1;
        //scale = (float)(num_levels-i)/(float)num_levels*49+1; // linear
        scale = (std::pow(2.0, (float)(num_levels-i)/(float)num_levels)-1)*49+1;
        for (int j = 0; j < len; j++) {
            if (wavelet_name == "gauss_d1")
                wvs[i*len+j] = std::sqrt(scale)*wavelet_gauss_d1_psi(scale*(float)(j-len/2)/(float)(FOC_MOX_DAQ_FREQ*FOC_MOX_INTERP_FACTOR), 0.4);
        }
        // normalize
        sum_h = 0;
        for (int j = 0; j < len; j++)
            sum_h += std::pow(wvs[i*len+j],2);
        for (int j = 0; j < len; j++)
            wvs[i*len+j] /= std::sqrt(sum_h);
    }

    return true;
}

/* Continuous Wavelet Transform
 */
void foc_cwt_init(float *data_wvs, std::vector<int>& data_wvs_idx, std::vector<float> data_wt_out[FOC_NUM_SENSORS][FOC_WT_LEVELS])
{
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        for (int j = 0; j < FOC_WT_LEVELS; j++)
            data_wt_out[i][j].clear();

    // index of data_wavelets, for the debugging purpose
    data_wvs_idx.reserve(FOC_WT_LEVELS);
    data_wvs_idx.clear();
    for (int i = 0; i < FOC_WT_LEVELS; i++)
        data_wvs_idx.push_back(i*FOC_LEN_WAVELET);

#if 0
    // storing cuda device properties, for the sheduling of parallel computing
    cudaDeviceProp prop;
    int count; // number of devices
    HANDLE_ERROR( cudaGetDeviceCount(&count) );
    for (int i = 0; i < count; i++) {// print out info of all graphic cards
        HANDLE_ERROR( cudaGetDeviceProperties(&prop, i) );
        printf("======== Card %d ========\n", i+1);
        printf("Graphic card name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total global memory: %ld MByte\n", prop.totalGlobalMem/1024/1024);
        printf("Total constant memoty: %ld kByte\n", prop.totalConstMem/1024);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
#endif

/* allocate memory for wavelet tranformation of signals */ 
    // allocate device memory for wavelets
    HANDLE_ERROR( cudaMalloc((void**)&dev_wavelets, FOC_WT_LEVELS*FOC_LEN_WAVELET*sizeof(float)) );

/* make wavelets */
    if (!sample_wavelets("gauss_d1", data_wvs, FOC_LEN_WAVELET, FOC_WT_LEVELS)) {
        printf("Error: make wavelets failed!\n");
    }
    HANDLE_ERROR( cudaMemcpy(dev_wavelets, data_wvs, FOC_WT_LEVELS*FOC_LEN_WAVELET*sizeof(float), cudaMemcpyHostToDevice) );
}

__global__ void RealConvWT(float *dev_wt_in, float *dev_wavelets, float *dev_wt_out, int len_signal)
{ 
    int len_result = len_signal-FOC_LEN_WAVELET+1;
    if (len_result < 1) return;

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int idx_sensor = tid/(FOC_WT_LEVELS*len_result);
    int idx_level = (tid%(FOC_WT_LEVELS*len_result))/len_result;
    int idx_offset = (tid%(FOC_WT_LEVELS*len_result))%len_result;

    if (tid < FOC_NUM_SENSORS*FOC_WT_LEVELS*len_result)
    {
        float sum = 0;
        for (int i = 0; i < FOC_LEN_WAVELET; i++)
            sum += dev_wt_in[idx_sensor*len_signal+idx_offset+i]*dev_wavelets[idx_level*FOC_LEN_WAVELET+(FOC_LEN_WAVELET-1-i)];
        dev_wt_out[tid] = sum;
    }
}

/* Continuous Wavelet Transform
 */
bool foc_cwt_update(std::vector<float> *signal, std::vector<float> data_wt_out[FOC_NUM_SENSORS][FOC_WT_LEVELS])
{
    static int index_in_signal = FOC_LEN_WAVELET;

    static float *wt_in;
    static float *dev_wt_in;
    static float *wt_out;
    static float *dev_wt_out;

    if (signal[0].size() < index_in_signal + FOC_MOX_INTERP_FACTOR) // signal[0..FOC_NUM_SENSORS-1] have the same length
        return false;

/* Wavele Transform */
    // CWT Phase 0: prepare for data
    int len_signal = signal[0].size() - index_in_signal -1 + FOC_LEN_WAVELET; 
    //  allocate a page-locked host memory & device memory containing signals
    HANDLE_ERROR( cudaHostAlloc((void**)&wt_in, FOC_NUM_SENSORS*len_signal*sizeof(float), cudaHostAllocDefault) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_wt_in, FOC_NUM_SENSORS*len_signal*sizeof(float)) );
    //  allocate a page-locked host memory & device memory containing all of the output wavelet levels
    HANDLE_ERROR( cudaHostAlloc((void**)&wt_out, FOC_NUM_SENSORS*FOC_WT_LEVELS*(len_signal-FOC_LEN_WAVELET+1)*sizeof(float), cudaHostAllocDefault) ); 
    HANDLE_ERROR( cudaMalloc((void**)&dev_wt_out, FOC_NUM_SENSORS*FOC_WT_LEVELS*(len_signal-FOC_LEN_WAVELET+1)*sizeof(float)) );
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        std::copy(signal[idx].end()-len_signal, signal[idx].end(), &(wt_in[idx*len_signal]));
    HANDLE_ERROR( cudaMemcpy(dev_wt_in, wt_in, FOC_NUM_SENSORS*len_signal*sizeof(float), cudaMemcpyHostToDevice) );
    // CWT Phase 1: conv
    RealConvWT<<<(FOC_NUM_SENSORS*FOC_WT_LEVELS*(len_signal-FOC_LEN_WAVELET+1)+128-1)/128, 128>>>(dev_wt_in, dev_wavelets, dev_wt_out, len_signal);
    // CWT Phase 2: result
    HANDLE_ERROR( cudaMemcpy(wt_out, dev_wt_out, FOC_NUM_SENSORS*FOC_WT_LEVELS*(len_signal-FOC_LEN_WAVELET+1)*sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        for (int j = 0; j < FOC_WT_LEVELS; j++)
            for (int k = 0; k < len_signal-FOC_LEN_WAVELET+1; k++)
                data_wt_out[i][j].push_back(wt_out[i*FOC_WT_LEVELS*(len_signal-FOC_LEN_WAVELET+1)+j*(len_signal-FOC_LEN_WAVELET+1)+k]);
    // CWT Phase 3: clean battlefield
    HANDLE_ERROR( cudaFree(dev_wt_in) );
    HANDLE_ERROR( cudaFree(dev_wt_out) );
    HANDLE_ERROR( cudaFreeHost(wt_in) );
    HANDLE_ERROR( cudaFreeHost(wt_out) );

    index_in_signal = signal[0].size();

    return true;
}


void foc_identify_modmax_init(std::vector<FOC_ModMax_t> data_modmax[FOC_NUM_SENSORS][FOC_WT_LEVELS][2])
{
    for (int i = 0; i < FOC_NUM_SENSORS; i++)
        for (int j = 0; j < FOC_WT_LEVELS; j++)
            for (int k = 0; k < 2; k++)
                data_modmax[i][j][k].clear();
}

bool foc_identify_modmax_update(std::vector<float> data_wt_out[FOC_NUM_SENSORS][FOC_WT_LEVELS], std::vector<FOC_ModMax_t> data_modmax[FOC_NUM_SENSORS][FOC_WT_LEVELS][2])
{
    static int index_in_signal = 2; // -2 in the following code

    if (data_wt_out[0][0].size() < index_in_signal + FOC_MOX_INTERP_FACTOR) // data_wt_out[0..FOC_NUM_SENSORS-1][0...FOC_WT_LEVELS-1] have the same length
        return false;

    int i;
    FOC_ModMax_t new_modmax;
    for (int idx_sensor = 0; idx_sensor < FOC_NUM_SENSORS; idx_sensor++) {
        for (int idx_level = 0; idx_level < FOC_WT_LEVELS; idx_level++) {
            // Non-maximum suppression
            // traverse every points to find the local maxima
            i = index_in_signal -1;
            while (i < data_wt_out[idx_sensor][idx_level].size() -1)
            {
                if (data_wt_out[idx_sensor][idx_level].at(i) > data_wt_out[idx_sensor][idx_level].at(i+1)) {
                    if (data_wt_out[idx_sensor][idx_level].at(i) >= data_wt_out[idx_sensor][idx_level].at(i-1)) {
                        if (data_wt_out[idx_sensor][idx_level].at(i) > 0) {
                            new_modmax.t = i;
                            new_modmax.value = data_wt_out[idx_sensor][idx_level].at(i);
                            new_modmax.level = idx_level;
                            data_modmax[idx_sensor][idx_level][0].push_back(new_modmax);
                        }
                    }
                }
                else {
                    i++;
                    while (i < data_wt_out[idx_sensor][idx_level].size()-1 and data_wt_out[idx_sensor][idx_level].at(i) <= data_wt_out[idx_sensor][idx_level].at(i+1))
                        i++;
                    if (i < data_wt_out[idx_sensor][idx_level].size()-1) {
                        if (data_wt_out[idx_sensor][idx_level].at(i) > 0) {
                            new_modmax.t = i;
                            new_modmax.value = data_wt_out[idx_sensor][idx_level].at(i);
                            new_modmax.level = idx_level;
                            data_modmax[idx_sensor][idx_level][0].push_back(new_modmax);
                        }
                    }
                }
                i = i+2;
            }

            // Non-minimum suppression
            i = index_in_signal -1;
            while (i < data_wt_out[idx_sensor][idx_level].size() -1)
            {
                if (data_wt_out[idx_sensor][idx_level].at(i) <= data_wt_out[idx_sensor][idx_level].at(i+1)) {
                    if (data_wt_out[idx_sensor][idx_level].at(i) < data_wt_out[idx_sensor][idx_level].at(i-1)) {
                        if (data_wt_out[idx_sensor][idx_level].at(i) < 0) {
                            new_modmax.t = i;
                            new_modmax.value = data_wt_out[idx_sensor][idx_level].at(i);
                            new_modmax.level = idx_level;
                            data_modmax[idx_sensor][idx_level][1].push_back(new_modmax);
                        }
                    }
                }
                else {
                    i++;
                    while (i < data_wt_out[idx_sensor][idx_level].size()-1 and data_wt_out[idx_sensor][idx_level].at(i) > data_wt_out[idx_sensor][idx_level].at(i+1))
                        i++;
                    if (i < data_wt_out[idx_sensor][idx_level].size()-1) {
                        if (data_wt_out[idx_sensor][idx_level].at(i) < 0) {
                            new_modmax.t = i;
                            new_modmax.value = data_wt_out[idx_sensor][idx_level].at(i);
                            new_modmax.level = idx_level;
                            data_modmax[idx_sensor][idx_level][1].push_back(new_modmax);
                        }
                    }
                }
                i = i+2;
            }
        }
    }

    index_in_signal = data_wt_out[0][0].size();

    return true;
}

void foc_chain_maxline_init(std::vector<FOC_Maxline_t> data_maxline[FOC_NUM_SENSORS][2])
{
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        for (int sign = 0; sign < 2; sign++)
        data_maxline[idx][sign].clear();
}

float decision_function_chain_maxline(float s1, float s2, float alpha, float n, float m, float wf_s1, float wf_s2)
{
    float delta = std::exp(-std::abs(n-m)*std::pow(s1,-alpha));
    float D = std::exp( -std::abs(std::log(std::abs(wf_s2)/std::abs(wf_s1))/std::log(s2/s1)-0.5)*std::pow(s1,alpha) );
    return delta*D;
}

bool foc_chain_maxline_update(std::vector<FOC_ModMax_t> data_modmax[FOC_NUM_SENSORS][FOC_WT_LEVELS][2], std::vector<FOC_Maxline_t> data_maxline[FOC_NUM_SENSORS][2], int size_of_wt_out)
{
    static int previous_t[FOC_NUM_SENSORS][2] = {0};

    if (data_modmax[0][0][0].size() == 0)
        return false;

    int t_modmax_bound = FOC_MOX_INTERP_FACTOR*10;

    FOC_Maxline_t new_maxline;
    float probability, temp_probability;
    int temp_idx_modmax;
    bool flag_should_grow_maxline = false;
    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++) {
        for (int sign = 0; sign < 2; sign++) {
            if (data_modmax[idx][0][sign].size() > 0) {
                for (int i = data_modmax[idx][0][sign].size()-1; i >= 0; i--) {
                    if (data_modmax[idx][0][sign].at(i).t > previous_t[idx][sign]) {
                        if (data_modmax[idx][0][sign].at(i).t + t_modmax_bound < size_of_wt_out) {// 1/FOC_MOX_DAQ_FREQ s
                            new_maxline.t[0] = data_modmax[idx][0][sign].at(i).t;
                            new_maxline.value[0] = data_modmax[idx][0][sign].at(i).value;
                            new_maxline.levels = 1;
                            for (int level = 1; level < FOC_WT_LEVELS; level++) {
                                if (data_modmax[idx][level][sign].size() > 0 and level == new_maxline.levels) {
                                    probability = 0;
                                    for (int j = data_modmax[idx][level][sign].size()-1; j >= 0; j--) {
                                        if (data_modmax[idx][level][sign].at(j).t > new_maxline.t[level-1] - t_modmax_bound) {
                                            if (data_modmax[idx][level][sign].at(j).t < new_maxline.t[level-1] + t_modmax_bound) {
                                                temp_probability = decision_function_chain_maxline(std::sqrt((float)level), std::sqrt((float)level+1), -0.3, data_modmax[idx][level][sign].at(j).t, new_maxline.t[level-1], new_maxline.value[level-1], data_modmax[idx][level][sign].at(j).value);
                                                if (temp_probability > probability) {
                                                    probability = temp_probability;
                                                    temp_idx_modmax = j;
                                                }
                                            }
                                        }
                                        else {
                                            if (probability > 0) { // found
                                                // eliminate wrong link in previous maxlines
                                                if (data_maxline[idx][sign].size() > 0) {
                                                    for (int pre_ml = (int)data_maxline[idx][sign].size()-1>=0?data_maxline[idx][sign].size()-1:0; pre_ml < data_maxline[idx][sign].size(); pre_ml++) {
                                                        if (level < data_maxline[idx][sign].at(pre_ml).levels and data_modmax[idx][level][sign].at(temp_idx_modmax).t == data_maxline[idx][sign].at(pre_ml).t[level] and data_modmax[idx][level][sign].at(temp_idx_modmax).value == data_maxline[idx][sign].at(pre_ml).value[level]) {
                                                            if (probability > decision_function_chain_maxline(std::sqrt((float)level), std::sqrt((float)level+1), -0.3, data_modmax[idx][level][sign].at(temp_idx_modmax).t, data_maxline[idx][sign].at(pre_ml).t[level-1], data_maxline[idx][sign].at(pre_ml).value[level-1], data_modmax[idx][level][sign].at(temp_idx_modmax).value)) {
                                                                data_maxline[idx][sign].at(pre_ml).levels = level;
                                                                flag_should_grow_maxline = true;
                                                            }
                                                            else
                                                                flag_should_grow_maxline = false;
                                                            break;
                                                        }
                                                        else
                                                            flag_should_grow_maxline = true;
                                                    }
                                                }
                                                else
                                                    flag_should_grow_maxline = true;
                                                if (flag_should_grow_maxline) {
                                                    new_maxline.t[level] = data_modmax[idx][level][sign].at(temp_idx_modmax).t;
                                                    new_maxline.value[level] = data_modmax[idx][level][sign].at(temp_idx_modmax).value;
                                                    new_maxline.levels++;
                                                    flag_should_grow_maxline = false;
                                                }
                                            }
                                            else {
                                                data_maxline[idx][sign].push_back(new_maxline);
                                                break;
                                            }
                                            if (new_maxline.levels == FOC_WT_LEVELS)
                                                data_maxline[idx][sign].push_back(new_maxline);
                                            break;
                                        }
                                    }
                                }
                                else {
                                    data_maxline[idx][sign].push_back(new_maxline);
                                    break;
                                }
                            }
                        }
                    }
                    else
                        break;
                }
            }
        }
    }

    for (int idx = 0; idx < FOC_NUM_SENSORS; idx++)
        for (int sign = 0; sign < 2; sign++)
            previous_t[idx][sign] = size_of_wt_out-t_modmax_bound-1;

    return true;
}
