/*
 * Initilize cuda in the main thread
 *
 * */

#include "method/foc/error_cuda.h"

extern void GSRAO_cuda_init(void)
{
    // storing cuda device properties, for the sheduling of parallel computing
    int count; // number of devices
    HANDLE_ERROR( cudaGetDeviceCount(&count) );
#if 0
    cudaDeviceProp prop; 
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
    if (count > 1) {// multiple graphic cards
        if(count == 2){
         HANDLE_ERROR( cudaSetDevice(1) );   //Set the second graphic to calculate
        }
        else{
            printf("Warning: Multiple graphic cards have been found on this machine. Please modify the function WakeInit in the file src/model/wake.cu to choose the most appropriate card.\n");
            exit(EXIT_FAILURE); // force the user to choose which card to use
        }
    }
    else if (count <= 0) {// no graphic card found
        printf("Error: No graphic cards have been found on this machine. Please run this program on the machine with NVIDIA graphic cards.\n");
        exit(EXIT_FAILURE);
    }
}
