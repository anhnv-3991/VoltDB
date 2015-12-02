
#ifndef GPUH
#define GPUH

#include <cuda.h>

using namespace voltdb;

class GPU{

public:
    GPU(){
        res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            printf("cuInit failed: res = %lu\n", (unsigned long)res);
        }
        res = cuDeviceGet(&dev, 0);
        if (res != CUDA_SUCCESS) {
            printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
        }
        res = cuCtxCreate(&ctx, 0, dev);
        if (res != CUDA_SUCCESS) {
            printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
        }
    }
   
    ~GPU(){
        res = cuCtxDestroy(ctx);
        if (res != CUDA_SUCCESS) {
            printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
        }
    }

private:

    CUdevice dev;
    CUcontext ctx;
    CUresult res;
};


#endif
