/*
 * sample program to check NVIDIA GPU compute capability
 */

#include <stdio.h>
#include <cuda.h>
#include "drvapi_error_string.h"

/* CUDA error check macro */
#define MY_CUDA_CHECK(res, text)                                        \
  if ((res) != CUDA_SUCCESS) {                                          \
    fprintf(stderr, "%s failed: res = %d->%s\n", (text), (res), getCudaDrvErrorString((res))); \
    exit(1);                                                            \
  }

main()
{
  /* initialize CUDA */
  CUresult res;
  res = cuInit(0);
  MY_CUDA_CHECK(res, "cuInit()");

  /* check GPU is setted or not */
  int device_num;
  res = cuDeviceGetCount(&device_num);
  MY_CUDA_CHECK(res, "cuDeviceGetCount()");

  if (device_num == 0) {        // no GPU is detected
    fprintf(stderr, "no CUDA capable GPU is detected...\n");
    exit(1);
  }

  //printf("%d GPUs are detected\n", device_num);

  /* get device handle of GPU No.0 */
  CUdevice dev;
  res = cuDeviceGet(&dev, 0);
  MY_CUDA_CHECK(res, "cuDeviceGet()");

  /* search compute capability of GPU No.0 */
  int major=0, minor=0;
  res = cuDeviceComputeCapability(&major, &minor, dev);
  MY_CUDA_CHECK(res, "cuDeviceComputeCapability()");

  printf("%d%d\n", major, minor);
  //printf("actual compute capability is : %d%d\n", major, minor);

}
