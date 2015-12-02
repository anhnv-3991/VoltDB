/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/*
yabuta's after comment

this file arrange interface of scan kernel. 

presum calls a suit kernel per size.

part_presum calculate difference of discrete element of array.


 */


#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUTUPLE.h"
#include "GPUNIJ.h"
#include "GPUIJ.h"
#include "scan_common.h"

using namespace voltdb;

