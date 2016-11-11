#ifndef CUDAHEADER_H
#define CUDAHEADER_H

#ifdef __CUDACC__
#define CUDAH __host__ __device__
#else
#define CUDAH
#endif
#define CUDAD __device__

#endif
