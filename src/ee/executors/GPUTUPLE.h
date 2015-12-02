
#ifndef GPUTUPLE_H
#define GPUTUPLE_H

#include <GPUetc/common/GNValue.h>


namespace voltdb{

//1blockでのスレッド数の定義。
#define BLOCK_SIZE_X 512  //outer ,left
#define BLOCK_SIZE_Y 512  //inner ,right

#define PARTITION 64
#define RADIX 6
#define PART_C_NUM 16
#define SHARED_MAX PARTITION * PART_C_NUM

#define RIGHT_PER_TH 256

#define PART_STANDARD 1
#define JOIN_SHARED 256
#define MAX_GNVALUE 5

/*
typedef struct _TUPLE {
    int key;
    int val;
} TUPLE;
*/

typedef struct _RESULT {
    int lkey;
    int rkey;
} RESULT;

typedef struct _COLUMNDATA{
    GNValue gn;
    int num;
} COLUMNDATA __attribute__((aligned(32)));

typedef struct _INDEXDATA {
	GNValue gn[MAX_GNVALUE];
	int num;
} IndexData;

typedef IndexData PostData;

}

#endif
