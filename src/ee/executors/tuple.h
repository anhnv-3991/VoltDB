//#define SZ_PAGE 40960
//#define NB_BUFR (SZ_PAGE * 2 / sizeof(TUPLE))
//#define NB_BUFS (SZ_PAGE * 16 / sizeof(TUPLE))
/*
1048576*1048576
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 262144

4194304*4194304    
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 4194304

16777216*16777216  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 16777216

33554432*33554432  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 33554432

67108864*67108864  
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 67108864

134217728*134217728
#define BLOCK_SIZE_X 1024
#define NB_BKT_ENT 67108864

*/
#define MATCH_RATE 0.01

#define JT_SIZE 120000000
#define SELECTIVITY 1000000000

#define PARTITION 64
#define RADIX 6
#define PART_C_NUM 16
#define SHARED_MAX PARTITION * PART_C_NUM

#define RIGHT_PER_TH 256

#define PART_STANDARD 1
#define JOIN_SHARED 256


int right,left;

typedef struct _TUPLE {
  int key;
  int val;
} TUPLE;

typedef struct _RESULT {
  int rkey;
  int rval;
  int lkey;
  int lval;
} RESULT;

typedef struct _BUCKET {

    int val;
    int adr;

} BUCKET;
