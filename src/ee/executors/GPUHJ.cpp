#include "GPUHJ.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <math.h>
#include "ghash.h"
#include "common/types.h"
#include <inttypes.h>

namespace voltdb {

const uint64_t GPUHJ::MAX_BUCKETS[] = {
	        3,
	        7,
	        13,
	        31,
	        61,
	        127,
	        251,
	        509,
	        1021,
	        2039,
	        4093,
	        8191,
	        16381,
	        32749,
	        65521,
	        131071,
	        262139,
	        524287,
	        1048573,
	        2097143,
	        4194301,
	        8388593,
	        16777213,
	        33554393,
	        67108859,
	        134217689,
	        268435399,
	        536870909,
	        1073741789,
	        2147483647,
	        4294967291,
	        8589934583
	};

GPUHJ::GPUHJ()
{
		outer_table_ = inner_table_ = NULL;
		outer_rows_ =  inner_rows_ = 0;
		outer_cols_ = inner_cols_ = 0;
		join_result_ = NULL;
		result_size_ = 0;
		end_size_ = 0;
		post_size_ = 0;
		initial_size_ = 0;
		skipNull_size_ = 0;
		prejoin_size_ = 0;
		where_size_ = 0;
		indices_size_ = 0;
		search_exp_size_ = NULL;
		search_exp_num_ = 0;
		indices_ = NULL;
		maxNumberOfBuckets_ = 0;

		search_exp_ = NULL;
		end_expression_ = NULL;
		post_expression_ = NULL;
		initial_expression_ = NULL;
		skipNullExpr_ = NULL;
		prejoin_expression_ = NULL;
		where_expression_ = NULL;
}

GPUHJ::GPUHJ(GNValue *outer_table,
				GNValue *inner_table,
				int outer_rows,
				int outer_cols,
				int inner_rows,
				int inner_cols,
				std::vector<TreeExpression> search_exp,
				std::vector<int> indices,
				TreeExpression end_expression,
				TreeExpression post_expression,
				TreeExpression initial_expression,
				TreeExpression skipNullExpr,
				TreeExpression prejoin_expression,
				TreeExpression where_expression,
				IndexLookupType lookup_type)
{
	/**** Table data *********/
	outer_table_ = outer_table;
	inner_table_ = inner_table;
	outer_rows_ = outer_rows;
	inner_rows_ = inner_rows;
	outer_cols_ = outer_cols;
	inner_cols_ = inner_cols;
	join_result_ = NULL;
	result_size_ = 0;
	end_size_ = end_expression.getSize();
	post_size_ = post_expression.getSize();
	initial_size_ = initial_expression.getSize();
	skipNull_size_ = skipNullExpr.getSize();
	prejoin_size_ = prejoin_expression.getSize();
	where_size_ = where_expression.getSize();
	search_exp_num_ = search_exp.size();
	indices_size_ = indices.size();
	lookup_type_ = lookup_type;

	maxNumberOfBuckets_ = MAX_BUCKETS[14];


	bool ret = true;
	int tmp_size = 0;

	search_exp_size_ = (int *)malloc(sizeof(int) * search_exp_num_);
	assert(search_exp_size_ != NULL);
	for (int i = 0; i < search_exp_num_; i++) {
		search_exp_size_[i] = search_exp[i].getSize();
		tmp_size += search_exp_size_[i];
	}

	search_exp_ = (GTreeNode *)malloc(sizeof(GTreeNode) * tmp_size);
	assert(search_exp_ != NULL);
	GTreeNode *exp_ptr = search_exp_;
	for (int i = 0; i < search_exp_num_; i++) {
		getTreeNodes2(exp_ptr, search_exp[i]);
		exp_ptr += search_exp_size_[i];
	}

	indices_ = (int *)malloc(sizeof(int) * indices_size_);
	assert(indices_ != NULL);
	for (int i = 0; i < indices_size_; i++) {
		indices_[i] = indices[i];
	}

	/**** Expression data ****/

	ret = getTreeNodes(&end_expression_, end_expression);
	assert(ret == true);

	ret = getTreeNodes(&post_expression_, post_expression);
	assert(ret == true);

	ret = getTreeNodes(&initial_expression_, initial_expression);
	assert(ret == true);

	ret = getTreeNodes(&skipNullExpr_, skipNullExpr);
	assert(ret == true);

	ret = getTreeNodes(&prejoin_expression_, prejoin_expression);
	assert(ret == true);

	ret = getTreeNodes(&where_expression_, where_expression);
	assert(ret == true);

	int size = 0;

	for (int i = 0; i < indices_size_; i++) {
		switch(inner_table_[indices_[i]].getValueType()) {
		case VALUE_TYPE_TINYINT: {
			size += sizeof(int8_t);
			break;
		}
		case VALUE_TYPE_SMALLINT: {
			size += sizeof(int16_t);
			break;
		}
		case VALUE_TYPE_INTEGER: {
			size += sizeof(int32_t);
			break;
		}
		case VALUE_TYPE_BIGINT: {
			size += sizeof(int64_t);
			break;
		}
		}
	}

	keySize_ = (size - 1)/8 + 1;
	printf("KEYSIZE = %d\n", keySize_);
}

bool GPUHJ::getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression)
{
	int tmp_size = tree_expression.getSize();
	if (tmp_size >= 1) {
		*expression = (GTreeNode *)malloc(sizeof(GTreeNode) * tmp_size);
		if (expression == NULL) {
			std::cout << "Error: malloc(expression) failed." << std::endl;
			return false;
		}
		tree_expression.getNodesArray(*expression);
	} else {
		*expression = NULL;
	}

	return true;
}

GPUHJ::~GPUHJ()
{
	freeArrays<RESULT>(join_result_);
	freeArrays<GTreeNode>(search_exp_);
	freeArrays<int>(search_exp_size_);
	freeArrays<int>(indices_);
	freeArrays<GTreeNode>(end_expression_);
	freeArrays<GTreeNode>(post_expression_);
	freeArrays<GTreeNode>(initial_expression_);
	freeArrays<GTreeNode>(skipNullExpr_);
	freeArrays<GTreeNode>(where_expression_);
}

template <typename T> void GPUHJ::freeArrays(T *expression)
{
	if (expression != NULL) {
		free(expression);
	}
}

void GPUHJ::getResult(RESULT *output) const
{
	memcpy(output, join_result_, sizeof(RESULT) * result_size_);
}

int GPUHJ::getResultSize() const
{
	return result_size_;
}

bool GPUHJ::getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression)
{
	if (tree_expression.getSize() >= 1)
		tree_expression.getNodesArray(expression);

	return true;
}

void GPUHJ::debug(void)
{
	std::cout << "Size of outer table = " << outer_rows_ << std::endl;
	if (outer_rows_ != 0) {
		std::cout << "Outer table" << std::endl;
		for (int i = 0; i < outer_rows_; i++) {
			for (int j = 0; j < MAX_GNVALUE; j++) {
				NValue tmp;
				setNValue(&tmp, outer_table_[i * outer_cols_ + j]);
				std::cout << tmp.debug().c_str() << std::endl;
			}
		}
	} else
		std::cout << "Empty outer table" << std::endl;

	std::cout << "Size of inner table =" << inner_rows_ << std::endl;
	if (inner_rows_ != 0) {
		for (int i = 0; i < inner_rows_; i++) {
			for (int j = 0; j < MAX_GNVALUE; j++) {
				NValue tmp;
				setNValue(&tmp, inner_table_[i * inner_cols_ + j]);
				std::cout << tmp.debug().c_str() << std::endl;
			}
		}
	} else
		std::cout << "Empty inner table" << std::endl;

	std::cout << "Size of end_expression = " << end_size_ << std::endl;
	if (end_size_ != 0) {
		std::cout << "Content of end_expression" << std::endl;
		debugGTrees(end_expression_, end_size_);
	} else
		std::cout << "Empty end expression" << std::endl;

	std::cout << "Size of post_expression = " << post_size_ << std::endl;
	if (post_size_ != 0) {
		std::cout << "Content of post_expression" << std::endl;
		debugGTrees(post_expression_, post_size_);
	} else
		std::cout << "Empty post expression" << std::endl;

	std::cout << "Size of initial_expression = " << initial_size_ << std::endl;
	if (initial_size_ != 0) {
		std::cout << "Content of initial_expression" << std::endl;
		debugGTrees(initial_expression_, initial_size_);
	} else
		std::cout << "Empty initial expression" << std::endl;

	std::cout << "Size of skip null expression = " << skipNull_size_ << std::endl;
	if (skipNull_size_ != 0) {
		std::cout << "Content of skip null_expression" << std::endl;
		debugGTrees(skipNullExpr_, skipNull_size_);
	} else
		std::cout << "Empty skip null expression" << std::endl;

	std::cout << "Size of prejoin_expression = " << prejoin_size_ << std::endl;
	if (prejoin_size_ != 0) {
		std::cout << "Content of prejoin_expression" << std::endl;
		debugGTrees(prejoin_expression_, prejoin_size_);
	} else
		std::cout << "Empty prejoin expression " << std::endl;

	std::cout << "Size of where expression = " << where_size_ << std::endl;
	if (where_size_ != 0) {
		std::cout << "Content of where_expression" << std::endl;
		debugGTrees(where_expression_, where_size_);
	} else
		std::cout << "Empty where expression" << std::endl;

	std::cout << "Size of search_exp_ array = " << search_exp_num_ << std::endl;
	int search_exp_ptr = 0;
	if (search_exp_num_ != 0) {
		std::cout << "Content of search_exp" << std::endl;
		for (int i = 0; i < search_exp_num_; i++) {
			std::cout << "search_exp[" << i << std::endl;
			debugGTrees(search_exp_ + search_exp_ptr, search_exp_size_[i]);
			search_exp_ptr += search_exp_size_[i];
		}
	} else
		std::cout << "Empty search keys array" << std::endl;

	std::cout << "Size of innner_indices = " << indices_size_ << std::endl;
	if (indices_size_ != 0) {
		std::cout << "Content of inner indices" << std::endl;
		for (int i = 0; i < indices_size_; i++) {
			std::cout << "indices[" << i << "] = " << indices_[i] << std::endl;
		}
	} else
		std::cout << "Empty indices array" << std::endl;
}

void GPUHJ::setNValue(NValue *nvalue, GNValue &gnvalue)
{
	double tmp = gnvalue.getMdata();
	char gtmp[16];
	memcpy(gtmp, &tmp, sizeof(double));
	nvalue->setMdataFromGPU(gtmp);
//	nvalue->setSourceInlinedFromGPU(gnvalue.getSourceInlined());
	nvalue->setValueTypeFromGPU(gnvalue.getValueType());
}

void GPUHJ::debugGTrees(const GTreeNode *expression, int size)
{
	std::cout << "DEBUGGING INFORMATION..." << std::endl;
	for (int index = 0; index < size; index++) {
		switch (expression[index].type) {
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				std::cout << "[" << index << "] CONJUNCTION AND" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				std::cout << "[" << index << "] CONJUNCTION OR" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				std::cout << "[" << index << "] COMPARE EQUAL" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				std::cout << "[" << index << "] COMPARE NOTEQUAL" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				std::cout << "[" << index << "] COMPARE LESS THAN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				std::cout << "[" << index << "] COMPARE GREATER THAN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				std::cout << "[" << index << "] COMPARE LESS THAN OR EQUAL TO" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				std::cout << "[" << index << "] COMPARE GREATER THAN OR EQUAL TO" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LIKE: {
				std::cout << "[" << index << "] COMPARE LIKE" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_IN: {
				std::cout << "[" << index << "] COMPARE IN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				std::cout << "[" << index << "] TUPLE(";
				std::cout << expression[index].column_idx << "," << expression[index].tuple_idx;
				std::cout << ")" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT: {
				NValue tmp;
				GNValue tmp_gnvalue = expression[index].value;

				setNValue(&tmp, tmp_gnvalue);
				std::cout << "[" << index << "] VALUE TUPLE = " << tmp.debug().c_str()  << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_NULL:
			case EXPRESSION_TYPE_INVALID:
			default: {
				std::cout << "NULL value" << std::endl;
				break;
			}
		}
	}
}

unsigned long GPUHJ::timeDiff(struct timeval start, struct timeval end)
{
	return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

void GPUHJ::keyGenerateTest(GNValue *tuple, int *keyIndices, int indexNum, uint64_t *packedKey)
{
	int keyOffset = 0;
	int intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
	GNValue tmp_val;

	if (keyIndices != NULL) {
		for (int i = 0; i < indexNum; i++) {
			tmp_val = tuple[keyIndices[i]];

			switch (tmp_val.getValueType()) {
				case VALUE_TYPE_TINYINT: {
					int8_t value = static_cast<int8_t>(tmp_val.getValue());
					uint8_t keyValue = static_cast<uint8_t>(value + INT8_MAX + 1);
					for (int j = sizeof(uint8_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}
					break;
				}
				case VALUE_TYPE_SMALLINT: {
					int16_t value = static_cast<int16_t>(tmp_val.getValue());
					uint16_t keyValue = static_cast<uint16_t>(value + INT16_MAX + 1);
					for (int j = sizeof(uint16_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_INTEGER: {
					int32_t value = static_cast<int32_t>(tmp_val.getValue());
					uint32_t keyValue = static_cast<uint32_t>(value + INT32_MAX + 1);
					for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_BIGINT: {
					int64_t value = tmp_val.getValue();
					uint64_t keyValue = static_cast<uint64_t>(value + INT64_MAX + 1);
					for (int j = sizeof(uint64_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				default: {
					printf("Error: no match type. Type = %d\n", tmp_val.getValueType());
				}
			}

			printf("middle val = %" PRIu64 "\n", packedKey[0]);

		}
	} else {
		for (int i = 0; i < indexNum; i++) {
			tmp_val = tuple[i];

			switch (tmp_val.getValueType()) {
				case VALUE_TYPE_TINYINT: {
					int8_t value = static_cast<int8_t>(tmp_val.getValue());
					uint8_t keyValue = static_cast<uint8_t>(value + INT8_MAX + 1);
					for (int j = sizeof(uint8_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}
					break;
				}
				case VALUE_TYPE_SMALLINT: {
					int16_t value = static_cast<int16_t>(tmp_val.getValue());
					uint16_t keyValue = static_cast<uint16_t>(value + INT16_MAX + 1);
					for (int j = sizeof(uint16_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_INTEGER: {
					int32_t value = static_cast<int32_t>(tmp_val.getValue());
					uint32_t keyValue = static_cast<uint32_t>(value + INT32_MAX + 1);
					for (int j = sizeof(uint32_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_BIGINT: {
					int64_t value = tmp_val.getValue();
					uint64_t keyValue = static_cast<uint64_t>(value + INT64_MAX + 1);
					for (int j = sizeof(uint64_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				default:
					printf("Error: cannot detect type at index %d\n", i);
			}

		}
	}
}

bool GPUHJ::join()
{
	GNValue *outer_dev, *inner_dev, *stack;
	int outer_partition, inner_partition;
	GTreeNode *initial_dev, *end_dev, *post_dev, *where_dev, *search_exp_dev;
	int *indices_dev, *search_exp_size;
	ulong *indexCount, jr_size;
	RESULT *jresult_dev;


	if (initial_size_ > 0) {
		checkCudaErrors(cudaMalloc(&initial_dev, sizeof(GTreeNode) * initial_size_));
		checkCudaErrors(cudaMemcpy(initial_dev, initial_expression_, sizeof(GTreeNode) * initial_size_, cudaMemcpyHostToDevice));
	}

	if (end_size_ > 0) {
		checkCudaErrors(cudaMalloc(&end_dev, sizeof(GTreeNode) * end_size_));
		checkCudaErrors(cudaMemcpy(end_dev, end_expression_, sizeof(GTreeNode) * end_size_, cudaMemcpyHostToDevice));
	}

	if (post_size_ > 0) {
		checkCudaErrors(cudaMalloc(&post_dev, sizeof(GTreeNode) * post_size_));
		checkCudaErrors(cudaMemcpy(post_dev, post_expression_, sizeof(GTreeNode) * post_size_, cudaMemcpyHostToDevice));
	}

	if (where_size_ > 0) {
		checkCudaErrors(cudaMalloc(&where_dev, sizeof(GTreeNode) * where_size_));
		checkCudaErrors(cudaMemcpy(where_dev, where_expression_, sizeof(GTreeNode) * where_size_, cudaMemcpyHostToDevice));
	}

	int tmp_size = 0;

	for (int i = 0; i < search_exp_num_; i++) {
		tmp_size += search_exp_size_[i];
	}

	checkCudaErrors(cudaMalloc(&search_exp_dev, sizeof(GTreeNode) * tmp_size));
	checkCudaErrors(cudaMalloc(&search_exp_size, sizeof(int) * search_exp_num_));

	checkCudaErrors(cudaMemcpy(search_exp_dev, search_exp_, sizeof(GTreeNode) * tmp_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(search_exp_size, search_exp_size_, sizeof(int) * search_exp_num_, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&indices_dev, sizeof(int) * indices_size_));
	checkCudaErrors(cudaMemcpy(indices_dev, indices_, sizeof(int) * indices_size_, cudaMemcpyHostToDevice));


	int count_test = 0;
	GNValue tmp1 = inner_table_[0], tmp2 = inner_table_[1];
	uint64_t tmp0 = tmp1.getValue() * (int)pow(2,32) + tmp1.getValue();
	uint64_t tmp3, tmp4;
	for (int i = 0; i < inner_rows_; i++) {
		tmp3 = static_cast<uint64_t>(inner_table_[i * inner_cols_].getValue() + INT64_MAX + 1);
		tmp4 = static_cast<uint64_t>(inner_table_[i * inner_cols_ + 1].getValue() + INT64_MAX + 1);
		if (tmp3 * (int)pow(2,32) + tmp4 == tmp0)
			count_test++;
	}

	printf("Count test new is %d\n", count_test);

	/******** Hash the inner table *******/
	GHashNode innerHash;
	ulong *hashCount, *tmpHashCount;
	uint64_t *innerKey;
	ulong sum;
	uint64_t *innerTest;
	ulong *testCount;


	printf("Max number of buckets is %d\n", maxNumberOfBuckets_);
	int block_x = (inner_rows_ < BLOCK_SIZE_X) ? inner_rows_ : BLOCK_SIZE_X;
	int grid_x = (inner_rows_ % block_x == 0) ? (inner_rows_ / block_x) : (inner_rows_ / block_x + 1);
	innerHash.keySize = keySize_;
	innerHash.size = inner_rows_;
	innerHash.bucketNum = maxNumberOfBuckets_;

	checkCudaErrors(cudaMalloc(&inner_dev, sizeof(GNValue) * inner_rows_ * inner_cols_));
	checkCudaErrors(cudaMemcpy(inner_dev, inner_table_, sizeof(GNValue) * inner_rows_ * inner_cols_, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(&innerKey, sizeof(uint64_t) * inner_rows_ * keySize_));
	checkCudaErrors(cudaMalloc(&hashCount, sizeof(ulong) * (block_x * maxNumberOfBuckets_ + 1)));

	struct timeval innerPackStart, innerPackEnd, innerHashCountStart, innerHashCountEnd, innerHashStart, innerHashEnd, innerPrefixStart, innerPrefixEnd;
	std::vector<unsigned long> innerPack, innerHashCount, innerHasher, innerPrefix;

	gettimeofday(&innerPackStart, NULL);
	packKeyWrapper(block_x, 1, 1, 1, inner_dev, inner_rows_, inner_cols_, indices_dev, indices_size_, innerKey, keySize_);

	gettimeofday(&innerPackEnd, NULL);
	innerPack.push_back(timeDiff(innerPackStart, innerPackEnd));

	gettimeofday(&innerHashCountStart, NULL);
	ghashCountWrapper(block_x, 1, 1, 1, innerKey, inner_rows_, keySize_, hashCount, maxNumberOfBuckets_);


	gettimeofday(&innerHashCountEnd, NULL);
	innerHashCount.push_back(timeDiff(innerHashCountStart, innerHashCountEnd));


	gettimeofday(&innerPrefixStart, NULL);
	hprefixSumWrapper(hashCount, block_x * maxNumberOfBuckets_ + 1, &sum);

	gettimeofday(&innerPrefixEnd, NULL);
	innerPrefix.push_back(timeDiff(innerPrefixStart, innerPrefixEnd));

	checkCudaErrors(cudaMalloc(&(innerHash.hashedIdx), sizeof(uint64_t) * inner_rows_));
	checkCudaErrors(cudaMalloc(&(innerHash.hashedKey), sizeof(uint64_t) * inner_rows_ * keySize_));
	checkCudaErrors(cudaMalloc(&(innerHash.bucketLocation), sizeof(uint64_t) * (maxNumberOfBuckets_ + 1)));
	innerTest = (uint64_t *)malloc(sizeof(uint64_t) * inner_rows_ * keySize_);
	checkCudaErrors(cudaMemcpy(innerTest, innerKey, sizeof(uint64_t) * inner_rows_ * keySize_, cudaMemcpyDeviceToHost));

	gettimeofday(&innerHashStart, NULL);
	ghashWrapper(block_x, 1, 1, 1, innerKey, hashCount, innerHash);
	gettimeofday(&innerHashEnd, NULL);
	innerHasher.push_back(timeDiff(innerHashStart, innerHashEnd));
//
//	uint64_t *innerTestIdx;
//	uint64_t *innerTestKey;
//	innerTestIdx = (uint64_t *)malloc(sizeof(uint64_t) * inner_rows_);
//	innerTestKey = (uint64_t *)malloc(sizeof(uint64_t) * inner_rows_ * keySize_);
//
//	cudaMemcpy(innerTestKey, innerKey, sizeof(uint64_t) * inner_rows_ * keySize_, cudaMemcpyDeviceToHost);
//	printf("Before hash *************\n");
//	for (int i = 0; i < inner_rows_; i++) {
//		printf("Keyval = %" PRIu64 " at ", innerTestKey[i * keySize_]);
//		for (int j = 0; j < inner_cols_; j++) {
//			printf("col %d = %d", j, (int)(inner_table_[i * inner_cols_ + j].getValue()));
//		}
//		printf("\n");
//	}
//	cudaMemcpy(innerTestIdx, innerHash.hashedIdx, sizeof(uint64_t) * inner_rows_, cudaMemcpyDeviceToHost);
//	cudaMemcpy(innerTestKey, innerHash.hashedKey, sizeof(uint64_t) * inner_rows_ * keySize_, cudaMemcpyDeviceToHost);
//	printf("After hash ***************\n");
//	for (int i = 0; i < inner_rows_; i++) {
//		printf("Keyval = %" PRIu64 " at ", innerTestKey[i * keySize_]);
//		for (int j = 0; j < inner_cols_; j++) {
//			printf("col %d = %d", j, (int)(inner_table_[innerTestIdx[i] * inner_cols_ + j].getValue()));
//		}
//		printf("\n");
//	}

	checkCudaErrors(cudaFree(inner_dev));
	checkCudaErrors(cudaFree(hashCount));
	checkCudaErrors(cudaFree(innerKey));


	/******* Hash the outer table *******/
	uint64_t *outerKey;
	GHashNode outerHash;



	block_x = (outer_rows_ < BLOCK_SIZE_X) ? outer_rows_ : BLOCK_SIZE_X;
	outerHash.keySize = keySize_;
	outerHash.size = outer_rows_;
	outerHash.bucketNum = maxNumberOfBuckets_;

	checkCudaErrors(cudaMalloc(&outer_dev, sizeof(GNValue) * outer_rows_ * outer_cols_));
	checkCudaErrors(cudaMemcpy(outer_dev, outer_table_, sizeof(GNValue) * outer_rows_ * outer_cols_, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(&outerKey, sizeof(uint64_t) * outer_rows_ * keySize_));
	checkCudaErrors(cudaMalloc(&hashCount, sizeof(ulong) * (block_x * maxNumberOfBuckets_ + 1)));
	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * MAX_STACK_SIZE * block_x));

	struct timeval outerPackStart, outerPackEnd, outerHashCountStart, outerHashCountEnd, outerPrefixStart, outerPrefixEnd, outerHashStart, outerHashEnd;
	std::vector<unsigned long> outerPack, outerHashCount, outerPrefix, outerHasher;

	gettimeofday(&outerPackStart, NULL);
	packSearchKeyWrapper(block_x, 1, 1, 1, outer_dev, outer_rows_, outer_cols_,
							outerKey, search_exp_dev, search_exp_size, search_exp_num_,
							keySize_, stack);
	gettimeofday(&outerPackEnd, NULL);
	printf("PASSED outer pack key\n");
	outerPack.push_back(timeDiff(outerPackStart, outerPackEnd));

	gettimeofday(&outerHashCountStart, NULL);
	ghashCountWrapper(block_x, 1, 1, 1, outerKey, outer_rows_, keySize_, hashCount, maxNumberOfBuckets_);
	gettimeofday(&outerHashCountEnd, NULL);
	outerHashCount.push_back(timeDiff(outerHashCountStart, outerHashCountEnd));

	gettimeofday(&outerPrefixStart, NULL);
	hprefixSumWrapper(hashCount, block_x * maxNumberOfBuckets_ + 1, &sum);
	gettimeofday(&outerPrefixEnd, NULL);
	outerPrefix.push_back(timeDiff(outerPrefixStart, outerPrefixEnd));


	checkCudaErrors(cudaMalloc(&(outerHash.hashedIdx), sizeof(uint64_t) * outer_rows_));
	checkCudaErrors(cudaMalloc(&(outerHash.hashedKey), sizeof(uint64_t) * outer_rows_ * keySize_));
	checkCudaErrors(cudaMalloc(&(outerHash.bucketLocation), sizeof(uint64_t) * (maxNumberOfBuckets_ + 1)));

	gettimeofday(&outerHashStart, NULL);
	ghashWrapper(block_x, 1, 1, 1, outerKey, hashCount, outerHash);
	gettimeofday(&outerHashEnd, NULL);
	outerHasher.push_back(timeDiff(outerHashStart, outerHashEnd));
//
//	uint64_t *outerTestIdx;
//	uint64_t *outerTestKey;
//	uint64_t *outerBucketLoc;
//	outerTestIdx = (uint64_t *)malloc(sizeof(uint64_t) * outer_rows_);
//	outerTestKey = (uint64_t *)malloc(sizeof(uint64_t) * outer_rows_ * keySize_);
//	outerBucketLoc = (uint64_t *)malloc(sizeof(uint64_t) * (maxNumberOfBuckets_ + 1));
//
//	cudaMemcpy(outerTestKey, outerKey, sizeof(uint64_t) * outer_rows_ * keySize_, cudaMemcpyDeviceToHost);
//	printf("Before hash outer *************\n");
//	for (int i = 0; i < outer_rows_; i++) {
//		printf("Keyval = %" PRIu64 " at ", outerTestKey[i * keySize_]);
//		for (int j = 0; j < outer_cols_; j++) {
//			printf("col %d = %d", j, (int)(outer_table_[i * outer_cols_ + j].getValue()));
//		}
//		printf("\n");
//	}
//	cudaMemcpy(outerTestIdx, outerHash.hashedIdx, sizeof(uint64_t) * outer_rows_, cudaMemcpyDeviceToHost);
//	cudaMemcpy(outerTestKey, outerHash.hashedKey, sizeof(uint64_t) * outer_rows_ * keySize_, cudaMemcpyDeviceToHost);
//	printf("After hash outer ***************\n");
//	for (int i = 0; i < outer_rows_; i++) {
//		printf("Keyval = %" PRIu64 " at ", outerTestKey[i * keySize_]);
//		for (int j = 0; j < inner_cols_; j++) {
//			printf("col %d = %d", j, (int)(outer_table_[outerTestIdx[i] * outer_cols_ + j].getValue()));
//		}
//		printf("\n");
//	}
//
//	cudaMemcpy(outerBucketLoc, outerHash.bucketLocation, sizeof(uint64_t) * (maxNumberOfBuckets_ + 1), cudaMemcpyDeviceToHost);
//	printf("Outer bucket location *******\n");
//	for (int i = 0; i < maxNumberOfBuckets_ + 1; i++) {
//		printf("Bucket location %d is %d\n", i, (int)outerBucketLoc[i]);
//	}

	checkCudaErrors(cudaFree(stack));
	checkCudaErrors(cudaFree(outer_dev));
	checkCudaErrors(cudaFree(outerKey));

	/**** Allocate table memory for join ****/
	checkCudaErrors(cudaMalloc(&outer_dev, sizeof(GNValue) * outer_rows_ * outer_cols_));
	checkCudaErrors(cudaMalloc(&inner_dev, sizeof(GNValue) * inner_rows_ * inner_cols_));

	printf("END OF ALLOCATING NEW buffer\n");
	checkCudaErrors(cudaMemcpy(outer_dev, outer_table_, sizeof(GNValue) * outer_rows_ * outer_cols_, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(inner_dev, inner_table_, sizeof(GNValue) * inner_rows_ * inner_cols_, cudaMemcpyHostToDevice));

	int partitionSize = (outer_rows_ < DEFAULT_PART_SIZE_) ? outer_rows_ : DEFAULT_PART_SIZE_;
	int sizeOfBuckets = (outer_rows_ - 1) / maxNumberOfBuckets_ + 1;
	printf("Size of Buckets = %d\n", sizeOfBuckets);
	double tmp = (partitionSize - 1)/sizeOfBuckets + 1;
	int bucketStride = (int)pow(2, (int)(log2(tmp)));

	printf("bucketStride = %d\n", bucketStride);
	/* Calculate blockDim and gridDim. Each block takes responsibility of one
	 * bucket.
	 *
	 * The blockDim hence is set equal to the number of tuples in each
	 * bucket. When the number of tuple in each bucket exceed the maximum size
	 * of block, each thread iterate through tuples to complete the join.
	 *
	 * Since GPU memory capacity is limited, join all tuples simultaneously is
	 * sometimes becomes impossible. We can only compare several buckets at
	 * the same time, and move to the next buckets after completing. The list
	 * of buckets are divided to chunks, which contains several buckets. Each
	 * time we compare only one chunks. Size of chunk is set equal to power of
	 * 2.
	 *
	 * The gridDim is set equal to the number of buckets in each chunk.
	 */
	block_x = (sizeOfBuckets <= BLOCK_SIZE_X) ? sizeOfBuckets : BLOCK_SIZE_X;
	grid_x = (bucketStride < GRID_SIZE_X) ? bucketStride : GRID_SIZE_X;
	int grid_y = (bucketStride < GRID_SIZE_X) ? 1 : bucketStride/GRID_SIZE_X;
	partitionSize = grid_x * grid_y * block_x;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * MAX_STACK_SIZE * partitionSize));
	checkCudaErrors(cudaMalloc(&indexCount, sizeof(ulong) * (partitionSize + 1)));

	struct timeval indexCountStart, indexCountEnd, prefixStart, prefixEnd, joinStart, joinEnd;
	std::vector<unsigned long> indexHCount, prefixSum, joinTime;
	int realBucketStride;
	printf("PartitionSize = %d\n", partitionSize);
	/* Iterate over chunks of buckets */
	for (int bucketIdx = 0; bucketIdx < maxNumberOfBuckets_; bucketIdx += bucketStride) {
		realBucketStride = (bucketIdx + bucketStride < maxNumberOfBuckets_) ? bucketStride : (maxNumberOfBuckets_ - bucketIdx);
		printf("Real bucket stride = %d\n", realBucketStride);
		gettimeofday(&indexCountStart, NULL);
		indexCountWrapper(block_x, 1, grid_x, grid_y,
							outerHash, innerHash,
							bucketIdx, bucketIdx + realBucketStride,
							indexCount, partitionSize);
		gettimeofday(&indexCountEnd, NULL);
		indexHCount.push_back(timeDiff(indexCountStart, indexCountEnd));

		gettimeofday(&prefixStart, NULL);
		hprefixSumWrapper(indexCount, partitionSize + 1, &jr_size);
		printf("JR SIZE = %lu current size = %d\n", jr_size, result_size_);
		gettimeofday(&prefixEnd, NULL);
		prefixSum.push_back(timeDiff(prefixStart, prefixEnd));

		if (jr_size < 0) {
			printf("Scanning failed\n");
			return false;
		}

		if (jr_size == 0) {
			continue;
		}

		checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));

		gettimeofday(&joinStart, NULL);
		hashJoinWrapper(block_x, 1, grid_x, 1,
							outer_dev, inner_dev,
							outer_cols_, inner_cols_,
							end_dev, end_size_,
							post_dev, post_size_,
							outerHash, innerHash,
							bucketIdx, bucketIdx + realBucketStride,
							indexCount, partitionSize,
							stack, jresult_dev);
		gettimeofday(&joinEnd, NULL);
		joinTime.push_back(timeDiff(joinStart, joinEnd));

		join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size) * sizeof(RESULT));

		checkCudaErrors(cudaMemcpy(join_result_ + result_size_, jresult_dev, jr_size * sizeof(RESULT), cudaMemcpyDeviceToHost));
//		for (int i = 0; i < result_size_; i++)  {
//
//		}
		checkCudaErrors(cudaFree(jresult_dev));
		result_size_ += jr_size;
//		for (int i = 0; i < result_size_; i++)  {
//			printf("result at index %d is left = %d, right = %d\n", i, join_result_[i].lkey, join_result_[i].rkey);
//		}
		jr_size = 0;
	}
	unsigned long innerPackFinal, innerHashCountFinal, innerPrefixFinal, innerHashFinal;
	unsigned long outerPackFinal, outerHashCountFinal, outerPrefixFinal, outerHashFinal;
	unsigned long indexCountFinal, prefixSumFinal, joinFinal;

	innerPackFinal = 0;
	for (int i = 0; i < innerPack.size(); i++) {
		innerPackFinal += innerPack[i];
	}

	innerHashCountFinal = 0;
	for (int i = 0; i < innerHashCount.size(); i++) {
		innerHashCountFinal += innerHashCount[i];
	}

	innerPrefixFinal = 0;
	for (int i = 0; i < innerPrefix.size(); i++) {
		innerPrefixFinal += innerPrefix[i];
	}

	innerHashFinal = 0;
	for (int i = 0; i < innerHasher.size(); i++) {
		innerHashFinal += innerHasher[i];
	}

	outerPackFinal = 0;
	for (int i = 0; i < outerPack.size(); i++) {
		outerPackFinal += outerPack[i];
	}

	outerHashCountFinal = 0;
	for (int i = 0; i < outerHashCount.size(); i++) {
		outerHashCountFinal += outerHashCount[i];
	}

	outerPrefixFinal = 0;
	for (int i = 0; i < outerPrefix.size(); i++) {
		outerPrefixFinal += outerPrefix[i];
	}

	outerHashFinal = 0;
	for (int i = 0; i < outerHasher.size(); i++) {
		outerHashFinal += outerHasher[i];
	}

	indexCountFinal = 0;
	for (int i = 0; i < indexHCount.size(); i++) {
		indexCountFinal += indexHCount[i];
	}

	prefixSumFinal = 0;
	for (int i = 0; i < prefixSum.size(); i++) {
		prefixSumFinal += prefixSum[i];
	}

	joinFinal = 0;
	for (int i = 0; i < joinTime.size(); i++) {
		joinFinal += joinTime[i];
	}

	printf("\n*** Execution time *****************************\n"
			"Inner pack: %lu\n"
			"Inner hash count: %lu\n"
			"Inner prefix sum: %lu\n"
			"Inner hash: %lu\n"
			"Outer pack: %lu\n"
			"Outer hash count: %lu\n"
			"Outer prefix sum: %lu\n"
			"Outer hash: %lu\n"
			"Index count: %lu\n"
			"Prefix Sum: %lu\n"
			"Join final: %lu\n", innerPackFinal, innerHashCountFinal, innerPrefixFinal, innerHashFinal,
									outerPackFinal, outerHashCountFinal, outerPrefixFinal, outerHashFinal,
									indexCountFinal, prefixSumFinal, joinFinal);


	checkCudaErrors(cudaFree(outer_dev));
	checkCudaErrors(cudaFree(inner_dev));

	if (initial_size_ > 0)
		checkCudaErrors(cudaFree(initial_dev));

	if (end_size_ > 0)
		checkCudaErrors(cudaFree(end_dev));

	if (post_size_ > 0)
		checkCudaErrors(cudaFree(post_dev));

	if (where_size_ > 0)
		checkCudaErrors(cudaFree(where_dev));

	checkCudaErrors(cudaFree(search_exp_dev));
	checkCudaErrors(cudaFree(search_exp_size));
	checkCudaErrors(cudaFree(indices_dev));
	checkCudaErrors(cudaFree(stack));
	checkCudaErrors(cudaFree(indexCount));
	checkCudaErrors(cudaFree(innerHash.bucketLocation));
	checkCudaErrors(cudaFree(innerHash.hashedIdx));
	checkCudaErrors(cudaFree(innerHash.hashedKey));
	checkCudaErrors(cudaFree(outerHash.bucketLocation));
	checkCudaErrors(cudaFree(outerHash.hashedIdx));
	checkCudaErrors(cudaFree(outerHash.hashedKey));

	return true;
}
}
