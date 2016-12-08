
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <inttypes.h>


#include "ghash.h"


extern "C" {
#define MASK_BITS 0x9e3779b9


__device__ void keyGenerate(GNValue *tuple, int *keyIndices, int indexNum, uint64_t *packedKey)
{
	int keyOffset = 0;
	int intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);

	if (keyIndices != NULL) {
		for (int i = 0; i < indexNum; i++) {
			//uint64_t keyValue = static_cast<uint64_t>(tuple[keyIndices[i]].getValue() + INT64_MAX + 1);

			switch (tuple[keyIndices[i]].getValueType()) {
				case VALUE_TYPE_TINYINT: {
					printf("Value Tiny Integer\n");
					uint64_t keyValue = static_cast<uint8_t>((int8_t)tuple[keyIndices[i]].getValue() + INT8_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
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
					printf("Value Small Integer\n");
					uint64_t keyValue = static_cast<uint16_t>((int16_t)tuple[keyIndices[i]].getValue() + INT16_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
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
					uint64_t keyValue = static_cast<uint32_t>((int32_t)tuple[keyIndices[i]].getValue() + INT32_MAX + 1);

					if (keyValue == tuple[keyIndices[i]].getValue())
						printf("Error at inner\n");

					for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= ((0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8));
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_BIGINT: {
					uint64_t keyValue = static_cast<uint64_t>((int64_t)tuple[keyIndices[i]].getValue() + INT64_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
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
					return;
				}
			}
		}
	} else {
		for (int i = 0; i < indexNum; i++) {
			//uint64_t keyValue = static_cast<uint64_t>(tuple[i].getValue() + INT64_MAX + 1);

			switch (tuple[i].getValueType()) {
				case VALUE_TYPE_TINYINT: {
					uint64_t keyValue = static_cast<uint8_t>((int8_t)tuple[i].getValue() + INT8_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
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
					printf("Outer Value Small Integer\n");
					uint64_t keyValue = static_cast<uint16_t>((int16_t)tuple[i].getValue() + INT16_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
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
					uint64_t keyValue = static_cast<uint32_t>((int32_t)tuple[i].getValue() + INT32_MAX + 1);

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
					uint64_t keyValue = static_cast<uint64_t>((int64_t)tuple[i].getValue() + INT64_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
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
					return;
				}
			}

		}
	}
}

__device__ uint64_t hasher(uint64_t *packedKey, int keySize)
{
	uint64_t seed = 0;

	for (int i = 0; i <  keySize; i++) {
		seed ^= packedKey[i] + MASK_BITS + (seed << 6) + (seed >> 2);
	}

	return seed;
}

__device__ bool equalityChecker(uint64_t *leftKey, uint64_t *rightKey, int keySize)
{
	bool res = true;

	while (--keySize >= 0) {
		res &= (leftKey[keySize] == rightKey[keySize]);
	}

	return res;
}

__device__ GNValue hashEvaluate(GTreeNode *tree_expression,
									int tree_size,
									GNValue *outer_tuple,
									GNValue *inner_tuple,
									GNValue *stack,
									int offset)
{
	int top = 0;
	memset(&stack[0], 0, sizeof(GNValue));

	for (int i = 0; i < tree_size; i++) {

		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[tree_expression[i].column_idx];
					top += offset;
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx];
					top += offset;
				}
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				stack[top] = tree_expression[i].value;
				top += offset;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_and(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_or(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_equal(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_notEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_lessThan(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_lessThanOrEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_greaterThan(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_greaterThanOrEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_add(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_subtract(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_divide(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_multiply(stack[top - offset]);
				top -= offset;

				break;
			}
			default: {
				return GNValue::getFalse();
			}
		}
	}

	return stack[0];
}


__device__ GNValue hashEvaluate2(GTreeNode *tree_expression,
									int tree_size,
									GNValue *outer_tuple,
									GNValue *inner_tuple,
									int64_t *stack,
									ValueType *gtype,
									int offset)
{
	ValueType ltype, rtype;
	int l_idx, r_idx;

	int top = 0;
	double left_d, right_d, res_d;
	int64_t left_i, right_i;

	for (int i = 0; i < tree_size; i++) {

		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[tree_expression[i].column_idx].getValue();
					gtype[top] = outer_tuple[tree_expression[i].column_idx].getValueType();
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx].getValue();
					gtype[top] = inner_tuple[tree_expression[i].column_idx].getValueType();

				}

				top += offset;
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				stack[top] = (tree_expression[i].value).getValue();
				gtype[top] = (tree_expression[i].value).getValueType();
				top += offset;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				if (gtype[l_idx] == VALUE_TYPE_BOOLEAN && gtype[r_idx] == VALUE_TYPE_BOOLEAN) {
					stack[l_idx] = (int64_t)((bool)(stack[l_idx]) && (bool)(stack[r_idx]));
					gtype[l_idx] = VALUE_TYPE_BOOLEAN;
				} else {
					stack[l_idx] = 0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				if (gtype[l_idx] == VALUE_TYPE_BOOLEAN && gtype[r_idx] == VALUE_TYPE_BOOLEAN) {
					stack[l_idx] = (int64_t)((bool)(stack[l_idx]) || (bool)(stack[r_idx]));
					gtype[l_idx] = VALUE_TYPE_BOOLEAN;
				} else {
					stack[l_idx] = 0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					stack[l_idx] =  (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d == right_d) : (left_i == right_i);
					gtype[l_idx] = VALUE_TYPE_BOOLEAN;
				} else {
					stack[l_idx] =  0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d != right_d) : (left_i != right_i);
					gtype[r_idx] = VALUE_TYPE_BOOLEAN;
				} else {
					stack[l_idx] =  0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d < right_d) : (left_i < right_i);
					gtype[l_idx] = VALUE_TYPE_BOOLEAN;
				} else {
					stack[l_idx] =  0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;

				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d <= right_d) : (left_i <= right_i);
					gtype[l_idx] = VALUE_TYPE_BOOLEAN;
				} else {
					stack[l_idx] =  0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d > right_d) : (left_i > right_i);
					gtype[l_idx] = VALUE_TYPE_BOOLEAN;
				} else {
					stack[l_idx] = 0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					stack[l_idx] = (int64_t)((ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d >= right_d) : (left_i >= right_i));
					gtype[l_idx] = VALUE_TYPE_BOOLEAN;
				} else {
					stack[l_idx] =  0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}

			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					res_d = left_d + right_d;
					if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
						stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
						gtype[l_idx] = VALUE_TYPE_DOUBLE;
					} else {
						stack[l_idx] = left_i + right_i;
						gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
					}
				} else {
					stack[l_idx] =  0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					res_d = left_d - right_d;
					if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
						stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
						gtype[l_idx] = VALUE_TYPE_DOUBLE;
					} else {
						stack[l_idx] = left_i - right_i;
						gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
					}
				} else {
					stack[l_idx] =  0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					res_d = left_d * right_d;
					if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
						stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
						gtype[l_idx] = VALUE_TYPE_DOUBLE;
					} else {
						stack[l_idx] = left_i * right_i;
						gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
					}
				} else {
					stack[l_idx] =  0;
					gtype[l_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				l_idx = top - 2 * offset;
				r_idx = top - offset;
				ltype = gtype[l_idx];
				rtype = gtype[r_idx];
				if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
					left_i = stack[l_idx];
					right_i = stack[r_idx];
					left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
					right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
					res_d = (right_d != 0) ? left_d / right_d : 0;
					if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
						stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
						gtype[l_idx] = (right_d != 0) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_INVALID;
					} else {
						stack[l_idx] = (right_i != 0) ? left_i / right_i : 0;
						gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
						gtype[l_idx] = (right_i != 0) ? ltype : VALUE_TYPE_INVALID;
					}
				} else {
					stack[l_idx] =  0;
					gtype[r_idx] = VALUE_TYPE_INVALID;
				}
				top = r_idx;
				break;
			}
			default: {
				return GNValue::getFalse();
			}
		}
	}

	GNValue retval(gtype[0], stack[0]);

	return retval;
}

__device__ GNValue hashEvaluate3(GTreeNode *tree_expression,
									int root,
									int tree_size,
									GNValue *outer_tuple,
									GNValue *inner_tuple)
{
	if (root == 0)
		return GNValue::getTrue();

	if (root >= tree_size)
		return GNValue::getNullValue();

	GTreeNode tmp_node = tree_expression[root];

	if (tmp_node.type == EXPRESSION_TYPE_VALUE_TUPLE) {
		if (tmp_node.tuple_idx == 0) {
			return outer_tuple[tmp_node.column_idx];
		} else if (tmp_node.tuple_idx == 1) {
			return inner_tuple[tmp_node.column_idx];
		}
	} else if (tmp_node.type == EXPRESSION_TYPE_VALUE_CONSTANT || tmp_node.type == EXPRESSION_TYPE_VALUE_PARAMETER) {
		return tmp_node.value;
	}


	GNValue left = hashEvaluate3(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
	GNValue right = hashEvaluate3(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);


	switch (tmp_node.type) {
	case EXPRESSION_TYPE_CONJUNCTION_AND: {
		return left.op_and(right);
	}
	case EXPRESSION_TYPE_CONJUNCTION_OR: {
		return left.op_or(right);
	}
	case EXPRESSION_TYPE_COMPARE_EQUAL: {
		return left.op_equal(right);
	}
	case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
		return left.op_notEqual(right);
	}
	case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
		return left.op_lessThan(right);
	}
	case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
		return left.op_lessThanOrEqual(right);
	}
	case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
		return left.op_greaterThan(right);
	}
	case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
		return left.op_greaterThanOrEqual(right);
	}
	case EXPRESSION_TYPE_OPERATOR_PLUS: {
		return left.op_add(right);
	}
	case EXPRESSION_TYPE_OPERATOR_MINUS: {
		return left.op_subtract(right);
	}
	case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
		return left.op_multiply(right);
	}
	case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
		return left.op_divide(right);
	}
	default:
		return GNValue::getNullValue();
	}
}

__global__ void packKey(GNValue *index_table, int tuple_num, int col_num, int *indices, int index_num, uint64_t *packedKey, int keySize)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < tuple_num; i += stride) {
		keyGenerate(index_table + i * col_num, indices, index_num, packedKey + i * keySize);
	}
}


__global__ void packSearchKey(GNValue *outer_table, int outer_rows, int outer_cols,
								uint64_t *searchPackedKey, GTreeNode *searchKeyExp,
								int *searchKeySize, int searchExpNum,
								int keySize,
#ifdef FUNC_CALL_
								GNValue *stack
#else
								int64_t *val_stack,
								ValueType *type_stack
#endif
								)
{
	int index = threadIdx.x + blockIdx.x *blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GNValue tmp_outer[4];
	int search_ptr = 0;

	for (int i = index; i < outer_rows; i += stride) {
		search_ptr = 0;
		for (int j = 0; j < searchExpNum; search_ptr += searchKeySize[j], j++) {
#ifdef FUNC_CALL_
			tmp_outer[j] = hashEvaluate(searchKeyExp + search_ptr, searchKeySize[j], outer_table + i * outer_cols, NULL, stack + index, stride);
#else
			tmp_outer[j] = hashEvaluate2(searchKeyExp + search_ptr, searchKeySize[j], outer_table + i * outer_cols, NULL, val_stack + index, type_stack + index, stride);
#endif
		}

		keyGenerate(tmp_outer, NULL, searchExpNum, searchPackedKey + i * keySize);
	}
}


__global__ void ghashCount(uint64_t *packedKey, int tupleNum, int keySize, ulong *hashCount, uint64_t maxNumberOfBuckets)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;


	for (int i = index; i < tupleNum; i += stride) {
		uint64_t hash = hasher(packedKey + i * keySize, keySize);
		uint64_t bucketOffset = hash % maxNumberOfBuckets;
		hashCount[bucketOffset * stride + index]++;
	}

}


__global__ void ghash(uint64_t *packedKey, ulong *hashCount, GHashNode hashTable)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int i;
	int keySize = hashTable.keySize;
	int maxNumberOfBuckets = hashTable.bucketNum;

	for (i = index; i <= maxNumberOfBuckets; i+= stride) {
		hashTable.bucketLocation[i] = hashCount[i * stride];
	}

	__syncthreads();

	for (i = index; i < hashTable.size; i += stride) {
		uint64_t hash = hasher(packedKey + i * keySize, keySize);
		uint64_t bucketOffset = hash % maxNumberOfBuckets;
		ulong hashIdx = hashCount[bucketOffset * stride + index];

		hashTable.hashedIdx[hashIdx] = i;

		for (int j = 0; j < keySize; j++) {
			hashTable.hashedKey[hashIdx * keySize + j] = packedKey[i * keySize + j];
		}

		hashCount[bucketOffset * stride + index]++;
	}
}


__global__ void hashIndexCount(GHashNode outerHash, GHashNode innerHash, int lowerBound, int upperBound, ulong *indexCount, int size)
{
	int outerIdx, innerIdx;
	ulong count_res = 0;
	int threadGlobalIndex = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int bucketIdx = lowerBound + blockIdx.x + blockIdx.y * gridDim.x;
	int keySize = outerHash.keySize;

	if (threadGlobalIndex < size && bucketIdx < upperBound) {
		for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx]; outerIdx < outerHash.bucketLocation[bucketIdx + 1]; outerIdx += blockDim.x)
			for (innerIdx = innerHash.bucketLocation[bucketIdx]; innerIdx < innerHash.bucketLocation[bucketIdx + 1]; innerIdx++)
				count_res += equalityChecker(outerHash.hashedKey + outerIdx * keySize, innerHash.hashedKey + innerIdx * keySize, keySize) ? 1 : 0;
		indexCount[threadGlobalIndex] = count_res;
	}
}

__global__ void hashIndexCountShared(GHashNode outerHash, GHashNode innerHash, int lowerBound, int upperBound, ulong *indexCount, int size)
{
	int outerIdx, innerIdx, endInnerIdx;
	ulong count_res = 0;
	int threadGlobalIndex = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int bucketIdx = lowerBound + blockIdx.x + blockIdx.y * gridDim.x;
	int keySize = outerHash.keySize;
	__shared__ uint64_t tmpInnerKey[SHARED_MEM];
	int realSize = 0;
	int sharedSize = SHARED_MEM;

	if (threadGlobalIndex < size && bucketIdx < upperBound) {
		for (innerIdx = innerHash.bucketLocation[bucketIdx], endInnerIdx = innerHash.bucketLocation[bucketIdx + 1]; innerIdx < endInnerIdx; innerIdx += (realSize/keySize)) {
			realSize = ((innerIdx + (sharedSize/keySize)) < endInnerIdx) ? ((sharedSize/keySize) * keySize) : ((endInnerIdx - innerIdx) * keySize);
			for (int i = threadIdx.x; i < realSize; i += blockDim.x) {
				tmpInnerKey[i] = innerHash.hashedKey[innerIdx * keySize + i];
			}

			__syncthreads();

			for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx]; outerIdx < outerHash.bucketLocation[bucketIdx + 1]; outerIdx += blockDim.x) {
				for (int tmpInnerIdx = 0; tmpInnerIdx < realSize/keySize; tmpInnerIdx++)
					count_res += equalityChecker(outerHash.hashedKey + outerIdx * keySize, tmpInnerKey + tmpInnerIdx * keySize, keySize) ? 1 : 0;
			}
			//__syncthreads();
		}
		indexCount[threadGlobalIndex] = count_res;
	}
}

__global__ void hashJoin(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							GHashNode outerHash,
							GHashNode innerHash,
							int lowerBound,
							int upperBound,
							ulong *indexCount,
							int size,
#ifdef FUNC_CALL_
							GNValue *stack,
#else
							int64_t *val_stack,
							ValueType *type_stack,
#endif
							RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int bucketIdx = lowerBound + blockIdx.x + blockIdx.y * gridDim.x;

	ulong write_location;
	int outerIdx, innerIdx;
	int outerTupleIdx, innerTupleIdx;
	int endOuterIdx, endInnerIdx;
	GNValue exp_check(VALUE_TYPE_BOOLEAN, true);

	if (index < size && bucketIdx < upperBound) {
		write_location = indexCount[index];
		for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx], endOuterIdx = outerHash.bucketLocation[bucketIdx + 1]; outerIdx < endOuterIdx; outerIdx += blockDim.x) {
			for (innerIdx = innerHash.bucketLocation[bucketIdx], endInnerIdx = innerHash.bucketLocation[bucketIdx + 1]; innerIdx < endInnerIdx; innerIdx++) {
				outerTupleIdx = outerHash.hashedIdx[outerIdx];
				innerTupleIdx = innerHash.hashedIdx[innerIdx];

				//key_check = equalityChecker(&outerHash.hashedKey[outerIdx * outerHash.keySize], &innerHash.hashedKey[innerIdx * outerHash.keySize], outerHash.keySize);

#ifdef FUNC_CALL_
				exp_check = (exp_check.isTrue()) ? hashEvaluate(end_expression, end_size,
																					outer_table + outerTupleIdx * outer_cols,
																					inner_table + innerTupleIdx * inner_cols,
																					stack + index, gridDim.x * gridDim.y * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate(post_expression, post_size,
																					outer_table + outerTupleIdx * outer_cols,
																					inner_table + innerTupleIdx * inner_cols,
																					stack + index, gridDim.x * gridDim.y * blockDim.x) : exp_check;
#else
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(end_expression, end_size,
																	outer_table + outerTupleIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * gridDim.y * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(post_expression, post_size,
																	outer_table + outerTupleIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * gridDim.y * blockDim.x) : exp_check;
#endif

				result[write_location].lkey = (exp_check.isTrue()) ? outerTupleIdx : result[write_location].lkey;
				result[write_location].rkey = (exp_check.isTrue()) ? innerTupleIdx : result[write_location].rkey;
				write_location += (exp_check.isTrue()) ? 1 : 0;
			}
		}
	}
}

__global__ void hashJoinShared(GNValue *outer_table, GNValue *inner_table,
								int outer_cols, int inner_cols,
								GTreeNode *end_expression, int end_size,
								GTreeNode *post_expression,	int post_size,
								GHashNode outerHash, GHashNode innerHash,
								int lowerBound, int upperBound,
								ulong *indexCount, int size,
#ifdef FUNC_CALL_
								GNValue *stack,
#else
								int64_t *val_stack,
								ValueType *type_stack,
#endif
								RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int bucketIdx = lowerBound + blockIdx.x + blockIdx.y * gridDim.x;

	ulong write_location;
	int outerIdx, innerIdx;
	int outerTupleIdx, innerTupleIdx;
	int endOuterIdx, endInnerIdx;
	GNValue exp_check(VALUE_TYPE_BOOLEAN, true);
	__shared__ GNValue tmpInner[SHARED_MEM];
	int realSize = 0;
	int sharedSize = SHARED_MEM;

	if (index < size && bucketIdx < upperBound) {
		write_location = indexCount[index];
		for (innerIdx = innerHash.bucketLocation[bucketIdx], endInnerIdx = innerHash.bucketLocation[bucketIdx + 1]; innerIdx < endInnerIdx; innerIdx += (sharedSize / inner_cols)) {
			realSize = ((innerIdx + (sharedSize/inner_cols)) < endInnerIdx) ? ((sharedSize/inner_cols) * inner_cols) : ((endInnerIdx - innerIdx) * inner_cols);
			for (int i = threadIdx.x; i < realSize; i += blockDim.x) {
				tmpInner[i] = inner_table[innerHash.hashedIdx[innerIdx + i / inner_cols] * inner_cols + i % inner_cols];
			}
			__syncthreads();

			for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx], endOuterIdx = outerHash.bucketLocation[bucketIdx + 1]; outerIdx < endOuterIdx; outerIdx += blockDim.x) {
				outerTupleIdx = outerHash.hashedIdx[outerIdx];
				for (int tmpInnerIdx = 0; tmpInnerIdx < realSize/inner_cols; tmpInnerIdx++) {
					innerTupleIdx = innerHash.hashedIdx[tmpInnerIdx + innerIdx];
#ifdef FUNC_CALL_
				exp_check = (exp_check.isTrue()) ? hashEvaluate(end_expression, end_size,
																	outer_table + outerTupleIdx * outer_cols,
																	tmpInner + tmpInnerIdx * inner_cols,
																	stack + index, gridDim.x * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate(post_expression, post_size,
																	outer_table + outerTupleIdx * outer_cols,
																	tmpInner + tmpInnerIdx * inner_cols,
																	stack + index, gridDim.x * blockDim.x) : exp_check;
#else
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(end_expression, end_size,
																	outer_table + outerTupleIdx * outer_cols,
																	tmpInner + tmpInnerIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(post_expression, post_size,
																	outer_table + outerTupleIdx * outer_cols,
																	tmpInner + tmpInnerIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x) : exp_check;
#endif

				result[write_location].lkey = (exp_check.isTrue()) ? (outerTupleIdx) : result[write_location].lkey;
				result[write_location].rkey = (exp_check.isTrue()) ? (innerTupleIdx) : result[write_location].rkey;
				write_location += (exp_check.isTrue()) ? 1 : 0;
				}
			}
			__syncthreads();
		}

	}
}

__global__ void ghashPhysical(GNValue *inputTable, GNValue *outputTable, int colNum, int rowNum, GHashNode hashTable)
{
	for (int index = threadIdx.x + blockIdx.x * blockDim.x; index < rowNum; index += blockDim.x * gridDim.x) {
		for (int i = 0; i < colNum; i++) {
			outputTable[index * colNum + i] = inputTable[hashTable.hashedIdx[index] * colNum + i];
		}
	}
}


__global__ void hashPhysicalJoin(GNValue *outer_table, GNValue *inner_table,
									int outer_cols, int inner_cols,
									GTreeNode *end_expression, int end_size,
									GTreeNode *post_expression,	int post_size,
									GHashNode outerHash,
									GHashNode innerHash,
									int lowerBound,
									int upperBound,
									ulong *indexCount,
									int size,
							#ifdef FUNC_CALL_
									GNValue *stack,
							#else
									int64_t *val_stack,
									ValueType *type_stack,
							#endif
									RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int bucketIdx = lowerBound + blockIdx.x;

	ulong write_location;
	int outerIdx, innerIdx;
	int endOuterIdx, endInnerIdx;
	GNValue exp_check(VALUE_TYPE_BOOLEAN, true);

	if (index < size && bucketIdx < upperBound) {
		write_location = indexCount[index];
		for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx], endOuterIdx = outerHash.bucketLocation[bucketIdx + 1]; outerIdx < endOuterIdx; outerIdx += blockDim.x) {
			for (innerIdx = innerHash.bucketLocation[bucketIdx], endInnerIdx = innerHash.bucketLocation[bucketIdx + 1]; innerIdx < endInnerIdx; innerIdx++) {

				//key_check = equalityChecker(&outerHash.hashedKey[outerIdx * outerHash.keySize], &innerHash.hashedKey[innerIdx * outerHash.keySize], outerHash.keySize);
				//GNValue exp_check(VALUE_TYPE_BOOLEAN, key_check);
#ifdef FUNC_CALL_
				exp_check = (exp_check.isTrue()) ? hashEvaluate(end_expression, end_size,
																	outer_table + outerIdx * outer_cols,
																	inner_table + innerIdx * inner_cols,
																	stack + index, gridDim.x * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate(post_expression, post_size,
																	outer_table + outerIdx * outer_cols,
																	inner_table + innerIdx * inner_cols,
																	stack + index, gridDim.x * blockDim.x) : exp_check;
#else
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(end_expression, end_size,
																	outer_table + outerIdx * outer_cols,
																	inner_table + innerIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(post_expression, post_size,
																	outer_table + outerIdx * outer_cols,
																	inner_table + innerIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x) : exp_check;
#endif

				result[write_location].lkey = (exp_check.isTrue()) ? outerHash.hashedIdx[outerIdx] : result[write_location].lkey;
				result[write_location].rkey = (exp_check.isTrue()) ? innerHash.hashedIdx[innerIdx] : result[write_location].rkey;
				write_location += (exp_check.isTrue()) ? 1 : 0;
			}
		}
	}
}

__global__ void hashJoin2(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							GHashNode outerHash, GHashNode innerHash,
							int baseOuterIdx, int baseInnerIdx,
							ulong *indexCount, int size,
#ifdef FUNC_CALL_
							GNValue *stack,
#else
							int64_t *val_stack,
							ValueType *type_stack,
#endif
							RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int bucketIdx = blockIdx.x + blockIdx.y * gridDim.x;

	ulong write_location;
	int outerIdx, innerIdx;
	int outerTupleIdx, innerTupleIdx;
	int endOuterIdx, endInnerIdx;
	GNValue exp_check(VALUE_TYPE_BOOLEAN, true);

	if (index < size && bucketIdx < outerHash.bucketNum) {
		write_location = indexCount[index];
		for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx], endOuterIdx = outerHash.bucketLocation[bucketIdx + 1]; outerIdx < endOuterIdx; outerIdx += blockDim.x) {
			for (innerIdx = innerHash.bucketLocation[bucketIdx], endInnerIdx = innerHash.bucketLocation[bucketIdx + 1]; innerIdx < endInnerIdx; innerIdx++) {
				outerTupleIdx = outerHash.hashedIdx[outerIdx];
				innerTupleIdx = innerHash.hashedIdx[innerIdx];

				//key_check = equalityChecker(&outerHash.hashedKey[outerIdx * outerHash.keySize], &innerHash.hashedKey[innerIdx * outerHash.keySize], outerHash.keySize);

#ifdef FUNC_CALL_
				exp_check = (exp_check.isTrue()) ? hashEvaluate(end_expression, end_size,
																	outer_table + outerTupleIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	stack + index, gridDim.x * blockDim.x * gridDim.y) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate(post_expression, post_size,
																	outer_table + outerTupleIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	stack + index, gridDim.x * blockDim.x * gridDim.y) : exp_check;
#else
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(end_expression, end_size,
																	outer_table + outerTupleIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x * gridDim.y) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(post_expression, post_size,
																	outer_table + outerTupleIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x * gridDim.y) : exp_check;
#endif

				result[write_location].lkey = (exp_check.isTrue()) ? (outerTupleIdx + baseOuterIdx) : result[write_location].lkey;
				result[write_location].rkey = (exp_check.isTrue()) ? (innerTupleIdx + baseInnerIdx) : result[write_location].rkey;
				write_location += (exp_check.isTrue()) ? 1 : 0;
			}
		}
	}
}

__global__ void hashJoin3(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							GHashNode outerHash,
							GHashNode innerHash,
							int lowerBound, int upperBound,
							int baseOuterIdx, int baseInnerIdx,
							ulong *indexCount,
							int size,
#ifdef FUNC_CALL_
							GNValue *stack,
#else
							int64_t *val_stack,
							ValueType *type_stack,
#endif
							RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int bucketIdx = lowerBound + blockIdx.x;

	ulong write_location;
	int outerIdx, innerIdx;
	int outerTupleIdx, innerTupleIdx;
	int endOuterIdx, endInnerIdx;
	GNValue exp_check(VALUE_TYPE_BOOLEAN, true);

	if (index < size && bucketIdx < upperBound) {
		write_location = indexCount[index];
		for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx], endOuterIdx = outerHash.bucketLocation[bucketIdx + 1]; outerIdx < endOuterIdx; outerIdx += blockDim.x) {
			for (innerIdx = innerHash.bucketLocation[bucketIdx], endInnerIdx = innerHash.bucketLocation[bucketIdx + 1]; innerIdx < endInnerIdx; innerIdx++) {
				outerTupleIdx = outerHash.hashedIdx[outerIdx];
				innerTupleIdx = innerHash.hashedIdx[innerIdx];

				//key_check = equalityChecker(&outerHash.hashedKey[outerIdx * outerHash.keySize], &innerHash.hashedKey[innerIdx * outerHash.keySize], outerHash.keySize);

#ifdef FUNC_CALL_
				exp_check = (exp_check.isTrue()) ? hashEvaluate(end_expression, end_size,
																					outer_table + outerTupleIdx * outer_cols,
																					inner_table + innerTupleIdx * inner_cols,
																					stack + index, gridDim.x * gridDim.y * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate(post_expression, post_size,
																					outer_table + outerTupleIdx * outer_cols,
																					inner_table + innerTupleIdx * inner_cols,
																					stack + index, gridDim.x * gridDim.y * blockDim.x) : exp_check;
#else
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(end_expression, end_size,
																	outer_table + outerTupleIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * gridDim.y * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(post_expression, post_size,
																	outer_table + outerTupleIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * gridDim.y * blockDim.x) : exp_check;
#endif

				result[write_location].lkey = (exp_check.isTrue()) ? (outerTupleIdx + baseOuterIdx) : result[write_location].lkey;
				result[write_location].rkey = (exp_check.isTrue()) ? (innerTupleIdx + baseInnerIdx) : result[write_location].rkey;
				write_location += (exp_check.isTrue()) ? 1 : 0;
			}
		}
	}
}



__global__ void hashJoinShared2(GNValue *outer_table, GNValue *inner_table,
								int outer_cols, int inner_cols,
								GTreeNode *end_expression, int end_size,
								GTreeNode *post_expression,	int post_size,
								GHashNode outerHash, GHashNode innerHash,
								int baseOuterIdx, int baseInnerIdx,
								ulong *indexCount, int size,
#ifdef FUNC_CALL_
								GNValue *stack,
#else
								int64_t *val_stack,
								ValueType *type_stack,
#endif
								RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	ulong write_location;
	int outerIdx, innerIdx;
	int outerTupleIdx, innerTupleIdx;
	int endOuterIdx, endInnerIdx;
	GNValue exp_check(VALUE_TYPE_BOOLEAN, true);
	__shared__ GNValue tmpInner[SHARED_MEM];
	int realSize = 0;
	int sharedSize = SHARED_MEM;
	int tmp = 0;

	if (index < size && blockIdx.x < outerHash.bucketNum) {
		write_location = indexCount[index];
		for (innerIdx = innerHash.bucketLocation[blockIdx.x], endInnerIdx = innerHash.bucketLocation[blockIdx.x + 1]; innerIdx < endInnerIdx; innerIdx += (sharedSize / inner_cols)) {
			//innerTupleIdx = innerHash.hashedIdx[innerIdx];
			tmp = sharedSize/inner_cols;
			realSize = ((innerIdx + tmp) < endInnerIdx) ? (tmp * inner_cols) : ((endInnerIdx - innerIdx) * inner_cols);
			for (int i = threadIdx.x; i < realSize; i += blockDim.x) {
				tmpInner[i] = inner_table[innerHash.hashedIdx[innerIdx + i / inner_cols] * inner_cols + i % inner_cols];
			}
			__syncthreads();

			for (outerIdx = threadIdx.x + outerHash.bucketLocation[blockIdx.x], endOuterIdx = outerHash.bucketLocation[blockIdx.x + 1]; outerIdx < endOuterIdx; outerIdx += blockDim.x) {
				outerTupleIdx = outerHash.hashedIdx[outerIdx];
				for (int tmpInnerIdx = 0; tmpInnerIdx < realSize/inner_cols; tmpInnerIdx++) {
					innerTupleIdx = innerHash.hashedIdx[innerIdx + tmpInnerIdx];
#ifdef FUNC_CALL_
					exp_check = (exp_check.isTrue()) ? hashEvaluate(end_expression, end_size,
																	outer_table + outerTupleIdx * outer_cols,
																	tmpInner + tmpInnerIdx * inner_cols,
																	stack + index, gridDim.x * blockDim.x) : exp_check;
					exp_check = (exp_check.isTrue()) ? hashEvaluate(post_expression, post_size,
																	outer_table + outerTupleIdx * outer_cols,
																	tmpInner + tmpInnerIdx * inner_cols,
																	stack + index, gridDim.x * blockDim.x) : exp_check;
#else
					exp_check = (exp_check.isTrue()) ? hashEvaluate2(end_expression, end_size,
																	outer_table + outerTupleIdx * outer_cols,
																	tmpInner + tmpInnerIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x) : exp_check;
					exp_check = (exp_check.isTrue()) ? hashEvaluate2(post_expression, post_size,
																	outer_table + outerTupleIdx * outer_cols,
																	tmpInner + tmpInnerIdx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x) : exp_check;
#endif

					result[write_location].lkey = (exp_check.isTrue()) ? (outerTupleIdx + baseOuterIdx) : result[write_location].lkey;
					result[write_location].rkey = (exp_check.isTrue()) ? (innerTupleIdx + baseInnerIdx) : result[write_location].rkey;
					write_location += (exp_check.isTrue()) ? 1 : 0;
				}
			}
			__syncthreads();
		}

	}
}


__global__ void hashIndexCountLegacy(uint64_t *outer_key, int outer_rows, GHashNode innerHash, ulong *indexCount, int size)
{
	int outerIdx, innerIdx;
	ulong count_res = 0;
	int keySize = innerHash.keySize;
	int maxNumberOfBuckets = innerHash.bucketNum;

	for (outerIdx = threadIdx.x + blockIdx.x * blockDim.x; outerIdx < outer_rows; outerIdx += blockDim.x * gridDim.x) {
		uint64_t hashVal = hasher(outer_key + outerIdx * keySize, keySize);
		uint64_t bucketOffset = hashVal % maxNumberOfBuckets;

		for (innerIdx = innerHash.bucketLocation[bucketOffset]; innerIdx < innerHash.bucketLocation[bucketOffset + 1]; innerIdx++) {
			count_res += (equalityChecker(outer_key + outerIdx * keySize, innerHash.hashedKey + innerIdx * keySize, keySize)) ? 1 : 0;
		}
		indexCount[outerIdx] = count_res;
	}
}

__global__ void hashJoinLegacy(GNValue *outer_table, GNValue *inner_table,
								int outer_cols, int inner_cols,
								int outer_rows,
								uint64_t *outerKey,
								GTreeNode *end_expression, int end_size,
								GTreeNode *post_expression,	int post_size,
								GHashNode innerHash,
								int baseOuterIdx, int baseInnerIdx,
								ulong *indexCount, int size,
#ifdef FUNC_CALL_
								GNValue *stack,
#else
								int64_t *val_stack,
								ValueType *type_stack,
#endif
								RESULT *result)
{
	GNValue exp_check(VALUE_TYPE_BOOLEAN, true);
	int keySize = innerHash.keySize;
	int maxNumberOfBuckets = innerHash.bucketNum;


	for (int outerIdx = threadIdx.x + blockDim.x * blockIdx.x; outerIdx < outer_rows; outerIdx += blockDim.x * gridDim.x) {
		ulong write_location = indexCount[outerIdx];
		uint64_t hashVal = hasher(outerKey + outerIdx * keySize, keySize);
		uint64_t bucketOffset = hashVal % maxNumberOfBuckets;

		for (int bucketIdx = innerHash.bucketLocation[bucketOffset], endBucketIdx = innerHash.bucketLocation[bucketOffset + 1]; bucketIdx < endBucketIdx; bucketIdx++) {
			int innerIdx = innerHash.hashedIdx[bucketIdx];
#ifdef FUNC_CALL_
				exp_check = (exp_check.isTrue()) ? hashEvaluate(end_expression, end_size,
																	outer_table + outerIdx * outer_cols,
																	inner_table + innerIdx * inner_cols,
																	stack + outerIdx, gridDim.x * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate(post_expression, post_size,
																	outer_table + outerIdx * outer_cols,
																	inner_table + innerTupleIdx * inner_cols,
																	stack + outerIdx, gridDim.x * blockDim.x) : exp_check;
#else
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(end_expression, end_size,
																	outer_table + outerIdx * outer_cols,
																	inner_table + innerIdx * inner_cols,
																	val_stack + outerIdx, type_stack + outerIdx, gridDim.x * blockDim.x) : exp_check;
				exp_check = (exp_check.isTrue()) ? hashEvaluate2(post_expression, post_size,
																	outer_table + outerIdx * outer_cols,
																	inner_table + innerIdx * inner_cols,
																	val_stack + outerIdx, type_stack + outerIdx, gridDim.x * blockDim.x) : exp_check;
#endif

				result[write_location].lkey = (exp_check.isTrue()) ? (outerIdx + baseOuterIdx) : result[write_location].lkey;
				result[write_location].rkey = (exp_check.isTrue()) ? (innerIdx + baseInnerIdx) : result[write_location].rkey;
				write_location += (exp_check.isTrue()) ? 1 : 0;
		}
	}
}


void packKeyWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					GNValue *index_table,
					int tuple_num,
					int col_num,
					int *indices,
					int index_num,
					uint64_t *packedKey,
					int keySize)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	packKey<<<gridSize, blockSize>>>(index_table, tuple_num, col_num, indices, index_num, packedKey, keySize);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (packKey) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void ghashCountWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						uint64_t *packedKey,
						int keyNum,
						int keySize,
						ulong *hashCount,
						uint64_t maxNumberOfBuckets
						)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	ghashCount<<<gridSize, blockSize>>>(packedKey, keyNum, keySize, hashCount, maxNumberOfBuckets);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (ghashCount) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void ghashWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					uint64_t *packedKey,
					ulong *hashCount,
					GHashNode hashTable
					)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	ghash<<<gridSize, blockSize>>>(packedKey, hashCount, hashTable);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (ghash) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void ghashPhysicalWrapper(int block_x, int block_y, int grid_x, int grid_y,
							GNValue *inputTable, GNValue *outputTable,
							int colNum, int rowNum, GHashNode hashTable)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	ghashPhysical<<<gridSize, blockSize>>>(inputTable, outputTable, colNum, rowNum, hashTable);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (ghashPhysical) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void packSearchKeyWrapper(int block_x, int block_y,
							int grid_x, int grid_y,
							GNValue *outer_table, int outer_rows, int outer_cols,
							uint64_t *searchPackedKey, GTreeNode *searchKeyExp,
							int *searchKeySize, int searchExpNum,
							int keySize,
#ifdef FUNC_CALL_
							GNValue *stack
#else
							int64_t *val_stack,
							ValueType *type_stack
#endif
							)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	packSearchKey<<<gridSize, blockSize>>>(outer_table, outer_rows, outer_cols, searchPackedKey, searchKeyExp, searchKeySize, searchExpNum, keySize,
#ifdef FUNC_CALL_
											stack
#else
											val_stack,
											type_stack
#endif
											);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (ghash) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());

}

void indexCountWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						GHashNode outerHash,
						GHashNode innerHash,
						int lowerBound,
						int upperBound,
						ulong *indexCount,
						int size
						)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

#ifndef SHARED_
	hashIndexCount<<<gridSize, blockSize>>>(outerHash, innerHash, lowerBound, upperBound, indexCount, size);
#else
	hashIndexCountShared<<<gridSize, blockSize>>>(outerHash, innerHash, lowerBound, upperBound, indexCount, size);
#endif
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashIndexCount) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void hashJoinWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						GNValue *outer_table,
						GNValue *inner_table,
						int outer_cols,
						int inner_cols,
						GTreeNode *end_expression,
						int end_size,
						GTreeNode *post_expression,
						int post_size,
						GHashNode outerHash,
						GHashNode innerHash,
						int lowerBound,
						int upperBound,
						ulong *indexCount,
						int size,
#ifdef FUNC_CALL_
						GNValue *stack,
#else
						int64_t *val_stack,
						ValueType *type_stack,
#endif
						RESULT *result
						)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

#ifndef SHARED_
	hashJoin<<<gridSize, blockSize>>>(outer_table, inner_table,
										outer_cols, inner_cols,
										end_expression, end_size,
										post_expression, post_size,
										outerHash, innerHash,
										lowerBound, upperBound,
										indexCount, size,
#ifdef FUNC_CALL_
										stack,
#else
										val_stack,
										type_stack,
#endif
										result);
#else
	hashJoinShared<<<gridSize, blockSize>>>(outer_table, inner_table,
													outer_cols, inner_cols,
													end_expression, end_size,
													post_expression, post_size,
													outerHash, innerHash,
													lowerBound, upperBound,
													indexCount, size,
#ifdef FUNC_CALL_
													stack,
#else
													val_stack,
													type_stack,
#endif
													result);
#endif
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashJoin) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}



void hashPhysicalJoinWrapper(int block_x, int block_y,
								int grid_x, int grid_y,
								GNValue *outer_table,
								GNValue *inner_table,
								int outer_cols,
								int inner_cols,
								GTreeNode *end_expression,
								int end_size,
								GTreeNode *post_expression,
								int post_size,
								GHashNode outerHash,
								GHashNode innerHash,
								int lowerBound,
								int upperBound,
								ulong *indexCount,
								int size,
#ifdef FUNC_CALL_
								GNValue *stack,
#else
								int64_t *val_stack,
								ValueType *type_stack,
#endif
								RESULT *result
								)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	hashPhysicalJoin<<<gridSize, blockSize>>>(outer_table, inner_table,
												outer_cols, inner_cols,
												end_expression, end_size,
												post_expression, post_size,
												outerHash, innerHash,
												lowerBound, upperBound,
												indexCount, size,
#ifdef FUNC_CALL_
												stack,
#else
												val_stack,
												type_stack,
#endif
												result);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashPhysicalJoin) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void hashJoinWrapper2(int block_x, int block_y, int grid_x, int grid_y,
						GNValue *outer_table, GNValue *inner_table,
						int outer_cols, int inner_cols,
						GTreeNode *end_expression, int end_size,
						GTreeNode *post_expression, int post_size,
						GHashNode outerHash, GHashNode innerHash,
						int baseOuterIdx, int baseInnerIdx,
						ulong *indexCount, int size,
#ifdef FUNC_CALL_
						GNValue *stack,
#else
						int64_t *val_stack,
						ValueType *type_stack,
#endif
						RESULT *result)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

#ifndef SHARED_
	hashJoin2<<<gridSize, blockSize>>>(outer_table, inner_table,
												outer_cols, inner_cols,
												end_expression, end_size,
												post_expression, post_size,
												outerHash, innerHash,
												baseOuterIdx, baseInnerIdx,
												indexCount, size,
#ifdef FUNC_CALL_
												stack,
#else
												val_stack,
												type_stack,
#endif
												result);
#else
	hashJoinShared2<<<gridSize, blockSize>>>(outer_table, inner_table,
													outer_cols, inner_cols,
													end_expression, end_size,
													post_expression, post_size,
													outerHash, innerHash,
													baseOuterIdx, baseInnerIdx,
													indexCount, size,
#ifdef FUNC_CALL_
													stack,
#else
													val_stack,
													type_stack,
#endif
													result);
#endif
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashPhysicalJoin) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void indexCountLegacyWrapper(int block_x, int block_y,
								int grid_x, int grid_y,
								uint64_t *outerKey,
								int outer_rows,
								GHashNode innerHash,
								ulong *indexCount,
								int size)
{
	dim3 blockSize(block_x, block_y, 1);
	dim3 gridSize(grid_x, grid_y, 1);

	hashIndexCountLegacy<<<gridSize, blockSize>>>(outerKey, outer_rows, innerHash, indexCount, size);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashIndexCountLegacy) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void hashJoinLegacyWrapper(int block_x, int block_y, int grid_x, int grid_y,
							GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							int outer_rows,
							uint64_t *outerKey,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							GHashNode innerHash,
							int baseOuterIdx, int baseInnerIdx,
							ulong *indexCount, int size,
#ifdef FUNC_CALL_
							GNValue *stack,
#else
							int64_t *val_stack,
							ValueType *type_stack,
#endif
							RESULT *result)
{
	dim3 blockSize(block_x, block_y, 1);
	dim3 gridSize(grid_x, grid_y, 1);

	hashJoinLegacy<<<gridSize, blockSize>>>(outer_table, inner_table,
												outer_cols, inner_cols,
												outer_rows, outerKey,
												end_expression, end_size,
												post_expression, post_size,
												innerHash,
												baseOuterIdx, baseInnerIdx,
												indexCount, size,
#ifdef FUNC_CALL_
												stack,
#else
												val_stack,
												type_stack,
#endif
												result);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashJoinLegacy) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}


void hashJoinWrapper3(int block_x, int block_y,
						int grid_x, int grid_y,
						GNValue *outer_table, GNValue *inner_table,
						int outer_cols, int inner_cols,
						GTreeNode *end_expression, int end_size,
						GTreeNode *post_expression, int post_size,
						GHashNode outerHash, GHashNode innerHash,
						int lowerBound, int upperBound,
						int outerBaseIdx, int innerBaseIdx,
						ulong *indexCount, int size,
#ifdef FUNC_CALL_
						GNValue *stack,
#else
						int64_t *val_stack,
						ValueType *type_stack,
#endif
						RESULT *result
						)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	hashJoin3<<<gridSize, blockSize>>>(outer_table, inner_table,
										outer_cols, inner_cols,
										end_expression, end_size,
										post_expression, post_size,
										outerHash, innerHash,
										lowerBound, upperBound,
										outerBaseIdx, innerBaseIdx,
										indexCount, size,
#ifdef FUNC_CALL_
										stack,
#else
										val_stack,
										type_stack,
#endif
										result);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashJoin3) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}


void hprefixSumWrapper(ulong *input, int ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}
}
