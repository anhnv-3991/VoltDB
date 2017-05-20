#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "gexpression.h"

namespace voltdb {

GExpression::GExpression() {
	expression_ = NULL;
	size_ = 0;
}

GExpression::GExpression(ExpressionNode *expression) {
	int tree_size = 0;

#ifndef TREE_EVAL_
	tree_size =	getExpressionLength(expression);
#else
	tree_size = getTreeSize(expression, tmp_size) + 1;
#endif

	checkCudaErrors(cudaMalloc(&expression_, tree_size * sizeof(GTreeNode)));
	createExpression(expression);
}

bool GExpression::createExpression(ExpressionNode *expression) {
	GTreeNode *tmp_expression = (GTreeNode*)malloc(sizeof(GTreeNode) * size_);
#ifndef TREE_EVAL_
	int root = 0;

	if (!buildPostExpression(tmp_expression, expression, &root))
		return false;
#else
	if (!buildTreeExpression(tmp_expression, expression, 1))
		return false;
#endif
	checkCudaErrors(cudaMemcpy(expression_, tmp_expression, sizeof(GTreeNode) * size_, cudaMemcpyHostToDevice));
	free(tmp_expression);

	return true;
}

void GExpression::freeExpression() {
	if (size_ > 0)
		checkCudaErrors(cudaFree(expression_));
}

int GExpression::getExpressionLength(ExpressionNode *expression) {
	if (expression == NULL) {
		return 0;
	}

	int left, right;

	left = getExpressionLength(expression->left);
	right = getExpressionLength(expression->right);

	return (1 + left + right);
}

int GExpression::getTreeSize(ExpressionNode *expression, int size) {
	if (expression == NULL)
		return size / 2;

	int left, right;

	left = getTreeSize(expression->left, size * 2);
	right = getTreeSize(expression->right, size * 2 + 1);

	return (left > right) ? left : right;
}

bool GExpression::buildPostExpression(GTreeNode *output_expression, ExpressionNode *expression, int *index) {
	if (expression == NULL)
		return true;

	if (size_ <= *index)
		return false;

	if (!buildPostExpression(output_expression, expression->left, index))
		return false;

	if (!buildPostExpression(output_expression, expression->right, index))
		return false;

	output_expression[*index] = expression->node;
	(*index)++;

	return true;
}

bool GExpression::buildTreeExpression(GTreeNode *output_expression, ExpressionNode *expression, int index) {
	if (expression == NULL)
		return true;

	if (size_ <= index)
		return false;

	expression_[index] = expression->node;
	if (!buildTreeExpression(output_expression, expression->left, index * 2))
		return false;

	if (!buildTreeExpression(output_expression, expression->right, index * 2 + 1))
		return false;

	return true;
}

}
