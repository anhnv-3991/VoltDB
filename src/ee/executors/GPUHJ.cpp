#include "GPUHJ.h"

GPUHJ::GPUHJ()
{
		outer_table_ = inner_table_ = NULL;
		outer_rows_ =  0;
		outer_rows_ = 0;
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


uint GPUHJ::divUtility(uint dividend, uint divisor) const
{
	return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
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

bool GPUHJ::join()
{
	GNValue *outer_dev, *inner_dev;
	int outer_partition, inner_partition;
	GTreeNode *initial_dev, *end_dev, *post_dev, *where_dev, *search_exp_dev;
	int *indices_dev, *search_exp_size;


	outer_partition = (outer_rows_ < DEFAULT_PART_SIZE) ? outer_rows_ : DEFAULT_PART_SIZE;
	inner_partition = (inner_rows_ < DEFAULT_PART_SIZE) ? inner_rows_ : DEFAULT_PART_SIZE;

	/* Allocate table memory */
	checkCudaErrors(cudaMalloc(&outer_dev, sizeof(GNValue) * outer_cols_ * outer_partition));
	checkCudaErrors(cudaMalloc(&inner_dev, sizeof(GNValue) * inner_cols_ * inner_partition));

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

	checkCudaErrors(cudaMalloc(indices_dev, sizeof(int) * indices_size_));
	checkCudaErrors(cudaMemcpy(indices_dev, indices_, sizeof(int) * indices_size_, cudaMemcpyHostToDevice));

}
