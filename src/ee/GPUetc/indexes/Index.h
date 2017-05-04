#ifndef GINDEX_H_
#define GINDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "TreeIndex.h"
#include "HashIndex.h"

namespace voltdb {

class GIndex {
public:
	virtual ~GIndex();

	virtual void addEntry(GTuple new_tuple) {
		printf("Unsupported operation\n");
	}

	virtual void addBatchEntry(GTable table, int base_idx, int size) {
		printf("Unsupported operation\n");
	}

	virtual void merge(int old_left, int old_right, int new_left, int new_right) {
		printf("Unsupported operation\n");
	}

	virtual void removeIndex() {
		printf("Unsupported operation\n");
	}
};

}

#endif
