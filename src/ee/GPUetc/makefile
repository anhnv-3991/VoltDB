THIRD_PARTY_PATH = ../../../third_party/cpp

all:abstractexpression.o

abstractexpression.o:expressions/abstractexpression.cu
	nvcc -isystem $(THIRD_PARTY_PATH) -I. $^ -c -o $@
