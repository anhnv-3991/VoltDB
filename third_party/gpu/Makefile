########################################
# makefile for check_ComputeCapability #
########################################

TARGET	= check_cc
SOURCE	= check_cc.cpp

CPP_COMPILER	= g++

INC_PATH	= -I/usr/local/cuda/include
LIB_PATH	= -L/usr/local/cuda/lib
LIB	= -lcuda

all:
	$(CPP_COMPILER) $(INC_PATH) $(LIB_PATH) -o $(TARGET) $(SOURCE) $(LIB)

clean:
	rm -f ./*~ ./$(TARGET)
