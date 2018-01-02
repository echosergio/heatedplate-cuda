NVCC = nvcc
CUDAFLAGS = -arch=sm_35
# LIBDIRS = -L/usr/local/cuda-9.0/bin -L/usr/local/cuda-9.0/lib64

.SUFFIXES: .cpp .c .h .y .l .o .cu

.cu.o:
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

all: heated_plate

heated_plate: heated_plate.o
	$(NVCC) $(CUDAFLAGS) $< -o $@

clean:
	rm -f *.o heated_plate *~
