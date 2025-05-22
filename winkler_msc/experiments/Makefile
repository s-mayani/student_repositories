CLANG = clang++
CLANG_REL_FLAGS = -O3 -std=c++20 -march=native
NVCC_REL_FLAGS = -arch=compute_86 -code=sm_86 -O3 -DNDEBUG -std=c++20 --compiler-options -O3,-march=native
NVCC_DEB_FLAGS = -arch=compute_86 -code=sm_86 -G -std=c++20 --compiler-options -g
CUDA_FLAGS = --cuda-gpu-arch=sm_86
LIBS = -L /opt/cuda/lib64 -lcudart_static -lkokkoscore -lmpi -lcurand_static -lculibos

TARGET = mpi_schtuff
SRC = mpi_schtuff.cu
.PHONY: all clean
all: mpi_schtuff gridtest gridtest_clang
cfg.o: cfg.cpp
	-$(CLANG) $(CLANG_REL_FLAGS) $^ -c
stbi.o: stbi.cpp
	-$(CLANG) $(CLANG_REL_FLAGS) $^ -c
mpi_schtuff: mpi_schtuff.cu
	-$(CLANG) $(CLANG_REL_FLAGS) $(CUDA_FLAGS) $^ $(LIBS) -o $@
#shared_mem: shared_mem.cu
#	-$(CLANG) $(CLANG_REL_FLAGS) $(CUDA_FLAGS) -DNDEBUG $^ $(LIBS) -o $@
gridtest_clang: gridtest.cu cfg.o stbi.o
	$(CLANG) $(CLANG_REL_FLAGS) $(CUDA_FLAGS) $^ $(LIBS) -o $@
gridtest: gridtest.cu cfg.o stbi.o
	nvcc $(NVCC_REL_FLAGS) --expt-relaxed-constexpr --extended-lambda $^ $(LIBS) -o $@

clean:
	rm -f mpi_schtuff shared_mem gridtest gridtest_clang stbi.o cfg.o
