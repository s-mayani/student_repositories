#include <cufft.h>
#include <cublas.h>
#include <cuda.h>
#include <fstream>
#include <iostream>

#ifndef COMMON_CUH
#define COMMON_CUH
#include <cstdio>
#include <exception>
#define kf __host__ __device__
#ifdef kokkoscuda
#include <Kokkos_Core.hpp>
#else
#define host_only __host__
#define device_only __device__
#define KOKKOS_FUNCTION __host__ __device__
#define KOKKOS_INLINE_FUNCTION __host__ __device__ inline
#define KOKKOS_FORCEINLINE_FUNCTION __host__ __device__ __attribute__((always_inline))
#define KOKKOS_LAMBDA [=] __host__ __device__
#endif
template <typename T>
T *cuda_malloc_helper(size_t n) {
    T *ptr;
    cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc of %lu bytes failed: %s\n", n * sizeof(T), cudaGetErrorString(err));
        abort();
    }
    //fprintf(stderr, "cudaMalloc of %lu bytes succeeded\n", n * sizeof(T));
    return ptr;
}
template <typename T>
T *cuda_malloc_host_helper(size_t n) {
    T *ptr;
    cudaError_t err = cudaMallocHost(&ptr, n * sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    return ptr;
}
template <typename T>
T *cuda_malloc_managed_helper(size_t n) {
    T *ptr;
    cudaError_t err = cudaMallocManaged(&ptr, n * sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    return ptr;
}
template <typename T, typename... Ts>
T *cuda_make_managed(const Ts &...x) {
    T *ptr;
    cudaError_t err = cudaMallocManaged(&ptr, sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_make_managed failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    new (ptr) T(x...);
    return ptr;
}
KOKKOS_INLINE_FUNCTION bool isNaN(float x){
    #ifdef __CUDA_ARCH__
    return isnan(x);
    #else
    return std::isnan(x);
    #endif
}
KOKKOS_INLINE_FUNCTION bool isINF(float x){
    #ifdef __CUDA_ARCH__
    return isinf(x);
    #else
    return std::isinf(x);
    #endif
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define assert_isreal(X)// assert(!isNaN(X) && !isINF(X))
#endif
using scalar = double;
#define idx(x,y,z) (x + y * N + z * N2)
__global__ void bench_stencil(const scalar* src, scalar* dest, int N, int N2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if(x == 0 || y == 0 || z == 0)return;
    if(x >= N - 1 || y >= N - 1 || z >= N - 1)return;
    scalar v1 = src[idx(x,y,z)   ];
    scalar v2 = 0.03f * src[idx(x,y,z)+1 ];
    scalar v3 = 0.04f * src[idx(x,y,z)-1 ];
    scalar v4 = 0.05f * src[idx(x,y,z)+N ];
    scalar v5 = 0.06f * src[idx(x,y,z)-N ];
    scalar v6 = 0.07f * src[idx(x,y,z)+N2];
    scalar v7 = 0.08 * src[idx(x,y,z)-N2];
    dest[x + y * N + z * N2] = v1 + 0.01f *(v2+v3+v4+v5+v6+v7);
}
__global__ void bench_stencil5(const scalar* src, scalar* dest, int N, int N2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if(x <= 1 || y <= 1 || z <= 1)return;
    if(x >= N - 2 || y >= N - 2 || z >= N - 2)return;
    scalar v1 =        src[idx(x,y,z)     ];
    scalar v2 = 0.03 * src[idx(x,y,z)+1   ];
    scalar v3 = 0.04f * src[idx(x,y,z)+2   ];
    scalar v4 = 0.05f * src[idx(x,y,z)-1   ];
    scalar v5 = 0.06f * src[idx(x,y,z)-2   ];
    scalar v6 = 0.07f * src[idx(x,y,z)+1*N ];
    scalar v7 = 0.08f * src[idx(x,y,z)+2*N ];
    scalar v8 = 0.09f * src[idx(x,y,z)-1*N ];
    scalar v9 = 0.10f * src[idx(x,y,z)-2*N ];
    scalar va = 0.11f * src[idx(x,y,z)+1*N2];
    scalar vb = 0.12f * src[idx(x,y,z)+2*N2];
    scalar vc = 0.13f * src[idx(x,y,z)-1*N2];
    scalar vd = 0.14f * src[idx(x,y,z)-2*N2];
    dest[x + y * N + z * N2] = v1 + 0.01f *(v2+v3+v4+v5+v6+v7+v8+v9+va+vb+vc+vd);
}
__global__ void bench_stencil7(const scalar* src, scalar* dest, int N, int N2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if(x <= 2 || y <= 2 || z <= 2)return;
    if(x >= N - 3 || y >= N - 3 || z >= N - 3)return;
    scalar v1 =        src[idx(x,y,z)     ];
    scalar v2 = 0.03f * src[idx(x,y,z)+1   ];
    scalar v3 = 0.04f * src[idx(x,y,z)+2   ];
    scalar v4 = 0.04f * src[idx(x,y,z)+3   ];
    scalar v5 = 0.05f * src[idx(x,y,z)-1   ];
    scalar v6 = 0.06f * src[idx(x,y,z)-2   ];
    scalar v7 = 0.06f * src[idx(x,y,z)-3   ];
    scalar v8 = 0.03f * src[idx(x,y,z)+1*N ];
    scalar v9 = 0.04f * src[idx(x,y,z)+2*N ];
    scalar va = 0.04f * src[idx(x,y,z)+3*N ];
    scalar vb = 0.05f * src[idx(x,y,z)-1*N ];
    scalar vc = 0.06f * src[idx(x,y,z)-2*N ];
    scalar vd = 0.06f * src[idx(x,y,z)-3*N ];
    scalar ve = 0.03f * src[idx(x,y,z)+1*N2];
    scalar vf = 0.04f * src[idx(x,y,z)+2*N2];
    scalar vg = 0.04f * src[idx(x,y,z)+3*N2];
    scalar vh = 0.05f * src[idx(x,y,z)-1*N2];
    scalar vi = 0.06f * src[idx(x,y,z)-2*N2];
    scalar vj = 0.06f * src[idx(x,y,z)-3*N2];
    dest[x + y * N + z * N2] = v1 + 0.01f *(v1+v2+v3+v4+v5+v6+v7+v8+v9+va+vb+vc+vd+ve+vf+vg+vh+vi+vj);
}
template<int stN>
__global__ void bench_stencil(const scalar* src, scalar* dest, int N, int N2){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if(x < stN || y < stN || z < stN)return;
    if(x >= N - stN || y >= N - stN || z >= N - stN)return;
    scalar destv = 0;
    for(int i = -stN;i <= stN;i++){
        scalar v1 = scalar(0.04) * src[idx(x,y,z)+i   ];
        scalar v2 = scalar(0.07) * src[idx(x,y,z)-i   ];
        scalar v3 = scalar(0.09) * src[idx(x,y,z)+i*N ];
        scalar v4 = scalar(0.01) * src[idx(x,y,z)-i*N ];
        scalar v5 = scalar(0.05) * src[idx(x,y,z)+i*N2];
        scalar v6 = scalar(0.11) * src[idx(x,y,z)-i*N2];
        destv += v1+v2+v3+v4+v5+v6;
    }
    dest[x + y * N + z * N2] = destv;
}
KOKKOS_INLINE_FUNCTION constexpr unsigned int idiv_roundup(unsigned int n, unsigned int d){
    return (n + d - 1) / d;
}
using complex = std::conditional_t<std::is_same_v<scalar, double>, cufftDoubleComplex, cufftComplex>;
int main(){
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::ofstream ctimes("cufft_times.txt");
    std::ofstream ntimes("naive3_times.txt");
    std::ofstream n5times("naive5_times.txt");
    std::ofstream n7times("naive7_times.txt");
    std::ofstream n20times("naive20_times.txt");
    int maxn = 1024;
    scalar* input  = cuda_malloc_helper<scalar>(size_t(maxn) * size_t(maxn) * size_t(maxn));
    
    complex* output = cuda_malloc_helper<complex>(size_t(maxn) * size_t(maxn) * (size_t(maxn) / 2 + 1));
    for(int _N = 28;_N < maxn;_N += 32){
        int N = _N + 4;
        std::cout << "Doing cufft with " << N << std::endl;
        cufftHandle plan;
        scalar alpha = 1.0f / (N * N * N);
        cufftPlan3d(&plan, N, N, N, CUFFT_R2C);
        for(size_t i = 0; i < 5;i++){
            if constexpr(std::is_same_v<scalar, double>){
                cufftExecD2Z  (plan, input, output);
                cufftExecZ2D  (plan, output, input);
                cublasDscal_v2(handle, N * N * N, &alpha, input, 1);
            }
            else if constexpr(std::is_same_v<scalar, float>){
                //cufftExecR2C  (plan, input, output);
                //cufftExecC2R  (plan, output, input);
                //cublasSscal_v2(handle, N * N * N, &alpha, input, 1);
            }
        }
        cudaEventRecord(start);
        for(size_t i = 0; i < 40;i++){
            if constexpr(std::is_same_v<scalar, double>){
                cufftExecD2Z  (plan, input, output);
                cufftExecZ2D  (plan, output, input);
                cublasDscal_v2(handle, N * N * N, &alpha, input, 1);
            }
            else if constexpr(std::is_same_v<scalar, float>){
                //cufftExecR2C  (plan, input, output);
                //cufftExecC2R  (plan, output, input);
                //cublasSscal_v2(handle, N * N * N, &alpha, input, 1);
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ctimes << _N << " " << ms << std::endl;
        cufftDestroy(plan);
        
    }
    scalar* dst_reinterpreted = reinterpret_cast<scalar*>(output);
    for(int _N = 28;_N < maxn;_N += 32){
        int N = _N;
        std::cout << "Doing naive with " << N << std::endl;
        cudaEventRecord(start);
        dim3 bd{idiv_roundup(N, 8), idiv_roundup(N, 8), idiv_roundup(N, 8)};
        dim3 td{8, 8, 8};
        for(size_t i = 0; i < 40;i++){
            bench_stencil<<<bd, td>>>(input, dst_reinterpreted, N, N * N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ntimes << N << " " << ms << std::endl;
    }
    for(int _N = 28;_N < maxn;_N += 32){
        int N = _N;
        std::cout << "Doing naive5 with " << N << std::endl;
        cudaEventRecord(start);
        dim3 bd{idiv_roundup(N, 8), idiv_roundup(N, 8), idiv_roundup(N, 8)};
        dim3 td{8, 8, 8};
        for(size_t i = 0; i < 40;i++){
            bench_stencil5<<<bd, td>>>(input, dst_reinterpreted, N, N * N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        n5times << N << " " << ms << std::endl;
    }
    for(int _N = 64;_N < maxn;_N += 31){
        int N = _N;
        std::cout << "Doing naive7 with " << N << std::endl;
        cudaEventRecord(start);
        dim3 bd{idiv_roundup(N, 8), idiv_roundup(N, 8), idiv_roundup(N, 8)};
        dim3 td{8, 8, 8};
        for(size_t i = 0; i < 40;i++){
            bench_stencil<7><<<bd, td>>>(input, dst_reinterpreted, N, N * N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        n7times << N << " " << ms << std::endl;
    }
    //for(int _N = 64;_N < maxn;_N += 31){
    //    int N = _N;
    //    std::cout << "Doing naive20 with " << N << std::endl;
    //    cudaEventRecord(start);
    //    dim3 bd{idiv_roundup(N, 8), idiv_roundup(N, 8), idiv_roundup(N, 8)};
    //    dim3 td{8, 8, 8};
    //    for(size_t i = 0; i < 40;i++){
    //        bench_stencil<20><<<bd, td>>>(input, dst_reinterpreted, N, N * N);
    //    }
    //    cudaEventRecord(stop);
    //    cudaEventSynchronize(stop);
    //    float ms;
    //    cudaEventElapsedTime(&ms, start, stop);
    //    n20times << N << " " << ms << std::endl;
    //}
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(output);
}