#include "grid.cu"
#include <cassert>
#include <cstdint>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <array>
#include <mpi.h>

struct indexmap {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    KOKKOS_INLINE_FUNCTION size_t operator()(uint32_t i, uint32_t j, uint32_t l) const noexcept {
        assert(i < m);
        assert(j < n);
        assert(l < k);
        return i * n * k + j * k + l;
    }
    KOKKOS_INLINE_FUNCTION size_t size() const noexcept {
        return m * n * k;
    }
};


enum axis_aligned_occlusion : int {
    NONE = 0,
    AT_MIN = -1,
    AT_MAX =  1,
};

KOKKOS_INLINE_FUNCTION axis_aligned_occlusion& operator|=(axis_aligned_occlusion &a, axis_aligned_occlusion b) {
    a = (axis_aligned_occlusion)(static_cast<int>(a) | static_cast<int>(b));
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion& operator-=(axis_aligned_occlusion &a, axis_aligned_occlusion b) {
    a = (axis_aligned_occlusion)(static_cast<int>(a) - static_cast<int>(b));
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion& operator-=(axis_aligned_occlusion &a, int b) {
    a = (axis_aligned_occlusion)(static_cast<int>(a) - static_cast<int>(b));
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion operator-(axis_aligned_occlusion a, axis_aligned_occlusion b) {
    a = (axis_aligned_occlusion)(static_cast<int>(a) - static_cast<int>(b));
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion operator-(axis_aligned_occlusion a, int b) {
    a = (axis_aligned_occlusion)(static_cast<int>(a) - static_cast<int>(b));
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion &operator|=(axis_aligned_occlusion &a, int b) {
    a = (axis_aligned_occlusion)(static_cast<int>(a) | b);
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion &operator&=(axis_aligned_occlusion &a, axis_aligned_occlusion b) {
    a = (axis_aligned_occlusion)(static_cast<int>(a) & static_cast<int>(b));
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion &operator&=(axis_aligned_occlusion &a, int b) {
    a = (axis_aligned_occlusion)(static_cast<int>(a) & b);
    return a;
}

template <typename T, unsigned n>
struct aray {
    T data[n];
    KOKKOS_INLINE_FUNCTION T &operator[](unsigned i) noexcept {
        return data[i];
    }
    KOKKOS_INLINE_FUNCTION const T &operator[](unsigned i) const noexcept {
        return data[i];
    }
    KOKKOS_INLINE_FUNCTION constexpr aray& operator+=(const aray& x)noexcept{
        for(unsigned i = 0;i < n;i++){
            data[i] += x[i];
        }
        return *this;
    }
};
template <typename index_type, unsigned Dim>
KOKKOS_INLINE_FUNCTION constexpr aray<axis_aligned_occlusion, Dim> boundary_occlusion_of(
    size_t boundary_distance, const aray<index_type, Dim> _index,
    const aray<index_type, Dim> _extents) {
    /*constexpr size_t Dim = std::tuple_size_v<std::tuple<extent_types...>>;

    constexpr auto get_array = []<typename... Ts>(Ts&&... x) {
        return aray<size_t, sizeof...(x)>{static_cast<size_t>(x)...};
    };*/

    aray<uint32_t, Dim> index = _index;
    aray<uint32_t, Dim> extents = _extents;
    aray<axis_aligned_occlusion, Dim> ret_array;

    uint32_t minimal_distance_to_zero = index[0];
    uint32_t minimal_distance_to_extent_minus_one = extents[0] - index[0] - 1;
    ret_array[0] = (axis_aligned_occlusion)(index[0] == boundary_distance);
    ret_array[0] -= int(index[0] == (extents[0] - 1 - boundary_distance));
    for (size_t i = 1; i < Dim; i++) {
        minimal_distance_to_zero = min(minimal_distance_to_zero, index[i]);
        minimal_distance_to_extent_minus_one =
            min(minimal_distance_to_extent_minus_one, extents[i] - index[i] - 1);
        ret_array[i] = (axis_aligned_occlusion)(index[i] == boundary_distance);
        ret_array[i] -= (int)(index[i] == (extents[i] - 1 - boundary_distance));
    }
    bool behindboundary = minimal_distance_to_zero < boundary_distance || minimal_distance_to_extent_minus_one < boundary_distance;
    if (behindboundary) {
        for (size_t i = 0; i < Dim; i++) {
            ret_array[i] = (axis_aligned_occlusion)0;
        }
    }
    return ret_array;
}

template <typename... Ts>
KOKKOS_INLINE_FUNCTION void primpf(const char *str, const Ts &...args) {
    printf(str, args...);
}

KOKKOS_INLINE_FUNCTION constexpr float sq(float v){
    return v * v;
}
template<int direction, typename T>
__global__ void stencil_iter_inplace(grid<aray<T, 2>> A) {
    constexpr T dt = 0.5;
    constexpr T dx = 1.0;
    constexpr T dy = 1.0;
    constexpr T dz = 1.0;

    constexpr T a1 = T(2) * (T(1) - sq(dt / dx) - sq(dt / dy) - sq(dt / dz));
    constexpr T a2 = sq(dt / dx);
    constexpr T a4 = sq(dt / dy);
    constexpr T a6 = sq(dt / dz);
    constexpr T a8 = sq(dt);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t j = threadIdx.y + blockDim.y * blockIdx.y;
    const uint32_t k = threadIdx.z + blockDim.z * blockIdx.z;
    ct_indexmap<8, 8, 16> shmem_imap;
    __shared__ aray<T, 2> shm[shmem_imap.total_size()];
    //shm[shmem_imap(tx, ty, tz)] = A(i, j, k)[0];
    if constexpr(direction == 0){
        A(i, j, k)[1] = -A(i, j, k)[1] + a1 * A(i, j, k)[0]
                                               + a2 * (A(i + 1, j, k)[0] + A(i - 1, j, k)[0])
                                               + a4 * (A(i, j + 1, k)[0] + A(i, j - 1, k)[0])
                                               + a6 * (A(i, j, k + 1)[0] + A(i, j, k - 1)[0]);
    }
    if constexpr(direction == 1){
        A(i, j, k)[0] = -A(i, j, k)[0] + a1 * A(i, j, k)[1]
                                               + a2 * (A(i + 1, j, k)[1] + A(i - 1, j, k)[1])
                                               + a4 * (A(i, j + 1, k)[1] + A(i, j - 1, k)[1])
                                               + a6 * (A(i, j, k + 1)[1] + A(i, j, k - 1)[1]);
    }
}
template <typename T>
__global__ void stencil_iter_trivial(grid<T> A_nm1, grid<T> A_n) {
    constexpr T dt = 0.5;
    constexpr T dx = 1.0;
    constexpr T dy = 1.0;
    constexpr T dz = 1.0;

    constexpr T a1 = T(2) * (T(1) - sq(dt / dx) - sq(dt / dy) - sq(dt / dz));
    constexpr T a2 = sq(dt / dx);
    constexpr T a4 = sq(dt / dy);
    constexpr T a6 = sq(dt / dz);
    constexpr T a8 = sq(dt);
    ct_indexmap<8, 8, 16> shmem_imap;
    __shared__ T shm[shmem_imap.total_size()];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    const uint32_t i = tx + blockDim.x * blockIdx.x;
    const uint32_t j = ty + blockDim.y * blockIdx.y;
    const uint32_t k = tz + blockDim.z * blockIdx.z;
    const uint32_t gr_imap_thread = A_n.imap(i, j, k);
    const uint32_t minstep = (A_n.m_imap.k + 2);
    const uint32_t majstep = (A_n.m_imap.n + 2) * minstep;
    shm[shmem_imap(tx, ty, tz)] = A_n(i, j, k);
    __syncthreads();
    
    //A_nm1[gr_imap_thread] = -A_nm1[gr_imap_thread] + a1 * A_n[gr_imap_thread]
    //                            + a2 * (A_n    [gr_imap_thread + majstep] + A_n[gr_imap_thread - majstep])
    //                            + a4 * (A_n    [gr_imap_thread + minstep] + A_n[gr_imap_thread - minstep])
    //                            + a6 * (A_n    [gr_imap_thread + 1      ] + A_n[gr_imap_thread - 1      ]);
    
    A_nm1(i, j, k) = -A_nm1(i, j, k) + a1 * A_n(i, j, k)
                                        + a2 * (A_n(i + 1, j, k) + A_n(i - 1, j, k))
                                        + a4 * (A_n(i, j + 1, k) + A_n(i, j - 1, k))
                                        + a6 * (A_n(i, j, k + 1) + A_n(i, j, k - 1));
}
template <typename T>
__global__ void stencil_iter(grid<T> A_nm1, grid<T> A_n) {
    constexpr T dt = 0.5;
    constexpr T dx = 1.0;
    constexpr T dy = 1.0;
    constexpr T dz = 1.0;

    constexpr T a1 = T(2) * (T(1) - sq(dt / dx) - sq(dt / dy) - sq(dt / dz));
    constexpr T a2 = sq(dt / dx);
    constexpr T a4 = sq(dt / dy);
    constexpr T a6 = sq(dt / dz);
    constexpr T a8 = sq(dt);    

    //const uint32_t tx = threadIdx.x;
    //const uint32_t ty = threadIdx.y;
    //const uint32_t tz = threadIdx.z;

    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t j = threadIdx.y + blockDim.y * blockIdx.y;
    const uint32_t k = threadIdx.z + blockDim.z * blockIdx.z;
    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t tz = threadIdx.z;
    __shared__ T preload[10 * 10 * 18];
    typename grid<T>::imap_type pr_imap(8, 8, 16);
    aray<axis_aligned_occlusion, 3> occ = boundary_occlusion_of(0, aray<unsigned, 3>{tx, ty, tz}, aray<unsigned, 3>{8, 8, 16});
    //__syncthreads();
    const uint32_t pr_imap_thread = pr_imap(tx, ty, tz);
    const uint32_t gr_imap_thread = A_n.imap(i, j, k);
    preload[pr_imap_thread] = A_n[gr_imap_thread];
    uint32_t pr_imap_thread_ghost = pr_imap_thread;
    uint32_t gr_imap_thread_ghost = gr_imap_thread;
    assert(occ[0] == 0 || occ[0] == 1 || occ[0] == -1);
    assert(occ[1] == 0 || occ[1] == 1 || occ[1] == -1);
    assert(occ[2] == 0 || occ[2] == 1 || occ[2] == -1);
    pr_imap_thread_ghost += occ[0] ? (occ[0] * 180) : 0;
    pr_imap_thread_ghost += occ[1] ? (occ[1] * 18 ) : 0;
    pr_imap_thread_ghost += occ[2] ? (occ[2] * 1  ) : 0;

    gr_imap_thread_ghost += occ[0] ? (occ[0] * A_n.m_imap.n * A_n.m_imap.k) : 0;
    gr_imap_thread_ghost += occ[1] ? (occ[1] * A_n.m_imap.k) : 0;
    gr_imap_thread_ghost += occ[2] ? (occ[2]) : 0;
    //if(tx == 0 && ty == 0 && tz == 0)
    //    primpf("%d vs %d\n", pr_imap_thread_ghost, pr_imap_thread);
    //if(pr_imap_thread_ghost >= 1800){
    //    primpf("Error at: %d, %d, %d, %d\n", tx, ty, tz, pr_imap_thread_ghost);
    //    fflush(stdout);
    //}
    assert(pr_imap_thread_ghost < 1800);
    if(pr_imap_thread_ghost == pr_imap_thread){
        assert(gr_imap_thread_ghost == gr_imap_thread);
    }
    else{
        preload[pr_imap_thread_ghost] = A_n[gr_imap_thread_ghost];
    }
    /*if (occ[0] == AT_MIN) {
        preload[pr_imap(tx - 1, ty, tz)] = A_n(i - 1, j, k);
    }
    if (occ[0] == AT_MAX) {
        preload[pr_imap(tx + 1, ty, tz)] = A_n(i + 1, j, k);
    }
    if (occ[1] == AT_MIN) {
        preload[pr_imap(tx, ty - 1, tz)] = A_n(i, j - 1, k);
    }
    if (occ[1] == AT_MAX) {
        preload[pr_imap(tx, ty + 1, tz)] = A_n(i, j + 1, k);
    }
    if (occ[2] == AT_MIN) {
        preload[pr_imap(tx, ty, tz - 1)] = A_n(i, j, k - 1);
    }
    if (occ[2] == AT_MAX) {
        preload[pr_imap(tx, ty, tz + 1)] = A_n(i, j, k + 1);
    }*/
    __syncthreads();
    //__shared__ T preload[10 * 10 * 18];
    //typename grid<T>::imap_type pr_imap(8, 8, 16);
    //aray<axis_aligned_occlusion, 3> occ = boundary_occlusion_of(0, aray<unsigned, 3>{tx, ty, tz}, aray<unsigned, 3>{8, 8, 16});
    if(true){
        A_nm1(i, j, k) = -A_nm1(i, j, k) + a1 * A_n(i, j, k)
                                        + a2 * (A_n(i + 1, j, k) + A_n(i - 1, j, k))
                                        + a4 * (A_n(i, j + 1, k) + A_n(i, j - 1, k))
                                        + a6 * (A_n(i, j, k + 1) + A_n(i, j, k - 1));
        
    }
    if(false){
        //A_nm1(i, j, k) = -A_nm1(i, j, k) + a1 * preload[pr_imap(tx, ty, tz)]
        //                                + a2 * (preload[pr_imap(tx + 1, ty, tz)] + preload[pr_imap(ty - 1, ty, tz)])
        //                                + a4 * (preload[pr_imap(tx, ty + 1, tz)] + preload[pr_imap(ty, ty - 1,  tz)])
        //                                + a6 * (preload[pr_imap(tx, ty, tz + 1)] + preload[pr_imap(ty, ty, tz - 1)]);

        assert(pr_imap_thread >= 180);
        assert(pr_imap_thread + 180 < 10 * 10 * 18);
        assert(preload[pr_imap_thread] == A_n(i, j, k));
        A_nm1(i, j, k) = -A_nm1(i, j, k) + a1 * preload[pr_imap_thread]
                                        + a2 * (preload[pr_imap_thread + 180] + preload[pr_imap_thread - 180])
                                        + a4 * (preload[pr_imap_thread + 18 ] + preload[pr_imap_thread - 18 ])
                                        + a6 * (preload[pr_imap_thread + 1  ] + preload[pr_imap_thread - 1  ]);
        
    }
    // printf("%d\n", x);
}
template<typename T, typename callable>
__global__ void lambda_caller(grid<T> g, callable&& c){
    const uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    const uint32_t z = threadIdx.z + blockDim.z * blockIdx.z;
    c(g, x, y, z);

}
template <typename T>
void initial_cond(grid<T> *src, auto&& callable) {
    dim3 block_ext{(src->m_imap).m / 8, (src->m_imap).n / 8, (src->m_imap).k / 16};
    dim3 tg_extents{8, 8, 16};
    lambda_caller<<<block_ext, tg_extents>>>(*src, callable);
    //printf("update has finished\n");
}
template <typename T>
void update_inplace(grid<T> *src) {
    dim3 block_ext{(src->m_imap).m / 8, (src->m_imap).n / 8, (src->m_imap).k / 16};
    dim3 tg_extents{8, 8, 16};
    for (unsigned i = 0; i < 50; i++){
        stencil_iter_inplace<0><<<block_ext, tg_extents>>>(*src);
        stencil_iter_inplace<1><<<block_ext, tg_extents>>>(*src);
        //cudaDeviceSynchronize();
        //std::swap(src->data, dest->data);
    }
    //printf("update has finished\n");
}
template <typename T>
void update(grid<T> *src, grid<T> *dest) {
    dim3 block_ext{(src->m_imap).m / 8, (src->m_imap).n / 8, (src->m_imap).k / 16};
    dim3 tg_extents{8, 8, 16};
    for (unsigned i = 0; i < 100; i++){
        stencil_iter_trivial<<<block_ext, tg_extents>>>(*src, *dest);
        //cudaDeviceSynchronize();
        std::swap(src->data, dest->data);
    }
    //printf("update has finished\n");
}

KOKKOS_INLINE_FUNCTION float gauss(float x, float y, float z, float mean, float stddev){
    x -= mean;
    y -= mean;
    z -= mean;
    x *= x;
    y *= y;
    z *= z;
    #ifndef __CUDA_DEVICE__
    using std::exp;
    #endif
    return exp(-(x + y + z) / (stddev * stddev)); 
}
using u32 = std::uint32_t;
template<unsigned exp>
int iroot(int arg){
    return (int)std::sqrt((double)arg);
}
int main(int argc, char *argv[]) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    constexpr unsigned int ext = 512;
    constexpr unsigned int exth = ext / 2;
    grid<aray<float, 2>> gr(ext, ext, ext);
    initial_cond(&gr,[]KOKKOS_FUNCTION(grid<aray<float, 2>>& g, u32 i, u32 j, u32 k){
        float x = float(i) / (ext - 1);
        float y = float(j) / (ext - 1);
        float z = float(k) / (ext - 1);
        g(i,j,k)[0] = gauss(x, y, z, 0.5f, 0.1f);
        g(i,j,k)[1] = gauss(x, y, z, 0.5f, 0.1f);
    });
    update_inplace(&gr);
    grid<aray<float, 2>> grhc = gr.hostCopy();
    serial_for<3>([grhc](size_t i, size_t j, size_t k){
        printf("%f\n", grhc(i,j,k)[1]);
    }, {0, exth, exth}, {ext, exth+1, exth+1});
    fflush(stdout);
    return 0;
}
int main2(int argc, char *argv[]) {
    if(false){
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int cartext = iroot<3>(size);
        assert(cartext * cartext * cartext == size);
        printf("Rank: %d\n", rank);
        float hdata[5] = {1,2,3,4,5};
        float* data = cuda_malloc_helper<float>(5);
        if(rank == 0){
            cudaMemcpy(data, hdata, 5 * sizeof(float), cudaMemcpyHostToDevice);
            //cudaDeviceSynchronize();
            MPI_Send(data, 5, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
            //cudaDeviceSynchronize();
        }
        else{
            MPI_Recv(data, 5, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //cudaDeviceSynchronize();
            cudaMemcpy(hdata, data, 5 * sizeof(float), cudaMemcpyDeviceToHost);
            //cudaDeviceSynchronize();
            printf("%f, %f, %f, %f, %f\n", hdata[0], hdata[1], hdata[2], hdata[3], hdata[4]);
        }
        MPI_Finalize();
        return 0;
    }
    {
        ghosted_indexmap imap(8, 8, 16);
        //std::cout << imap.n << "\n";
        //std::cout << imap(7,7,15) + 180 + 18 + 1 << "\n";
        //return 0;
    }

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    //{
    //    grid<float> gr(32,32,32);
    //    initial_cond(&gr, []KOKKOS_FUNCTION(grid<float>& g, u32 i, u32 j, u32 k){
    //        assert(i < 32);
    //        assert(j < 32);
    //        assert(k < 32);
    //        printf("%u, %u, %u\n", i ,j ,k);
    //    });
    //    cudaDeviceSynchronize();
    //    return 0;
    //}
    //Kokkos::Array<uint32_t, 3> pol;
    // Kokkos::initialize(argc, argv);
    {
        constexpr uint32_t ext = 512;
        constexpr uint32_t exth = ext / 2;
        grid<float> gr(ext, ext, ext);
        
        grid<float> gr2(ext, ext, ext);
        initial_cond(&gr,[]KOKKOS_FUNCTION(grid<float>& g, u32 i, u32 j, u32 k){
            float x = float(i) / (ext - 1);
            float y = float(j) / (ext - 1);
            float z = float(k) / (ext - 1);
            g(i,j,k) = gauss(x, y, z, 0.5f, 0.1f);
        });
        initial_cond(&gr2,[]KOKKOS_FUNCTION(grid<float>& g, u32 i, u32 j, u32 k){
            float x = float(i) / (ext - 1);
            float y = float(j) / (ext - 1);
            float z = float(k) / (ext - 1);
            g(i,j,k) = gauss(x, y, z, 0.5f, 0.1f);
        });

        update(&gr, &gr2);
        grid<float> g2hc = gr2.hostCopy();
        serial_for<3>([g2hc](size_t i, size_t j, size_t k){
            printf("%f\n", g2hc(i,j,k));
        }, {0, exth, exth}, {ext, exth+1, exth+1});
        fflush(stdout);
        cudaDeviceSynchronize();
    }
    
    // Kokkos::finalize();
    return 0;
}