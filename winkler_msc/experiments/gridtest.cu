//#include <Kokkos_Core.hpp>
#include "grid.cu"
#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <list>
#include <map>
#include <cufft.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <xoshiro.hpp>
#include <curand.h>
#include <stb_image_write.h>
#include <cmath>
#include "BunchInit.hpp"
#include <eglutils.hpp>
#include "bunch.cu"



template<typename scalar>
KOKKOS_INLINE_FUNCTION ippl::Vector<scalar, 4> prepend_t(const ippl::Vector<scalar, 3>& x, scalar t = 0){
    return ippl::Vector<scalar, 4>{t, x[0], x[1], x[2]};
}
template<typename scalar>
KOKKOS_INLINE_FUNCTION ippl::Vector<scalar, 3> strip_t(const ippl::Vector<scalar, 4>& x){
    return ippl::Vector<scalar, 3>{x[1], x[2], x[3]};
}


template<typename scalar>
struct undulator_parameters{
    scalar lambda; //MITHRA: lambda_u
    scalar K; //Undulator parameter
    scalar length;
    scalar B_magnitude;
    undulator_parameters(scalar K_undulator_parameter, scalar lambda_u, scalar _length) : lambda(lambda_u), K(K_undulator_parameter), length(_length){
        B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K) / (electron_charge_in_unit_charges * lambda_u);
        //std::cout << "Setting bmag: " << B_magnitude << "\n";
    }
};
struct Xoshiro128 {
    uint64_t state[2];

    KOKKOS_INLINE_FUNCTION Xoshiro128(uint64_t seed) {
        state[0] = splitmix64(seed);
        state[1] = splitmix64(seed + 1);
    }

    KOKKOS_INLINE_FUNCTION uint64_t next64() {
        const uint64_t s0 = state[0];
        uint64_t s1 = state[1];
        const uint64_t result = s0 + s1;
        s1 ^= s0;
        state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
        state[1] = rotl(s1, 36); // c
        return result;
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION T next() {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_integral_v<T>);
        if constexpr(std::is_same_v<T, float>) {
            return next_float();
        }
        if constexpr(std::is_same_v<T, double>) {
            return next_double();
        }
        if constexpr(std::is_integral_v<T>) {
            return static_cast<T>(next64());
        }
        else {
            return T{};
        }
    }

    KOKKOS_INLINE_FUNCTION uint64_t operator()() {
        return next64();
    }

    KOKKOS_INLINE_FUNCTION double next_double() {
        #ifndef __CUDA_ARCH__
        using std::max;
        using std::min;
        #endif
        float ret = static_cast<float>(next64()) / static_cast<float>(UINT64_MAX);
        ret = max(min(ret, 1.f - 1e14f), 1e-14f);
        return ret;
    }

    KOKKOS_INLINE_FUNCTION float next_float() {
        #ifndef __CUDA_ARCH__
        using std::max;
        using std::min;
        #endif
        float ret = static_cast<float>(next64()) / static_cast<float>(UINT64_MAX);
        ret = max(min(ret, 1.f - 1e-7f), 1e-7f);
        return ret;
    }

    KOKKOS_INLINE_FUNCTION uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    KOKKOS_INLINE_FUNCTION uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    KOKKOS_INLINE_FUNCTION void scramble(uint64_t seed) {
        state[0] ^= splitmix64(seed);
        state[1] ^= splitmix64(seed + 1);
        // Additional mixing might be needed; consult Xoshiro128's original documentation.
    }
};
struct XORShift64 {
    uint64_t state;

    KOKKOS_INLINE_FUNCTION XORShift64(uint64_t seed) : state(seed) {}

    KOKKOS_INLINE_FUNCTION uint64_t next64() {
        assert(state != 0);
        state ^= (state << 13);
        state ^= (state >> 7);
        state ^= (state << 17);
        return state;
    }
    template<typename T>
    KOKKOS_INLINE_FUNCTION T next(){
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_integral_v<T>);
        if constexpr(std::is_same_v<T, float>){
            return next_float();
        }
        if constexpr(std::is_same_v<T, double>){
            return next_double();
        }
        if constexpr(std::is_integral_v<T>){
            return static_cast<T>(next64());
        }
        else{
            return T{};
        }
    }

    KOKKOS_INLINE_FUNCTION uint64_t operator()() {
        return next64();
    }

    KOKKOS_INLINE_FUNCTION double next_double() {
        #ifndef __CUDA_ARCH__
        using std::max;
        using std::min;
        #endif
        double ret = double(next64()) / double(UINT64_MAX);
        ret = max(min(ret, 1.0 - 1e-14), 1e-14);
        return ret;
    }

    KOKKOS_INLINE_FUNCTION float next_float() {
        #ifndef __CUDA_ARCH__
        using std::max;
        using std::min;
        #endif
        float ret = float(next64()) / float(UINT64_MAX);
        ret = max(min(ret, 1.f - 1e-7f), 1e-7f);
        return ret;
    }
    KOKKOS_INLINE_FUNCTION void scramble(uint64_t seed) {
        state ^= (seed + 0xC541F491AF6CAD1D);
        state = (state << 13) | (state >> (64 - 13));
        state *= 0x2545F4914F6CDD1D;
    }
};
__host__ __device__ double convertToNormal(double sample, double sigma) {
    double z = 1.4142135623730950488 * erfinv(2 * sample - 1);
    return z * sigma; // z has mean 0 and standard deviation 1, so scale by sigma
}
__host__ __device__ float convertToNormal(float sample, float sigma) {
    float z = 1.4142135623730950488f * erfinvf(2.0f * sample - 1.0f);
    return z * sigma; // z has mean 0 and standard deviation 1, so scale by sigma
}
template<typename scalar>
__device__ inline void scatterToGrid(grid<ippl::Vector<scalar, 4>, device>& g, const ippl::Vector<scalar, 3>& pos, const scalar value){ 
    auto [ipos, fracpos] = g.m_mesh.gridCoordinatesOf(pos);
    if(
        ipos[0] < 0
        ||ipos[1] < 0
        ||ipos[2] < 0
        ||ipos[0] >= g.m_imap.m - 1
        ||ipos[1] >= g.m_imap.n - 1
        ||ipos[2] >= g.m_imap.k - 1
        ||fracpos[0] < 0
        ||fracpos[1] < 0
        ||fracpos[2] < 0
    ){
        return;
    }
    //ippl::Vector<scalar, 3> relpos = pos - g.m_mesh.origin;
    //assert_isreal((relpos[0]));
    //assert_isreal((relpos[1]));
    //assert_isreal((relpos[2]));
    //ippl::Vector<scalar, 3> gridpos = relpos / g.m_mesh.spacing;
    //ippl::Vector<int, 3> ipos;
    //assert_isreal((gridpos[0]));
    //assert_isreal((gridpos[1]));
    //assert_isreal((gridpos[2]));
    //ippl::Vector<scalar, 3> fracpos = gridpos.decompose(&ipos);
    //ipos += ippl::Vector<int, 3>(g.nghost());
    ippl::Vector<scalar, 3> one_minus_fracpos = ippl::Vector<scalar, 3>(1) - fracpos;
    if(fracpos[0] < 0.0f){
        //printf("Fracpos: %.5e %.5e %.5e\n", fracpos[0], fracpos[1], fracpos[2]);
    }
    //if()
    assert(fracpos[0] >= 0.0f);
    assert(fracpos[0] <= 1.0f);
    assert(fracpos[1] >= 0.0f);
    assert(fracpos[1] <= 1.0f);
    assert(fracpos[2] >= 0.0f);
    assert(fracpos[2] <= 1.0f);
    assert(one_minus_fracpos[0] >= 0.0f);
    assert(one_minus_fracpos[0] <= 1.0f);
    assert(one_minus_fracpos[1] >= 0.0f);
    assert(one_minus_fracpos[1] <= 1.0f);
    assert(one_minus_fracpos[2] >= 0.0f);
    assert(one_minus_fracpos[2] <= 1.0f);
    scalar accum = 0;
    for(unsigned i = 0;i < 8;i++){
        scalar weight = 1;
        ippl::Vector<int, 3> ipos_l = ipos;
        for(unsigned d = 0;d < 3;d++){
            weight *= ((i & (1 << d)) ? fracpos[d] : one_minus_fracpos[d]);
            ipos_l[d] += !!(i & (1 << d));
        }
        assert_isreal(value);
        assert_isreal(weight);
        accum += weight;
        //atomicAdd((float4*)&(g(ipos_l[0], ipos_l[1], ipos_l[2])), value * weight);
        //printf("Schattering to: %d, %d, %d\n", ipos_l[0], ipos_l[1], ipos_l[2]);
        atomicAdd(&(g(ipos_l[0], ipos_l[1], ipos_l[2])[0]), value * weight);
        //printf("Depot: %d, %d, %d\n", ipos_l[0], ipos_l[1], ipos_l[2]);
        //printf("Source: %f\n", g(ipos_l[0], ipos_l[1], ipos_l[2])[1]);
    }
    assert(abs(accum - 1.0f) < 1e-6f);
}
template<typename scalar>
__device__ inline void scatterToGrid(grid<ippl::Vector<scalar, 4>, device>& g, const ippl::Vector<scalar, 3>& pos, const ippl::Vector<scalar, 3> value){ 
    auto [ipos, fracpos] = g.m_mesh.gridCoordinatesOf(pos);
    if(
        ipos[0] < 0
        ||ipos[1] < 0
        ||ipos[2] < 0
        ||ipos[0] >= g.m_imap.m - 1
        ||ipos[1] >= g.m_imap.n - 1
        ||ipos[2] >= g.m_imap.k - 1
        ||fracpos[0] < 0
        ||fracpos[1] < 0
        ||fracpos[2] < 0
    ){
        return;
    }
    //ippl::Vector<scalar, 3> relpos = pos - g.m_mesh.origin;
    //ippl::Vector<scalar, 3> gridpos = relpos / g.m_mesh.spacing;
    //ippl::Vector<int, 3> ipos;
    //ippl::Vector<scalar, 3> fracpos = gridpos.decompose(&ipos);
    //ipos += ippl::Vector<int, 3>(g.nghost());
    ippl::Vector<scalar, 3> one_minus_fracpos = ippl::Vector<scalar, 3>(1) - fracpos;
    scalar accum = 0;
    for(unsigned i = 0;i < 8;i++){
        scalar weight = 1;
        ippl::Vector<int, 3> ipos_l = ipos;
        for(unsigned d = 0;d < 3;d++){
            weight *= ((i & (1 << d)) ? fracpos[d] : one_minus_fracpos[d]);
            ipos_l[d] += !!(i & (1 << d));
        }

        for(unsigned k = 0;k < 3;k++){
            assert_isreal((value[k]));
            assert_isreal((weight));
            atomicAdd(&(g(ipos_l[0], ipos_l[1], ipos_l[2])[k + 1]), value[k] * weight);
        }
        accum += weight;
        //atomicAdd(&(g(ipos_l[0], ipos_l[1], ipos_l[2])[0]), value * weight);
        //printf("Depot: %d, %d, %d\n", ipos_l[0], ipos_l[1], ipos_l[2]);
        //printf("Source: %f\n", g(ipos_l[0], ipos_l[1], ipos_l[2])[1]);
    }
    assert(abs(accum - 1.0f) < 1e-6f);
}
template<typename scalar, bool assert_onecell = false>
__device__ inline void scatterLineToGrid(grid<ippl::Vector<scalar, 4>, device>& g, const ippl::Vector<scalar, 3>& from, const ippl::Vector<scalar, 3>& to, const scalar factor){ 

    
    pear<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> from_grid = g.m_mesh.gridCoordinatesOf(from);
    pear<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> to_grid   = g.m_mesh.gridCoordinatesOf(to  );
    //printf("Scatterdest: %.4e, %.4e, %.4e\n", from_grid.second[0], from_grid.second[1], from_grid.second[2]);
    for(int d = 0;d < 3;d++){
        assert(abs(from_grid.first[d] - to_grid.first[d]) <= 1);
    }
    //const uint32_t nghost = g.nghost();
    //from_ipos += ippl::Vector<int, 3>(nghost);
    //to_ipos += ippl::Vector<int, 3>(nghost);
    if(from_grid.first == to_grid.first){
        scatterToGrid(g, (from + to) * scalar(0.5), (to - from) * factor);
        //printf("Not ziggin\n");
        return;
    }
    assert(!assert_onecell);
    if constexpr(!assert_onecell){
        ippl::Vector<scalar, 3> relay;
        const int nghost = 1;
        const ippl::Vector<scalar, 3> hr   = g.m_mesh.spacing;
        const ippl::Vector<scalar, 3> orig = g.m_mesh.origin;
        for (unsigned int i = 0; i < 3; i++) {
            relay[i] = min(min(from_grid.first[i] - nghost, to_grid.first[i] - nghost) * hr[i] + scalar(1.0) * hr[i] + orig[i],
                           max(max(from_grid.first[i] - nghost, to_grid.first[i] - nghost) * hr[i] + scalar(0.0) * hr[i] + orig[i],
                               scalar(0.5) * (to[i] + from[i])));
        }
        //ippl::Vector<scalar, 3> first_seg = relay - from;
        //ippl::Vector<scalar, 3> second_seg = to - relay;
        //if(signbit(relay[2] - from[2]) != signbit(to[2] - relay[2])){
        //    printf("Mismatch: %f, %f, %f\n", from[2], relay[2], to[2]);
        //}
        //assert(signbit(relay[0] - from[0]) == signbit(to[0] - relay[0]) || from[0] == to[0]);
        //assert(signbit(relay[1] - from[1]) == signbit(to[1] - relay[1]) || from[1] == to[1]);
        //assert(signbit(relay[2] - from[2]) == signbit(to[2] - relay[2]) || from[2] == to[2]);
        scatterToGrid(g, (from + relay) * scalar(0.5), (relay - from) * factor);
        scatterToGrid(g, (relay + to) * scalar(0.5)  , (to - relay) * factor);
    }
}
template<typename value_type, typename scalar>
value_type gatherHost(grid<value_type, host> g, const ippl::Vector<scalar, 3>& pos){
    using vec3 = ippl::Vector<scalar, 3>;

    pear<ippl::Vector<int, 3>,ippl::Vector<scalar, 3>> gridpos = g.m_mesh.gridCoordinatesOf(pos);
    
    ippl::Vector<scalar, 3> fracpos = gridpos.second;
    ippl::Vector<scalar, 3> one_minus_fracpos = ippl::Vector<scalar, 3>(scalar(1)) - fracpos;
    
    ippl::Vector<int, 3> ipos = gridpos.first;
    size_t index = g.m_imap(ipos[0], ipos[1], ipos[2]);
    value_type accum(0);
    //#pragma unroll 8
    auto func = []<size_t Idx>(const vec3& whi, const vec3& wlo){
        scalar weight = 1;
        //ippl::Vector<int, 3> ipos_l = ipos;
        //#pragma unroll 3
        
        weight *= ((!!(Idx & (1 << 0))) * whi[0] + (!(Idx & (1 << 0))) * wlo[0]);
        weight *= ((!!(Idx & (1 << 1))) * whi[1] + (!(Idx & (1 << 1))) * wlo[1]);
        weight *= ((!!(Idx & (1 << 2))) * whi[2] + (!(Idx & (1 << 2))) * wlo[2]);
        //weight *= ((Idx & (1 << 0)) ? whi[0] : wlo[0]);
        //weight *= ((Idx & (1 << 1)) ? whi[1] : wlo[1]);
        //weight *= ((Idx & (1 << 2)) ? whi[2] : wlo[2]);
        
        return weight;

    };
    auto caller = [func]<size_t... Idx>(const std::index_sequence<Idx...>& seq,const vec3& whi, const vec3& wlo, grid<value_type, host> g, int i, int j, int k){
        //return func.template operator()<3>(whi, wlo);
        scalar accweight = (func.template operator()<Idx>(whi, wlo) + ...);
        assert(abs(accweight - 1) < 1e-6);
        return ((g(i + (Idx & 1),j + !!(Idx & 2), k + !!(Idx & 4)) * func.template operator()<Idx>(whi, wlo)) + ...);
    };
    return caller(std::make_index_sequence<8>{},fracpos, one_minus_fracpos, g, ipos[0], ipos[1], ipos[2]); 

}
template<typename value_type, typename scalar>
__device__ value_type gather(grid<value_type, device> g, const ippl::Vector<scalar, 3>& pos){
    using vec3 = ippl::Vector<scalar, 3>;

    pear<ippl::Vector<int, 3>,ippl::Vector<scalar, 3>> gridpos = g.m_mesh.gridCoordinatesOf(pos);
    
    ippl::Vector<scalar, 3> fracpos = gridpos.second;
    ippl::Vector<scalar, 3> one_minus_fracpos = ippl::Vector<scalar, 3>(scalar(1)) - fracpos;
    
    ippl::Vector<int, 3> ipos = gridpos.first;

    if(
        ipos[0] < 0
        ||ipos[1] < 0
        ||ipos[2] < 0
        ||ipos[0] >= g.m_imap.m - 1
        ||ipos[1] >= g.m_imap.n - 1
        ||ipos[2] >= g.m_imap.k - 1
        ||fracpos[0] < 0
        ||fracpos[1] < 0
        ||fracpos[2] < 0
    ){
        return value_type(0);
    }
    size_t index = g.m_imap(ipos[0], ipos[1], ipos[2]);
    value_type accum(0);
    //#pragma unroll 8
    auto func = []__device__<size_t Idx>(const vec3& whi, const vec3& wlo){
        scalar weight = 1;
        //ippl::Vector<int, 3> ipos_l = ipos;
        //#pragma unroll 3
        
        weight *= ((!!(Idx & (1 << 0))) * whi[0] + (!(Idx & (1 << 0))) * wlo[0]);
        weight *= ((!!(Idx & (1 << 1))) * whi[1] + (!(Idx & (1 << 1))) * wlo[1]);
        weight *= ((!!(Idx & (1 << 2))) * whi[2] + (!(Idx & (1 << 2))) * wlo[2]);
        //weight *= ((Idx & (1 << 0)) ? whi[0] : wlo[0]);
        //weight *= ((Idx & (1 << 1)) ? whi[1] : wlo[1]);
        //weight *= ((Idx & (1 << 2)) ? whi[2] : wlo[2]);
        
        return weight;

    };
    auto caller = [func]__device__<size_t... Idx>(const std::index_sequence<Idx...>& seq,const vec3& whi, const vec3& wlo, grid<value_type, device> g, int i, int j, int k){
        //return func.template operator()<3>(whi, wlo);
        scalar accweight = (func.template operator()<Idx>(whi, wlo) + ...);
        assert(abs(accweight - 1) < 1e-6);
        return ((g(i + (Idx & 1),j + !!(Idx & 2), k + !!(Idx & 4)) * func.template operator()<Idx>(whi, wlo)) + ...);
    };
    return caller(std::make_index_sequence<8>{},fracpos, one_minus_fracpos, g, ipos[0], ipos[1], ipos[2]); 

}
__host__ __device__ void printVec(const ippl::Vector<float, 3>& x){
    printf("V: %.4e, %.4e, %.4e", x[0], x[1], x[2]);
}
__host__ __device__ void printVec(const ippl::Vector<double, 3>& x){
    printf("V: %.4e, %.4e, %.4e", x[0], x[1], x[2]);
}

uint64_t nanoTime(){
    using namespace std;
    using namespace chrono;
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

//constexpr uint32_t extent = 400;
//constexpr uint32_t extenth = extent / 2;
template<typename scalar1, typename... scalar>
    requires((std::is_floating_point_v<scalar1>))
KOKKOS_INLINE_FUNCTION float gauss(scalar1 mean, scalar1 stddev, scalar... x){
    uint32_t dim = sizeof...(scalar);
    ippl::Vector<scalar1, sizeof...(scalar)> vec{scalar1(x - mean)...};
    for(unsigned d = 0;d < dim;d++){
        vec[d] = vec[d] * vec[d];
    }
    #ifndef __CUDA_ARCH__
    using std::exp;
    #endif
    return exp(-(vec.sum()) / (stddev * stddev)); 
}
auto constexpr sq(auto x){
    return x * x;
}

/**
 * @brief Evaluates E and B fields
 * 
 * @tparam even True after an even number of timesteps, false after an uneven number
 * @tparam scalar 
 * @param A 
 * @param EB 
 * @param dt 
 */
template<bool even, typename scalar>
void evaluate_EB(const grid<ippl::Vector<ippl::Vector<scalar, 4>, 2>, device> A, grid<ippl::Vector<ippl::Vector<scalar, 3>, 2>, device> EB, scalar dt){
    ippl::Vector<scalar, 3> inverse_2_spacing = ippl::Vector<scalar, 3>(0.5) / A.m_mesh.spacing;
    //std::cerr << "Eval_EB hostside called: " << A.getRange().begin  << " to " << A.getRange().end << "\n";
    const scalar idt = scalar(1.0) / dt;
    parallel_for_cuda([EB, A, inverse_2_spacing, idt]KOKKOS_FUNCTION(size_t i, size_t j, size_t k)mutable{
        //printf("Evaluate EB kernel called\n");
        for(int d = 0;d < 4;d++){
            assert_isreal(A(i,j,k)[0][d]);
            assert_isreal(A(i,j,k)[1][d]);
        }
        ippl::Vector<scalar, 3> dAdt = (A(i, j, k)[even].template tail<3>() - A(i, j, k)[!even].template tail<3>()) * idt;
        ippl::Vector<scalar, 4> dAdx = (A(i + 1, j, k)[even] - A(i - 1, j, k)[even]) * inverse_2_spacing[0];
        ippl::Vector<scalar, 4> dAdy = (A(i, j + 1, k)[even] - A(i, j - 1, k)[even]) * inverse_2_spacing[1];
        ippl::Vector<scalar, 4> dAdz = (A(i, j, k + 1)[even] - A(i, j, k - 1)[even]) * inverse_2_spacing[2];

        ippl::Vector<scalar, 3> grad_phi{
            dAdx[0], dAdy[0], dAdz[0]
        };
        ippl::Vector<scalar, 3> curlA{
            dAdy[3] - dAdz[2],
            dAdz[1] - dAdx[3],
            dAdx[2] - dAdy[1],
        };
        if(A(i,j,k).squaredNorm() > 0)
            printf("%f\n", A(i,j,k).squaredNorm());
        //if(grad_phi.squaredNorm() > 0){
        //    printf("Kgrad_phi: %f\n", grad_phi.squaredNorm());
        //}
        //if(curlA.squaredNorm() > 0){
        //    printf("Kurla: %f\n", curlA.squaredNorm());
        //}
        EB(i,j,k)[0] = -dAdt - grad_phi;
        EB(i,j,k)[1] = curlA;
        for(int d = 0;d < 3;d++){
            assert_isreal(EB(i,j,k)[0][d]);
            assert_isreal(EB(i,j,k)[1][d]);
        }
    }, A.getRange());
    #ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Error in evaluate_EB: " << cudaGetErrorString(err) << "\n";
    }
    #endif
    //std::cout << "Successfully evaluated EB" << std::endl;
}
template<typename value_type, space sp>
struct field_state{
    grid<value_type, sp> A_np1;
    grid<value_type, sp> A_n;
    grid<value_type, sp> A_nm1;
    grid<value_type, sp> source_n;
    void destroy(){
        A_np1.destroy();
        A_n.destroy();
        A_nm1.destroy();
        source_n.destroy();
    }
};
template<typename scalar>
void evaluate_EB(const field_state<ippl::Vector<scalar, 4>, device>& F, grid<ippl::Vector<ippl::Vector<scalar, 3>, 2>, device> EB, scalar dt){
    ippl::Vector<scalar, 3> inverse_2_spacing = ippl::Vector<scalar, 3>(0.5) / F.A_n.m_mesh.spacing;
    //std::cerr << "Eval_EB hostside called: " << A.getRange().begin  << " to " << A.getRange().end << "\n";
    const scalar idt = scalar(1.0) / dt;
    auto A_n = F.A_n;
    auto A_np1 = F.A_np1;
    auto A_nm1 = F.A_nm1;
    parallel_for_cuda([A_n, A_np1, A_nm1, F, EB, inverse_2_spacing, idt]KOKKOS_FUNCTION(size_t i, size_t j, size_t k)mutable{
        //printf("Evaluate EB kernel called\n");
        for(int d = 0;d < 4;d++){
            //assert_isreal(A(i,j,k)[0][d]);
            //assert_isreal(A(i,j,k)[1][d]);
        }
        ippl::Vector<scalar, 3> dAdt = (A_n(i, j, k).template tail<3>() - A_nm1(i, j, k).template tail<3>()) * idt;
        ippl::Vector<scalar, 4> dAdx = (A_n(i + 1, j, k) - A_n(i - 1, j, k)) * inverse_2_spacing[0];
        ippl::Vector<scalar, 4> dAdy = (A_n(i, j + 1, k) - A_n(i, j - 1, k)) * inverse_2_spacing[1];
        ippl::Vector<scalar, 4> dAdz = (A_n(i, j, k + 1) - A_n(i, j, k - 1)) * inverse_2_spacing[2];

        ippl::Vector<scalar, 3> grad_phi{
            dAdx[0], dAdy[0], dAdz[0]
        };
        ippl::Vector<scalar, 3> curlA{
            dAdy[3] - dAdz[2],
            dAdz[1] - dAdx[3],
            dAdx[2] - dAdy[1],
        };
        //if(grad_phi.squaredNorm() > 0){
        //    printf("Kgrad_phi: %f\n", grad_phi.squaredNorm());
        //}
        //if(curlA.squaredNorm() > 0){
        //    printf("Kurla: %.5e\n", curlA.squaredNorm());
        //}
        EB(i,j,k)[0] = -dAdt - grad_phi;
        EB(i,j,k)[1] = curlA;
        for(int d = 0;d < 3;d++){
            assert_isreal(EB(i,j,k)[0][d]);
            assert_isreal(EB(i,j,k)[1][d]);
        }
    }, A_n.getRange());
    #ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Error in evaluate_EB: " << cudaGetErrorString(err) << "\n";
    }
    #endif
    //std::cout << "Successfully evaluated EB" << std::endl;
}
template <typename View, typename Coords, unsigned int axis, size_t... Idx>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply_impl_with_offset(
    const View& view, const Coords& coords, int offset, const std::index_sequence<Idx...>&) {
    return view((coords[Idx] + offset * !!(Idx == axis))...);
}
template <typename View, typename Coords, unsigned axis>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply_with_offset(const View& view,
                                                                  const Coords& coords,
                                                                  int offset) {
    using Indices = std::make_index_sequence<Coords::dim>;
    return apply_impl_with_offset<View, Coords, axis>(view, coords, offset, Indices{});
}
template <typename _scalar, unsigned _main_axis, unsigned... _side_axes>
struct first_order_abc {
    using scalar                          = _scalar;
    constexpr static unsigned main_axis   = _main_axis;
    
    ippl::Vector<scalar, 3> hr_m;
    int sign;
    scalar beta0;
    scalar beta1;
    scalar beta2;
    scalar beta3;
    scalar beta4;
    first_order_abc() = default;
    KOKKOS_FUNCTION first_order_abc(ippl::Vector<scalar, 3> hr, scalar c, scalar dt, int _sign)
        : hr_m(hr)
        , sign(_sign) {
        beta0 = (c * dt - hr_m[main_axis]) / (c * dt + hr_m[main_axis]);
        beta1 = 2.0 * hr_m[main_axis] / (c * dt + hr_m[main_axis]);
        beta2 = -1.0;
        beta3 = beta1;
        beta4 = beta0;
    }
    template <typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& a_n, const view_type& a_nm1,
                                           const view_type& a_np1, const Coords& c) const ->
        typename view_type::value_type {
        constexpr static unsigned side_axes[] = {_side_axes...};
        using value_t = typename view_type::value_type;

        value_t ret = beta0
                          * (apply_with_offset<view_type, Coords, ~0u>(a_nm1, c, sign)
                             + apply_with_offset<view_type, Coords, main_axis>(a_np1, c, sign))
                      + beta1
                            * (apply_with_offset<view_type, Coords, ~0u>(a_n, c, sign)
                               + apply_with_offset<view_type, Coords, main_axis>(a_n, c, sign))
                      + beta2 * (apply_with_offset<view_type, Coords, main_axis>(a_nm1, c, sign));
        return ret;
    }
};
template<unsigned a, unsigned b>
constexpr auto first(){
    return a;
}
template<unsigned a, unsigned b>
constexpr auto second(){
    return b;
}
template<typename _scalar, unsigned _main_axis, unsigned... _side_axes>
struct second_order_abc_face{
    using scalar = _scalar;
    scalar Cweights[5];
    int sign;
    constexpr static unsigned main_axis = _main_axis;
    KOKKOS_FUNCTION second_order_abc_face(ippl::Vector<scalar, 3> hr, scalar dt, int _sign) : sign(_sign){
        constexpr scalar c = 1;
        constexpr unsigned side_axes[2] = {_side_axes...};
        static_assert(
            (main_axis == 0 && first<_side_axes...>() == 1 && second<_side_axes...>() == 2) ||
            (main_axis == 1 && first<_side_axes...>() == 0 && second<_side_axes...>() == 2) ||
            (main_axis == 2 && first<_side_axes...>() == 0 && second<_side_axes...>() == 1)
        );
        assert(_main_axis != side_axes[0]);
        assert(_main_axis != side_axes[1]);
        assert(side_axes[0] != side_axes[1]);
        constexpr scalar truncation_order = 2.0;
        scalar p      = ( 1.0 + 1 * 1 ) / ( 1 + 1 );
        scalar q      = - 1.0 / ( 1 + 1 );

        scalar d  	 = 1.0 / ( 2.0 * dt * hr[main_axis]) + p / ( 2.0 * c * dt * dt);

        Cweights[0]	= (   1.0 / ( 2.0 * dt * hr[main_axis] ) - p / (2.0 * c * dt * dt)) / d;
        Cweights[1]	= ( - 1.0 / ( 2.0 * dt * hr[main_axis] ) - p / (2.0 * c * dt * dt)) / d;
        assert(abs(Cweights[1] + 1) < 1e-6); //Like literally
        Cweights[2]  	= (   p / ( c * dt * dt ) + q * (truncation_order - 1.0) * (c / (hr[side_axes[0]] * hr[side_axes[0]]) + c / (hr[side_axes[1]] * hr[side_axes[1]]))) / d;
        Cweights[3]  	= -q * (truncation_order - 1.0) * ( c / ( 2.0 * hr[side_axes[0]] * hr[side_axes[0]] ) ) / d;
        Cweights[4]  	= -q * (truncation_order - 1.0) * ( c / ( 2.0 * hr[side_axes[1]] * hr[side_axes[1]] ) ) / d;
    }
    template<typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& A_n, const view_type& A_nm1,const view_type& A_np1, const Coords& c)const -> typename view_type::value_type{
        uint32_t i = c[0];
        uint32_t j = c[1];
        uint32_t k = c[2];
        constexpr unsigned side_axes[2] = {_side_axes...};
        ippl::Vector<uint32_t, 3> side_axis1_onehot = ippl::Vector<uint32_t, 3>{side_axes[0] == 0, side_axes[0] == 1, side_axes[0] == 2};
        ippl::Vector<uint32_t, 3> side_axis2_onehot = ippl::Vector<uint32_t, 3>{side_axes[1] == 0, side_axes[1] == 1, side_axes[1] == 2};
        ippl::Vector<uint32_t, 3> mainaxis_off = ippl::Vector<int32_t, 3>{(main_axis == 0) * sign, (main_axis == 1) * sign, (main_axis == 2) * sign}.cast<uint32_t>();
        return advanceBoundaryS(
		    A_nm1(i,j,k), A_n(i,j,k),
		    apply(A_nm1, c + mainaxis_off), apply(A_n, c + mainaxis_off), apply(A_np1, c + mainaxis_off),
		    apply(A_n, c + side_axis1_onehot + mainaxis_off), apply(A_n, c - side_axis1_onehot + mainaxis_off), apply(A_n, c + side_axis2_onehot + mainaxis_off),
		    apply(A_n, c - side_axis2_onehot + mainaxis_off), apply(A_n, c + side_axis1_onehot),                apply(A_n, c - side_axis1_onehot),
		    apply(A_n, c + side_axis2_onehot), apply(A_n, c - side_axis2_onehot)
        );
    }
    template<typename value_type>
    KOKKOS_FUNCTION value_type advanceBoundaryS (const value_type& v1 , const value_type& v2 ,
						 const value_type& v3 , const value_type& v4 , const value_type& v5 ,
						 const value_type& v6 , const value_type& v7 , const value_type& v8 ,
						 const value_type& v9 , const value_type& v10, const value_type& v11,
						 const value_type& v12, const value_type& v13)const noexcept
    {
        
      value_type v0 =
    	 Cweights[0]  * (v1 + v5) +
    	(Cweights[1]) * v3 +
    	(Cweights[2]) * ( v2 + v4 ) +
    	(Cweights[3]) * ( v6 + v7 + v10 + v11 ) +
    	(Cweights[4]) * ( v8 + v9 + v12 + v13 );
      return v0;
    }
};
template<typename _scalar, unsigned _main_axis, unsigned... _side_axes>
struct second_order_abc{
    using scalar = _scalar;
    constexpr static unsigned main_axis = _main_axis;
    //constexpr static ippl::Vector<unsigned, 2> side_axes{_side_axes...};
    ippl::Vector<scalar, 3> hr_m;
    int sign;
    scalar gamma0;
    scalar gamma1;
    scalar gamma2;
    scalar gamma3;
    scalar gamma4;
    scalar gamma5;
    scalar gamma6;
    scalar gamma7;
    scalar gamma8;
    scalar gamma9;
    scalar gamma10;
    scalar gamma11;
    scalar gamma12;
    second_order_abc() = default;

    KOKKOS_FUNCTION second_order_abc(ippl::Vector<scalar, 3> hr, scalar dt, int _sign) : hr_m(hr), sign(_sign){
        constexpr scalar c = 1;
        ippl::Vector<unsigned, 2> side_axes{_side_axes...};
        gamma0 = (c * dt - hr[main_axis]) / (c * dt + hr[main_axis]); 
        gamma1 = (hr[main_axis] * (2.0 - sq(c * dt / hr_m[side_axes[1]]) - sq(c * dt / hr_m[side_axes[0]])))
                   / (c * dt + hr_m[main_axis]);
        gamma2 = -1.0;
        gamma3 = gamma1;
        gamma4 = gamma0;
        gamma5 = sq(c * dt / hr_m[side_axes[0]]) * hr_m[main_axis] / (2.0 * (c * dt + hr_m[main_axis]));
        gamma6 = gamma5;
        gamma7 = sq(c * dt / hr_m[side_axes[1]]) * hr_m[main_axis] / (2.0 * (c * dt + hr_m[main_axis]));
        gamma8 = gamma7;
        gamma9 = gamma6;
        gamma10 = gamma9;
        gamma11 = gamma8;
        gamma12 = gamma8;
    }
    template<typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& a_n, const view_type& a_nm1,const view_type& a_np1, const Coords& c)const -> typename view_type::value_type{
        ippl::Vector<unsigned, 2> side_axes{_side_axes...};
        using value_t = typename view_type::value_type;
        value_t ret(0.0);
        ret += apply(a_nm1, c) * gamma0 + apply(a_n, c) * gamma1;
        {
            Coords acc = c;
            acc[main_axis] += sign;
            ret += gamma2 * apply(a_nm1, acc)
             + gamma3 * apply(a_n, acc)
             + gamma4 * apply(a_np1, acc);
        }
        {
            Coords acc = c;
            acc[main_axis] += sign;
            acc[side_axes[0]] += 1;
            ret += gamma5 * apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[main_axis] += sign;
            acc[side_axes[0]] -= 1;
            ret += gamma6 * apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[main_axis] += sign;
            acc[side_axes[1]] += 1;
            ret += gamma7 * apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[main_axis] += sign;
            acc[side_axes[1]] -= 1;
            ret += gamma8 * apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[side_axes[0]] += 1;
            ret += gamma9 * apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[side_axes[0]] -= 1;
            ret += gamma10 * apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[side_axes[1]] += 1;
            ret += gamma11 * apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[side_axes[1]] -= 1;
            ret += gamma12 * apply(a_n, acc);
        }

        return ret;
    }
};
template<typename _scalar, bool x0, bool y0, bool z0>
struct second_order_abc_corner{
    using scalar = _scalar;
    scalar Cweights[17];
    KOKKOS_FUNCTION second_order_abc_corner(ippl::Vector<scalar, 3> hr, scalar dt){
        constexpr scalar c0_ = scalar(1);
        Cweights[0]  =   ( - 1.0 / hr[0] - 1.0 / hr[1] - 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[1]  =   (   1.0 / hr[0] - 1.0 / hr[1] - 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[2]  =   ( - 1.0 / hr[0] + 1.0 / hr[1] - 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[3]  =   ( - 1.0 / hr[0] - 1.0 / hr[1] + 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[4]  =   (   1.0 / hr[0] + 1.0 / hr[1] - 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[5]  =   (   1.0 / hr[0] - 1.0 / hr[1] + 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[6]  =   ( - 1.0 / hr[0] + 1.0 / hr[1] + 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[7]  =   (   1.0 / hr[0] + 1.0 / hr[1] + 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[8]  = - ( - 1.0 / hr[0] - 1.0 / hr[1] - 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[9]  = - (   1.0 / hr[0] - 1.0 / hr[1] - 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[10] = - ( - 1.0 / hr[0] + 1.0 / hr[1] - 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[11] = - ( - 1.0 / hr[0] - 1.0 / hr[1] + 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[12] = - (   1.0 / hr[0] + 1.0 / hr[1] - 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[13] = - (   1.0 / hr[0] - 1.0 / hr[1] + 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[14] = - ( - 1.0 / hr[0] + 1.0 / hr[1] + 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[15] = - (   1.0 / hr[0] + 1.0 / hr[1] + 1.0 / hr[2] ) / (8.0 * dt) - 1.0 / ( 4.0 * c0_ * dt * dt);
        Cweights[16] = 1.0 / (2.0 * c0_ * dt * dt);
    }
    template<typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& A_n, const view_type& A_nm1,const view_type& A_np1, const Coords& c)const -> typename view_type::value_type{
        //First implementation: 0,0,0 corner
        constexpr uint32_t xoff = (x0) ? 1 : uint32_t(-1);
        constexpr uint32_t yoff = (y0) ? 1 : uint32_t(-1);
        constexpr uint32_t zoff = (z0) ? 1 : uint32_t(-1);
        constexpr ippl::Vector<uint32_t, 3> offsets[8] = {
            ippl::Vector<uint32_t, 3>{0,0,0},
            ippl::Vector<uint32_t, 3>{xoff,0,0},
            ippl::Vector<uint32_t, 3>{0,yoff,0},
            ippl::Vector<uint32_t, 3>{0,0,zoff},
            ippl::Vector<uint32_t, 3>{xoff,yoff,0},
            ippl::Vector<uint32_t, 3>{xoff,0,zoff},
            ippl::Vector<uint32_t, 3>{0,yoff,zoff},
            ippl::Vector<uint32_t, 3>{xoff,yoff,zoff},
        };
        return advanceCornerS(
                                          apply(A_n, c), apply(A_nm1, c),
            apply(A_np1, c + offsets[1]), apply(A_n, c + offsets[1]), apply(A_nm1, c + offsets[1]),
            apply(A_np1, c + offsets[2]), apply(A_n, c + offsets[2]), apply(A_nm1, c + offsets[2]),
            apply(A_np1, c + offsets[3]), apply(A_n, c + offsets[3]), apply(A_nm1, c + offsets[3]),
            apply(A_np1, c + offsets[4]), apply(A_n, c + offsets[4]), apply(A_nm1, c + offsets[4]),
            apply(A_np1, c + offsets[5]), apply(A_n, c + offsets[5]), apply(A_nm1, c + offsets[5]),
            apply(A_np1, c + offsets[6]), apply(A_n, c + offsets[6]), apply(A_nm1, c + offsets[6]),
            apply(A_np1, c + offsets[7]), apply(A_n, c + offsets[7]), apply(A_nm1, c + offsets[7])
        );
    }
    template<typename value_type>
    KOKKOS_INLINE_FUNCTION value_type advanceCornerS         
                            (       value_type v1 , value_type v2 ,
                             value_type v3 , value_type v4 , value_type v5 ,
                             value_type v6 , value_type v7 , value_type v8 ,
                             value_type v9 , value_type v10, value_type v11,
                             value_type v12, value_type v13, value_type v14,
                             value_type v15, value_type v16, value_type v17,
                             value_type v18, value_type v19, value_type v20,
                             value_type v21, value_type v22, value_type v23)const noexcept{
    return      - ( v1  * (Cweights[16]) + v2  * (Cweights[8]) +
    v3  * Cweights[1] + v4  * Cweights[16] + v5  * Cweights[9] +
    v6  * Cweights[2] + v7  * Cweights[16] + v8  * Cweights[10] +
    v9  * Cweights[3] + v10 * Cweights[16] + v11 * Cweights[11] +
    v12 * Cweights[4] + v13 * Cweights[16] + v14 * Cweights[12] +
    v15 * Cweights[5] + v16 * Cweights[16] + v17 * Cweights[13] +
    v18 * Cweights[6] + v19 * Cweights[16] + v20 * Cweights[14] +
    v21 * Cweights[7] + v22 * Cweights[16] + v23 * Cweights[15]) / Cweights[0];
  }
};
/**
 * @brief For example, the y-aligned edge at z=0,x=0 would be second_order_abc_edge<double, 1, 0, 2>
 * 
 * @tparam _scalar 
 * @tparam edge_axis 
 * @tparam normal_axis1 
 * @tparam normal_axis2 
 */
template<typename _scalar, unsigned edge_axis, unsigned normal_axis1, unsigned normal_axis2, bool na1_zero, bool na2_zero>
struct second_order_abc_edge{
    using scalar = _scalar;
    //
    scalar Eweights[5];
    
    KOKKOS_FUNCTION second_order_abc_edge(ippl::Vector<scalar, 3> hr, scalar dt){
        static_assert(normal_axis1 != normal_axis2);
        static_assert(edge_axis != normal_axis2);
        static_assert(edge_axis != normal_axis1);
        static_assert((edge_axis == 2 && normal_axis1 == 0 && normal_axis2 == 1) || (edge_axis == 0 && normal_axis1 == 1 && normal_axis2 == 2) || (edge_axis == 1 && normal_axis1 == 2 && normal_axis2 == 0));
        constexpr scalar c0_ = scalar(1);
        scalar d    =    ( 1.0 / hr[normal_axis1] + 1.0 / hr[normal_axis2] ) / ( 4.0 * dt ) + 3.0 / ( 8.0 * c0_ * dt * dt );
        if constexpr(normal_axis1 == 0 && normal_axis2 == 1){ // xy edge (along z)
            Eweights[0] = ( - ( 1.0 / hr[normal_axis2] - 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[1] = (   ( 1.0 / hr[normal_axis2] - 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[2] = (   ( 1.0 / hr[normal_axis2] + 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[3] = ( 3.0 / ( 4.0 * c0_ * dt * dt ) - c0_ / (4.0 * hr[edge_axis] * hr[edge_axis])) / d;
            Eweights[4] = c0_ / ( 8.0 * hr[edge_axis] * hr[edge_axis] ) / d;
        }
        else if constexpr(normal_axis1 == 2 && normal_axis2 == 0){ // zx edge (along y)
            Eweights[0] = ( - ( 1.0 / hr[normal_axis2] - 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[1] = (   ( 1.0 / hr[normal_axis2] - 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[2] = (   ( 1.0 / hr[normal_axis2] + 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[3] = ( 3.0 / ( 4.0 * c0_ * dt * dt ) - c0_ / (4.0 * hr[edge_axis] * hr[edge_axis])) / d;
            Eweights[4] = c0_ / ( 8.0 * hr[edge_axis] * hr[edge_axis] ) / d;
        }
        else if constexpr(normal_axis1 == 1 && normal_axis2 == 2){ // yz edge (along x)
            Eweights[0] = ( - ( 1.0 / hr[normal_axis2] - 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[1] = (   ( 1.0 / hr[normal_axis2] - 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[2] = (   ( 1.0 / hr[normal_axis2] + 1.0 / hr[normal_axis1] ) / ( 4.0 * dt ) - 3.0 / ( 8.0 * c0_ * dt * dt )) / d;
            Eweights[3] = ( 3.0 / ( 4.0 * c0_ * dt * dt ) - c0_ / (4.0 * hr[edge_axis] * hr[edge_axis])) / d;
            Eweights[4] = c0_ / ( 8.0 * hr[edge_axis] * hr[edge_axis] ) / d;
        }
        else{
            assert(false);
        }
        


        
    }
    template<typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& A_n, const view_type& A_nm1,const view_type& A_np1, const Coords& c)const -> typename view_type::value_type{
        uint32_t i = c[0];
        uint32_t j = c[1];
        uint32_t k = c[2];
        //constexpr unsigned nax[2] = {normal_axis1, normal_axis2};
        ippl::Vector<int32_t, 3> normal_axis1_onehot = ippl::Vector<int32_t, 3>{normal_axis1 == 0, normal_axis1 == 1, normal_axis1 == 2} * int32_t(na1_zero ? 1 : -1);
        ippl::Vector<int32_t, 3> normal_axis2_onehot = ippl::Vector<int32_t, 3>{normal_axis2 == 0, normal_axis2 == 1, normal_axis2 == 2} * int32_t(na2_zero ? 1 : -1);
        ippl::Vector<uint32_t, 3> acc0 = {i, j, k};
        ippl::Vector<uint32_t, 3> acc1 = acc0 + normal_axis1_onehot.cast<uint32_t>();
        ippl::Vector<uint32_t, 3> acc2 = acc0 + normal_axis2_onehot.cast<uint32_t>();
        ippl::Vector<uint32_t, 3> acc3 = acc0 + normal_axis1_onehot.cast<uint32_t>() + normal_axis2_onehot.cast<uint32_t>();
        //ippl::Vector<uint32_t, 3> axism = (-ippl::Vector<int, 3>{edge_axis == 0, edge_axis == 1, edge_axis == 2}).cast<uint32_t>();
        ippl::Vector<uint32_t, 3> axisp{edge_axis == 0, edge_axis == 1, edge_axis == 2};
        //return A_n(i, j, k);
        return advanceEdgeS(
                                        A_n(i, j, k),      A_nm1(i, j, k),
            apply(A_np1, acc1),   apply(A_n, acc1   ), apply(A_nm1, acc1),
            apply(A_np1, acc2),   apply(A_n, acc2   ), apply(A_nm1, acc2),
            apply(A_np1, acc3),   apply(A_n, acc3   ), apply(A_nm1, acc3),
            apply(A_n, acc0 - axisp), apply(A_n, acc1 - axisp), apply(A_n, acc2 - axisp), apply(A_n, acc3 - axisp),
            apply(A_n, acc0 + axisp), apply(A_n, acc1 + axisp), apply(A_n, acc2 + axisp), apply(A_n, acc3 + axisp)
        );
        //return advanceEdgeS(
        //                              A_n(i, j, k    ),       A_nm1(i, j, k),
        //    A_np1(i,  j, k + 1   ),   A_n(i, j, k + 1),       A_nm1(i, j, k + 1),
        //    A_np1(i + 1, j, k    ),   A_n(i + 1, j, k),       A_nm1(i + 1, j, k),
        //    A_np1(i + 1, j, k + 1),   A_n(i + 1,j,k + 1),     A_nm1(i + 1, j, k + 1),
        //    A_n  (i    , j - 1, k),   A_n(i, j - 1, k + 1),   A_n(i + 1, j - 1, k), 
        //    A_n  (i + 1, j - 1, k + 1), A_n(i    , j + 1, k), A_n(i, j + 1, k + 1), 
        //    A_n  (i + 1, j + 1, k),     A_n(i + 1, j + 1, k + 1));
        //    //xmin zmin --> y-aligned edge
    }
    template<typename value_type>
    KOKKOS_INLINE_FUNCTION value_type advanceEdgeS 		
            (              value_type v1 , value_type v2 ,
                           value_type v3 , value_type v4 , value_type v5 ,
                           value_type v6 , value_type v7 , value_type v8 ,
                           value_type v9 , value_type v10, value_type v11,
                           value_type v12, value_type v13, value_type v14,
                           value_type v15, value_type v16, value_type v17,
                           value_type v18, value_type v19)const noexcept{
    value_type v0 =
    Eweights[0] * (v3 + v8) +
    Eweights[1] * (v5 + v6) +
    Eweights[2] * (v2 + v9) +
    Eweights[3] * (v1 + v4 + v7 + v10) +
    Eweights[4] * (v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19) - v11;
    return v0;
  }
};
void tescht(){
    std::ofstream sbuckets("sbuckets.txt");
    using tscalar = float;
    std::map<int, unsigned int> buckets1;
    std::map<int, unsigned int> buckets2;
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<tscalar> dis(0, 1);
    std::normal_distribution<tscalar> ndis(0, 1);
    for(int i = 0;i < 10000000;i++){
        tscalar sample1 = convertToNormal(dis(gen), 1);
        tscalar sample2 = ndis(gen);
        buckets1[(int)std::round(sample1 * 10)]++;
        buckets2[(int)std::round(sample2 * 10)]++;
    }
    auto [s1, ignore] = *buckets1.begin();
    auto [s2, ignore_] = *buckets2.begin();
    auto [e1, ignore__] = *buckets1.end();
    auto [e2, ignore___] = *buckets2.end();
    int begin = std::min(s1, s2);
    int end = std::max(e1, e2);

    for(int i = begin; i <= end;i++){
        sbuckets << i / 10.0 << " ";
        if(buckets1.contains(i)){
            sbuckets << buckets1[i] << " ";
        }
        else{
            sbuckets << "0 ";
        }
        if(buckets2.contains(i)){
            sbuckets << buckets2[i]<< "\n";
        }
        else{
            sbuckets << "0\n";
        }
    }
}
static_assert(sizeof(ippl::Vector<float, 3>) == 12);
static_assert(sizeof(ippl::Vector<double, 3>) == 24);
static_assert(std::is_trivial_v<ippl::Vector<float, 3>>);
static_assert(std::is_standard_layout_v<ippl::Vector<float, 3>>);
static_assert(std::is_trivial_v<ippl::Vector<double, 3>>);
static_assert(std::is_standard_layout_v<ippl::Vector<double, 3>>);


template<typename Double>
void initializeBunchEllipsoid (BunchInitialize<Double> bunchInit, ChargeVector<Double> & chargeVector, int rank, int size, int ia){
    /* Correct the number of particles if it is not a multiple of four.					*/
    if ( bunchInit.numberOfParticles_ % 4 != 0 )
      {
        unsigned int n = bunchInit.numberOfParticles_ % 4;
        bunchInit.numberOfParticles_ += 4 - n;
        //printmessage(std::string(__FILE__), __LINE__, std::string("Warning: The number of particles in the bunch is not a multiple of four. ") +
        //    std::string("It is corrected to ") +  std::to_string(bunchInit.numberOfParticles_) );
    }

    /* Save the initially given number of particles.							*/
    unsigned int	Np = bunchInit.numberOfParticles_, i, Np0 = chargeVector.size();

    /* Declare the required parameters for the initialization of charge vectors.                      	*/
    Charge<Double>         	charge; charge.q  = bunchInit.cloudCharge_ / Np;
    FieldVector<Double> gb = bunchInit.initialGamma_ * bunchInit.betaVector_;
    FieldVector<Double> r  (0.0);
    FieldVector<Double> t  (0.0);
    Double            	t0, g;
    Double		zmin = 1e100;
    Double		Ne, bF, bFi;
    unsigned int	bmi;
    std::vector<Double>	randomNumbers;

    /* The initialization in group of four particles should only be done if there exists an undulator in
     * the interaction.											*/
    unsigned int	ng = ( bunchInit.lambda_ == 0.0 ) ? 1 : 4;

    /* Check the bunching factor.                                                                     	*/
    if ( bunchInit.bF_ > 2.0 || bunchInit.bF_ < 0.0 )
      {
        //printmessage(std::string(__FILE__), __LINE__, std::string("The bunching factor can not be larger than one or a negative value !!!") );
        //exit(1);
      }

    /* If the generator is random we should make sure that different processors do not produce the same
     * random numbers.											*/
    if 	( bunchInit.generator_ == "random" )
      {
        /* Initialize the random number generator.								*/
        srand ( time(NULL) );
        /* Np / ng * 20 is the maximum number of particles.							*/
        randomNumbers.resize( Np / ng * 20, 0.0);
        for ( unsigned int ri = 0; ri < Np / ng * 20; ri++)
          randomNumbers[ri] = (float)std::min(1 - 1e-7, std::max(1e-7, ((double) rand() ) / RAND_MAX));
      }

    /* Declare the generator function depending on the input.						*/
    auto generate = [&] (unsigned int n, unsigned int m) {
      //if 	( bunchInit.generator_ == "random" )
        return  ( randomNumbers.at( n * 2 * Np/ng + m ) );
      //else
      //  return  ( randomNumbers[ n * 2 * Np/ng + m ] );
    //TODO: Return halton properly
        //return 	( halton(n,m) );
    };

    /* Declare the function for injecting the shot noise.						*/
    auto insertCharge = [&] (Charge<Double> q) {

      for ( unsigned int ii = 0; ii < ng; ii++ )
        {
          /* The random modulation is introduced depending on the shot-noise being activated.		*/
          if ( bunchInit.shotNoise_ )
            {
              /* Obtain the number of beamlet.								*/
              bmi = int( ( charge.rnp[2] - zmin ) / bunchInit.lambda_ );

              /* Obtain the phase and amplitude of the modulation.					*/
              bFi = bF * sqrt( - 2.0 * log( generate( 8 , bmi ) ) );

              q.rnp[2]  = charge.rnp[2] - bunchInit.lambda_ / 4 * ii;

              q.rnp[2] -= bunchInit.lambda_ / M_PI * bFi * sin( 2.0 * M_PI / bunchInit.lambda_ * q.rnp[2] + 2.0 * M_PI * generate( 9 , bmi ) );
            }
          else if ( bunchInit.lambda_ != 0.0)
            {
              q.rnp[2]  = charge.rnp[2] - bunchInit.lambda_ / 4 * ii;

              q.rnp[2] -= bunchInit.lambda_ / M_PI * bunchInit.bF_ * sin( 2.0 * M_PI / bunchInit.lambda_ * q.rnp[2] + bunchInit.bFP_ * M_PI / 180.0 );
            }

          /* Set this charge into the charge vector.							*/
          chargeVector.push_back(q);
        }
    };

    /* If the shot noise is on, we need the minimum value of the bunch z coordinate to be able to
     * calculate the FEL bucket number.									*/
    if ( bunchInit.shotNoise_ )
      {
        for (i = 0; i < Np / ng; i++)
          {
            if ( bunchInit.distribution_ == "uniform" )
              zmin = std::min(   Double( 2.0 * generate(2, i + Np0) - 1.0 ) * bunchInit.sigmaPosition_[2] , zmin );
            else if ( bunchInit.distribution_ == "gaussian" )
              zmin = std::min(  (Double) (bunchInit.sigmaPosition_[2] * sqrt( - 2.0 * log( generate(2, i + Np0) ) ) * sin( 2.0 * M_PI * generate(3, i + Np0) ) ), zmin );
            else
              {
                //printmessage(std::string(__FILE__), __LINE__, std::string("The longitudinal type is not correctly given to the code !!!") );
                exit(1);
              }
          }

        if ( bunchInit.distribution_ == "uniform" )
          for ( ; i < unsigned( Np / ng * ( 1.0 + 2.0 * bunchInit.lambda_ * sqrt( 2.0 * M_PI ) / ( 2.0 * bunchInit.sigmaPosition_[2] ) ) ); i++)
            {
              t0  = 2.0 * bunchInit.lambda_ * sqrt( - 2.0 * log( generate( 2, i + Np0 ) ) ) * sin( 2.0 * M_PI * generate( 3, i + Np0 ) );
              t0 += ( t0 < 0.0 ) ? ( - bunchInit.sigmaPosition_[2] ) : ( bunchInit.sigmaPosition_[2] );

              zmin = std::min(   t0 , zmin );
            }

        //zmin = zmin + bunchInit.position_[ia][2];
        zmin = zmin + bunchInit.position_[2];

        /* Obtain the average number of electrons per FEL beamlet.					*/
        Ne = bunchInit.cloudCharge_ * bunchInit.lambda_ / ( 2.0 * bunchInit.sigmaPosition_[2] );

        /* Set the bunching factor level for the shot noise depending on the given values.		*/
        bF = ( bunchInit.bF_ == 0.0 ) ? 1.0 / sqrt(Ne) : bunchInit.bF_;

        //printmessage(std::string(__FILE__), __LINE__, std::string("The standard deviation of the bunching factor for the shot noise implementation is set to ") + stringify(bF) );
      }

    /* Determine the properties of each charge point and add them to the charge vector.               	*/
    for (i = rank; i < Np / ng; i += size)
      {
        /* Determine the transverse coordinate.								*/
        r[0] = bunchInit.sigmaPosition_[0] * sqrt( - 2.0 * log( generate(0, i + Np0) ) ) * cos( 2.0 * M_PI * generate(1, i + Np0) );
        r[1] = bunchInit.sigmaPosition_[1] * sqrt( - 2.0 * log( generate(0, i + Np0) ) ) * sin( 2.0 * M_PI * generate(1, i + Np0) );

        /* Determine the longitudinal coordinate.							*/
        if ( bunchInit.distribution_ == "uniform" )
          r[2] = ( 2.0 * generate(2, i + Np0) - 1.0 ) * bunchInit.sigmaPosition_[2];
        else if ( bunchInit.distribution_ == "gaussian" )
          r[2] = bunchInit.sigmaPosition_[2] * sqrt( - 2.0 * log( generate(2, i + Np0) ) ) * sin( 2.0 * M_PI * generate(3, i + Np0) );
        else
          {
            //printmessage(std::string(__FILE__), __LINE__, std::string("The longitudinal type is not correctly given to the code !!!") );
            exit(1);
          }

        /* Determine the transverse momentum.								*/
        t[0] = bunchInit.sigmaGammaBeta_[0] * sqrt( - 2.0 * log( generate(4, i + Np0) ) ) * cos( 2.0 * M_PI * generate(5, i + Np0) );
        t[1] = bunchInit.sigmaGammaBeta_[1] * sqrt( - 2.0 * log( generate(4, i + Np0) ) ) * sin( 2.0 * M_PI * generate(5, i + Np0) );
        t[2] = bunchInit.sigmaGammaBeta_[2] * sqrt( - 2.0 * log( generate(6, i + Np0) ) ) * cos( 2.0 * M_PI * generate(7, i + Np0) );

        if ( fabs(r[0]) < bunchInit.tranTrun_ && fabs(r[1]) < bunchInit.tranTrun_ && fabs(r[2]) < bunchInit.longTrun_)
          {
            /* Shift the generated charge to the center position and momentum space.			*/
            //charge.rnp    = bunchInit.position_[ia];
            charge.rnp    = bunchInit.position_;
            charge.rnp   += r;

            charge.gb   = gb;
            charge.gb  += t;
            //std::cout << gb << "\n";
            if(std::isinf(gb[2])){
                std::cerr << "it klonked here\n";
            }

            /* Insert this charge and the mirrored ones into the charge vector.				*/
            insertCharge(charge);
          }
      }

    /* If the longitudinal type of the bunch is uniform a tapered part needs to be added to remove the
     * CSE from the tail of the bunch.									*/
    if ( bunchInit.distribution_ == "uniform" ){
      for ( ; i < unsigned( uint32_t(Np / ng) * ( 1.0 + 2.0 * bunchInit.lambda_ * sqrt( 2.0 * M_PI ) / ( 2.0 * bunchInit.sigmaPosition_[2] ) ) ); i += size)
        {
            
          r[0] = bunchInit.sigmaPosition_[0] * sqrt( - 2.0 * log( generate(0, i + Np0) ) ) * cos( 2.0 * M_PI * generate(1, i + Np0) );
          r[1] = bunchInit.sigmaPosition_[1] * sqrt( - 2.0 * log( generate(0, i + Np0) ) ) * sin( 2.0 * M_PI * generate(1, i + Np0) );

          /* Determine the longitudinal coordinate.							*/
          r[2] = 2.0 * bunchInit.lambda_ * sqrt( - 2.0 * log( generate(2, i + Np0) ) ) * sin( 2.0 * M_PI * generate(3, i + Np0) );
          r[2] += ( r[2] < 0.0 ) ? ( - bunchInit.sigmaPosition_[2] ) : ( bunchInit.sigmaPosition_[2] );

          /* Determine the transverse momentum.								*/
          t[0] = bunchInit.sigmaGammaBeta_[0] * sqrt( - 2.0 * log( generate(4, i + Np0) ) ) * cos( 2.0 * M_PI * generate(5, i + Np0) );
          t[1] = bunchInit.sigmaGammaBeta_[1] * sqrt( - 2.0 * log( generate(4, i + Np0) ) ) * sin( 2.0 * M_PI * generate(5, i + Np0) );
          t[2] = bunchInit.sigmaGammaBeta_[2] * sqrt( - 2.0 * log( generate(6, i + Np0) ) ) * cos( 2.0 * M_PI * generate(7, i + Np0) );
          //std::cerr << "DOING UNIFORM tapering!!!\n";
          if ( fabs(r[0]) < bunchInit.tranTrun_ && fabs(r[1]) < bunchInit.tranTrun_ && fabs(r[2]) < bunchInit.longTrun_)
            {
                //std::cerr << "ACTUALLY DOING UNIFORM tapering!!!\n";
              /* Shift the generated charge to the center position and momentum space.			*/
              charge.rnp   = bunchInit.position_[ia];
              charge.rnp  += r;

              charge.gb  = gb;
              
              charge.gb += t;
              //std::cout << gb[0] << "\n";
              //if(std::isinf(gb.squaredNorm())){
              //    std::cerr << "it klonked here\n";
              //}
              /* Insert this charge and the mirrored ones into the charge vector.			*/
              insertCharge(charge);
            }
        }
    }

    /* Reset the value for the number of particle variable according to the installed number of
     * macro-particles and perform the corresponding changes.                                         	*/
    bunchInit.numberOfParticles_ = chargeVector.size();
}
template<typename Double>
void boost_bunch(ChargeVector<Double>& chargeVectorn_, Double frame_gamma){
    /****************************************************************************************************/
    Double frame_beta = std::sqrt((double)frame_gamma * frame_gamma - 1.0) / double(frame_gamma);
    /* Set the bunch update time step if it is given, otherwise set it according to the MITHRA rules.	*/
    //bunch_.timeStep_			/= gamma_;

    /* Adjust the given bunch time step according to the given field time step.				*/
    //if (bunch_.timeStep_ == 0)
    //  bunch_.timeStep_ 			 = mesh_.timeStep_ ;
    //else
    //  bunch_.timeStep_ 			 = mesh_.timeStep_ / ceil(mesh_.timeStep_ / bunch_.timeStep_);
    //nUpdateBunch_    			 = mesh_.timeStep_ / bunch_.timeStep_;
    //printmessage(std::string(__FILE__), __LINE__, std::string("Time step for the bunch update is set to " + stringify(bunch_.timeStep_ * gamma_) ) );

    /****************************************************************************************************/

    /* Boost the bunch sampling parameters into the electron rest frame.				*/
    //bunch_.rhythm_			/= gamma_;
    //bunch_.bunchVTKRhythm_		/= gamma_;
    //for (unsigned int i = 0; i < bunch_.bunchProfileTime_.size(); i++)
    //  bunch_.bunchProfileTime_[i] 	/= gamma_;
    //bunch_.bunchProfileRhythm_		/= gamma_;

    /****************************************************************************************************/

    /* Boost the coordinates of the macro-particles into the bunch rest frame. During the boosting, the
     * maximum value of the z-coordinate is important, since it needs to be used in the shift that will
     * be introduced to the bunch.									*/

    Double zmaxL = -1.0e100, zmaxG;
    for (auto iterQ = chargeVectorn_.begin(); iterQ != chargeVectorn_.end(); iterQ++ )
      {
        Double g  	= std::sqrt(1.0 + iterQ->gb.squaredNorm());
        if(std::isinf(g)){
            std::cerr << __FILE__  << ": " << __LINE__ << " inf gb: " << iterQ->gb << ", g = " << g << "\n";
            abort();
        }
        Double bz 	= iterQ->gb[2] / g;
        iterQ->rnp[2]  *= frame_gamma;
        
        iterQ->gb[2] 	= frame_gamma * g * ( bz - frame_beta );
        
        zmaxL 		= std::max( zmaxL , iterQ->rnp[2] );
      }
    zmaxG = zmaxL;
    
    /****************************************************************************************************/

    /* Here, we define the shift in time such that the bunch end is at the begin of fringing field
     * section. For optical undulator such a separation should be considered in the input parameters
     * where offset is given.										*/

    /* This shift in time makes sure that the maximum z in the bunch at time zero is at undulator[0].dist_
     * away from the undulator begin ( the default distance is 2 undulator periods for static undulators
     * and 10 undulator periods for optical ones, due to different fring field formats used in the two
     * cases.)												*/
    //if ( undulator_.size() > 0 )
    //  {
    //    Double nl = ( undulator_[0].type_ == STATIC ) ? 2.0 : 5.0 * undulator_[0].signal_.nR_;
    //    if (undulator_[0].dist_ == 0.0)
    //      undulator_[0].dist_ = nl * undulator_[0].lu_;
    //    else if (undulator_[0].dist_ < nl * undulator_[0].lu_)
    //      printmessage(std::string(__FILE__), __LINE__, std::string("Warning: the undulator is set very close to the bunch, the results may be inaccurate.") );
    //    //dt_ 		= - 1.0 / ( beta_ * undulator_[0].c0_ ) * ( zmaxG + undulator_[0].dist_ / gamma_ );
    //    printmessage(std::string(__FILE__), __LINE__, std::string("Initial distance from bunch head to undulator is ") + stringify(undulator_[0].dist_) );
    //  }

    /* The same shift in time should also be done for the seed field.					*/
    //seed_.dt_		= dt_;

    /* With the above definition in the time begin the entrance of the undulator, i.e. z = 0 in the lab
     * frame, corresponds to the z = zmax + undulator_[0].dist_ / gamma_ in the bunch rest frame at
     * the initialization instant, i.e. timeBunch = 0.0.
     * In MITHRA, we assume that the particle move on a straight line before reaching the start point.
     * This forces the bunch properties to be as given at the start point of the undulator. The start
     * point is here the undulator entrance minus the undulator distance.				*/
    struct {
       Double zu_;
       Double beta_;
    } bunch_;
    bunch_.zu_ 		= zmaxG;
    bunch_.beta_ 	= frame_beta;

    /****************************************************************************************************/

    for (auto iterQ = chargeVectorn_.begin(); iterQ != chargeVectorn_.end(); iterQ++ )
      {
        Double g	= std::sqrt(1.0 + iterQ->gb.squaredNorm());
        iterQ->rnp[0]  += iterQ->gb[0] / g * ( iterQ->rnp[2] - bunch_.zu_ ) * frame_beta;
        iterQ->rnp[1]  += iterQ->gb[1] / g * ( iterQ->rnp[2] - bunch_.zu_ ) * frame_beta;
        iterQ->rnp[2]  += iterQ->gb[2] / g * ( iterQ->rnp[2] - bunch_.zu_ ) * frame_beta;
        if(std::isnan(iterQ->rnp[2])){
            std::cerr << iterQ->gb[2] << ", " << g << ", " << iterQ->rnp[2] << ", " << bunch_.zu_  << ", " <<  frame_beta << "\n";
            std::cerr << __FILE__  << ": " << __LINE__ << "   OOOOOF\n";
            abort();
        }
      }


    /****************************************************************************************************/

    /* The bunch needs to be shifted such that it is centered in the computational domain when it enters
     * the undulator. For this the bunch mean z-coordinate and beta_z are required.			*/
    /*if ( mesh_.optimizePosition_ && ( undulator_.size() > 0 ))
      {
        Double zL  = 0.0, zG;
        Double bzL = 0.0, bzG;
        for (auto iterQ = chargeVectorn_.begin(); iterQ != chargeVectorn_.end(); iterQ++ )
          {
            zL += iterQ->rnp[2];
            bzL += iterQ->gb[2] / std::sqrt( 1 + iterQ->gb.norm2() );
          }
        
        zG = zl;//MPI_Allreduce(&zL, &zG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        bzG = bzL;//MPI_Allreduce(&bzL, &bzG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        unsigned int NqL = chargeVectorn_.size(), NqG = 0;
        //MPI_Allreduce(&NqL, &NqG, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        NqG = NqL;
        zG /= NqG;
        bzG /= NqG;

        Double shift 	 = bzG * (zmaxG + undulator_[0].dist_ / gamma_ - zG) / (bzG + beta_) + zG;
        zmaxG 		-= shift;
        bunch_.zu_ 	 = zmaxG;
        dt_ 		 = - 1.0 / ( beta_ * undulator_[0].c0_ ) * ( zmaxG + undulator_[0].dist_ / gamma_ );
        seed_.dt_ 	 = dt_;

        for (auto iterQ = chargeVectorn_.begin(); iterQ != chargeVectorn_.end(); iterQ++ )
          iterQ->rnp[2] -= shift;

        printmessage(std::string(__FILE__), __LINE__, std::string("The bunch center is shifted back by ") + stringify(shift) + std::string(" .") );
      }*/

    /****************************************************************************************************/

    /* Distribute particles in their respective processor, depending on their longituinal coordinate.	*/
    //distributeParticles(chargeVectorn_);
//
    ///* Print the total number of macro-particles for the user.						*/
    //unsigned int NqL = chargeVectorn_.size(), NqG = 0;
    //MPI_Reduce(&NqL,&NqG,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    //printmessage(std::string(__FILE__), __LINE__, std::string("The total number of macro-particles is equal to ") + stringify(NqG) + std::string(" .") );
//
    ///* Initialize the total number of charges.								*/
    //Nc_ = chargeVectorn_.size();
//
    ///****************************************************************************************************/
//
    ///* If the value of the total travel distance for the simulation is nonzero, correct the total time
    // * factor in the simulation.									*/
    //if ( mesh_.totalDist_ > 0.0 )
    //  {
    //    /* Define the necessary variables, and get zmin and average beta_z.				*/
    //    double Lu = 0.0;
    //    for (auto und = undulator_.begin(); und != undulator_.end(); und++)
    //      Lu += und->lu_ * und->length_ / gamma_;
    //    double zEnd = mesh_.totalDist_ / gamma_;
    //    double zMin = 1e100;
    //    double bz = 0;
    //    for (auto iter = chargeVectorn_.begin(); iter != chargeVectorn_.end(); iter++)
    //      {
    //        zMin = std::min(zMin, iter->rnp[2]);
    //        bz += iter->gb[2] / std::sqrt(1 + iter->gb.norm2());
    //      }
    //    MPI_Allreduce(MPI_IN_PLACE, &zMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    //    MPI_Allreduce(MPI_IN_PLACE, &bz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //    unsigned int Nq = chargeVectorn_.size();
    //    MPI_Allreduce(MPI_IN_PLACE, &Nq, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //    bz /= Nq;
//
    //    mesh_.totalTime_ = 1 / (c0_ * (bz + beta_)) * (zEnd - beta_ * c0_ * dt_ - zMin + bz / beta_* Lu);
//
    //    printmessage(std::string(__FILE__), __LINE__, std::string("The total time to simulate has been set to ") + stringify(mesh_.totalTime_ * gamma_) + std::string(" .") );
//
    //  }
  }
template<typename scalar>
particles<scalar, device>  initialize_bunch_mithra(
    const BunchInitialize<scalar>& bunchInit,
    scalar frame_gamma
){
    particles<scalar, device> ret;
    particles<scalar, host> mithrabunch;

    ChargeVector<scalar> oof;
    initializeBunchEllipsoid(bunchInit, oof, 0,1,0);
    for(auto& c : oof){
        if(std::isnan(c.rnp[0]) || std::isnan(c.rnp[1]) || std::isnan(c.rnp[2]))
            std::cout << "Pos before boost: " << c.rnp << "\n";
        if(std::isinf(c.rnp[0]) || std::isinf(c.rnp[1]) || std::isinf(c.rnp[2]))
            std::cout << "Pos before boost: " << c.rnp << "\n";
    }
    boost_bunch(oof, frame_gamma);
    std::cout << "boost gamma" << frame_gamma << "\n";
    for(auto& c : oof){
         if(std::isnan(c.rnp[0]) || std::isnan(c.rnp[1]) || std::isnan(c.rnp[2])){
            std::cout << "Pos after boost: " << c.rnp << "\n";
            break;
        }
    }
    auto iterQ = oof.begin();
    mithrabunch.resize(oof.size());
    for (size_t i = 0; i < oof.size(); i++) {
        assert_isreal(iterQ->gb[0]);
        assert_isreal(iterQ->gb[1]);
        assert_isreal(iterQ->gb[2]);
        assert(iterQ->gb[2] != 0.0f);
        //ippl::Vector<scalar, 4> bunchframe_spacetime = (frame_boost.unprimedToPrimed() * prepend_t(iterQ->rnp, scalar(0)));
        //ippl::Vector<scalar, 3> bunchframe_space = bunchframe_spacetime.template tail<3>();

        // std::cout << "GammaBeta: " << iterQ->gb[2] << "\n";
        scalar g = std::sqrt(1.0 + iterQ->gb.squaredNorm());
        assert_isreal(g);
        scalar bz = iterQ->gb[2] / g;
        assert_isreal(bz);
        // iterQ->rnp[2] *= frame_gamma;

        //iterQ->gb[2] = frame_gamma * g * (bz - frame_beta);
        //bunchframe_space -= iterQ->gb / sqrt(scalar(1) + iterQ->gb.dot(iterQ->gb)) * bunchframe_spacetime[0];
        //iterQ->rnp = bunchframe_space;
        //iterQ->rnm = bunchframe_space;
        //assert_isreal(iterQ->rnp[2]);
        //assert_isreal(iterQ->gb[2]);
        //initialpos << iterQ->rnp << "\n";
        mithrabunch.previous_positions[i] = iterQ->rnp;
        mithrabunch.positions[i] = iterQ->rnp;
        mithrabunch.gammaBeta[i] = iterQ->gb;
        ++iterQ;
    }
    
    ret.update_from_hostcopy(mithrabunch);
    auto mean_x = ret.covariance_matrix().first;
    std::cout << "MEANPOS: " << mean_x.template head<3>() << std::endl;
    using vec3 = decltype(ret)::vec3;
    ret.for_each([=]KOKKOS_FUNCTION(vec3& ppos, vec3& pos, vec3&){
        ppos -= mean_x.template head<3>();
        pos -= mean_x.template head<3>();
    });
    return ret;
}
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
        default: return "<unknown>";
    }
}
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\n \nerror %d: %s\nterminating!\n", file, line, err, 
                                    _cudaGetErrorEnum(err)); 

        cudaDeviceReset(); assert(0);
    } 
}
#define cufftSafeCall(expr) do{ \
    cufftResult res = expr; \
    __cufftSafeCall(res, __FILE__, __LINE__); \
} while(false);
template<typename scalar>
particles<scalar, device> initialize_bunch(
    size_t num_particles,
    scalar total_charge,
    scalar total_mass,
    typename particles<scalar, device>::vec3 mean_pos,
    typename particles<scalar, device>::vec3 stddev_pos, 
    typename particles<scalar, device>::vec3 truncations_pos,
    LorentzFrame<scalar> frame_boost,
    typename particles<scalar, device>::vec3 stddev_gammabeta,
    scalar mean_bunch_gamma_and_direction_in_z_direction,
    scalar undulator_lambda){
    uint32_t mirror_count = 4;
    num_particles -= (num_particles % mirror_count);
    using vec3 = typename particles<scalar, device>::vec3;
    
    const scalar bunch_beta = std::sqrt(mean_bunch_gamma_and_direction_in_z_direction * mean_bunch_gamma_and_direction_in_z_direction - 1) / mean_bunch_gamma_and_direction_in_z_direction;
    const scalar frame_beta = frame_boost.beta_m[2];
    const scalar frame_gamma = frame_boost.gamma_m;
    if(num_particles == 0){
        return particles<scalar, device>();
    }
    #define CURAND_CALL(x) do {                                     \
        if((x) != CURAND_STATUS_SUCCESS) {                          \
        printf("Error at %s:%d\n",__FILE__,__LINE__);               \
        return EXIT_FAILURE;                                        \
    }} while(0)
    
    
    scalar gammabeta_mean = mean_bunch_gamma_and_direction_in_z_direction * (sqrt(mean_bunch_gamma_and_direction_in_z_direction * mean_bunch_gamma_and_direction_in_z_direction - 1) / mean_bunch_gamma_and_direction_in_z_direction);
    const scalar k_u                 = scalar(2) * M_PI / undulator_lambda;
    const scalar chi = 1 + bunch_beta / frame_beta;
    size_t iter_limit = num_particles / mirror_count;
    std::vector<vec3> hpos,hppos;
    std::vector<vec3> hgb;
    std::complex<double> bfactor_recompute_accum = 0;
    constexpr double initial_bunching_factor = 0.01;
    for(size_t idx = 0;idx < iter_limit;idx++){
        assert(idx * mirror_count + mirror_count - 1 < num_particles);
        XORShift64 s(idx);
        s.scramble(123123219979ull);
        if(s.state == 0){
            printf("%lu and %lu\n", idx, 12312321ul);
        }
        
        //decltype(bunch)::vec3 direction{0, 0, 1};
        vec3 position;
        for(unsigned d = 0;d < 2;d++){
            
            scalar nsample;
            unsigned tries = 0;
            do{
                tries++;
                nsample = convertToNormal(s.next<scalar>(), stddev_pos[d]);
                //printf("%.5e\n", s.next<scalar>());
            }while(abs(nsample) > truncations_pos[d] && tries < 100);
            //assert(false);
            if(tries >= 100){
                nsample = 0.0f;
            }
            position[d] = (scalar)mean_pos[d] + nsample;
        };
        //position[2] = (scalar(idx) / (bunch.count_m - 1) - scalar(0.5)) * scalar(2) * stddev_pos[2] + mean_pos[2];
        //position[2] += (chi * frame_gamma * 0.01 * sin(2 * chi * frame_gamma * k_u * position[2])) / k_u;
        
        //bunch.positions[idx] = (frame_boost.unprimedToPrimed() * prepend_t(position, scalar(0))).template tail<3>();
        //bunch.previous_positions[idx] = (frame_boost.unprimedToPrimed() * prepend_t(position, scalar(0))).template tail<3>();

        //bunch.positions[idx][2] += (chi * frame_gamma * 0.0 * sin(2 * chi * frame_gamma * k_u * bunch.positions[idx][2])) / k_u;
        //bunch.previous_positions[idx][2] += (chi * frame_gamma * 0.0 * sin(2 * chi * frame_gamma * k_u * bunch.previous_positions[idx][2])) / k_u;
        //printf("dx = %f\n", (chi * frame_gamma * 0.001 * sin(2 * chi * frame_gamma * k_u * bunch.positions[idx][2])) / bunch.positions[idx][2]);
        const scalar lambda_r =  undulator_lambda / (scalar(2) * frame_gamma * frame_gamma); 
        const scalar lambda_s = lambda_r * bunch_beta / frame_beta;
        position[2] = (s.next<scalar>() - scalar(0.5)) * scalar(2) * stddev_pos[2] + mean_pos[2];
        vec3 gb;
        gb[0] = scalar(0) + convertToNormal(s.next<scalar>(), (scalar)stddev_gammabeta[0]);
        gb[1] = scalar(0) + convertToNormal(s.next<scalar>(), (scalar)stddev_gammabeta[1]);
        gb[2] = gammabeta_mean + convertToNormal(s.next<scalar>(), (scalar)stddev_gammabeta[2]);
        assert_isreal(gb[0]);
        assert_isreal(gb[1]);
        assert_isreal(gb[2]);
        
        for(int mirror = 0;mirror < mirror_count;mirror++){
            vec3 position_with_quarter_offset = position - vec3{scalar(0), scalar(0), scalar(mirror) / scalar(mirror_count) * lambda_s};

            //const scalar phase = position_with_quarter_offset[2] / lambda_s;

            //TODO: Bunching factor correctly. (I think this way it's correct tbh)
            position_with_quarter_offset[2] -= lambda_s / M_PI * initial_bunching_factor * sin(2.0 * M_PI / lambda_s * position_with_quarter_offset[2]);
            bfactor_recompute_accum += std::polar<double>(1.0 / iter_limit / 4.0, 2 * M_PI * position_with_quarter_offset[2] / lambda_s);
            vec3 gb_bunchframe = frame_boost.transformGammabeta(gb);
            ippl::Vector<scalar, 4> bunchframe_spacetime = (frame_boost.unprimedToPrimed() * prepend_t(position_with_quarter_offset, scalar(0)));
            ippl::Vector<scalar, 3> bunchframe_space = bunchframe_spacetime.template tail<3>();
            bunchframe_space -= gb_bunchframe / sqrt(scalar(1) + gb_bunchframe.dot(gb_bunchframe)) * bunchframe_spacetime[0];
            hpos.push_back(bunchframe_space);
            hppos.push_back(bunchframe_space);

            hgb.push_back(gb_bunchframe);
            assert_isreal(hpos.back()[0]);
            assert_isreal(hpos.back()[1]);
            assert_isreal(hpos.back()[2]);
        }
        //printf("GB: %f, %f, %f\n", bunch.gammaBeta[idx][0], bunch.gammaBeta[idx][1], bunch.gammaBeta[idx][2]);
    }
    //std::cerr << "Initial böntsching factor computed directly in the thing: " << bfactor_recompute_accum << "\n";
    const scalar lambda_r =  undulator_lambda / (scalar(2) * frame_gamma * frame_gamma); 
    const scalar lambda_s = lambda_r * bunch_beta / frame_beta;
    xoshiro_256 gen(42);
    std::uniform_real_distribution<scalar> dis(1e-7, scalar(1) - 1e-7);

          for ( unsigned i = num_particles / mirror_count; i < unsigned(num_particles / mirror_count * ( 1.0 + 2.0 * lambda_s * sqrt( 2.0 * M_PI ) / ( 2.0 * stddev_pos[2] ) ) ); i++){
            //std::cerr << "Outerloop\n";
        vec3 r;
        vec3 t;
        /**
         * @brief stupid af, this should just be done with a normal distribution
         * 
         */
      r[0] = stddev_pos[0] * convertToNormal(dis(gen), scalar(1));
      r[1] = stddev_pos[1] * convertToNormal(dis(gen), scalar(1));

      /* Determine the longitudinal coordinate.*/
      r[2] = 2.0 * lambda_s * sqrt( - 2.0 * log(dis(gen))) * sin(2.0 * M_PI * dis(gen));
      r[2] += ( r[2] < 0.0 ) ? (-stddev_pos[2]) : (stddev_pos[2]);

      /* Determine the transverse momentum.*/
      t[0] = stddev_gammabeta[0] *                  convertToNormal(dis(gen), scalar(1));
      t[1] = stddev_gammabeta[1] *                  convertToNormal(dis(gen), scalar(1));
      t[2] = gammabeta_mean + stddev_gammabeta[2] * convertToNormal(dis(gen), scalar(1));

      if (std::abs(r[0]) < truncations_pos[0] && std::abs(r[1]) < truncations_pos[1] /*&& std::abs(r[2]) < truncations_pos[2]*/)
        {
            //std::cerr << "Inserted weird particle\n";
          /* Shift the generated charge to the center position and momentum space.			*/
          vec3 insert_pos = r;
          insert_pos += mean_pos;
          vec3 insert_ppos = insert_pos;
          vec3 insert_gammabeta = t;

          for(int mirror = 0;mirror < mirror_count;mirror++){
            vec3 position_with_quarter_offset = insert_pos - vec3{scalar(0), scalar(0), scalar(mirror) / scalar(mirror_count) * lambda_s};

            //const scalar phase = position_with_quarter_offset[2] / lambda_s;

            //TODO: Bunching factor correctly. (I think this way it's correct tbh)
            position_with_quarter_offset[2] -= lambda_s / M_PI * initial_bunching_factor * sin(2.0 * M_PI / lambda_s * position_with_quarter_offset[2]);
            
            vec3 gb_bunchframe = frame_boost.transformGammabeta(insert_gammabeta);
            ippl::Vector<scalar, 4> bunchframe_spacetime = (frame_boost.unprimedToPrimed() * prepend_t(position_with_quarter_offset, scalar(0)));
            ippl::Vector<scalar, 3> bunchframe_space = bunchframe_spacetime.template tail<3>();
            bunchframe_space -= gb_bunchframe / sqrt(scalar(1) + gb_bunchframe.dot(gb_bunchframe)) * bunchframe_spacetime[0];
            hpos.push_back(bunchframe_space);
            hppos.push_back(bunchframe_space);

            hgb.push_back(gb_bunchframe);
            assert_isreal(hpos.back()[0]);
            assert_isreal(hpos.back()[1]);
            assert_isreal(hpos.back()[2]);
            }
        }
    }
    assert(hpos.size() == hppos.size());
    assert(hgb.size() == hppos.size());
    particles<scalar, host> hparticles(hppos.size(), total_charge / num_particles, total_mass / num_particles);
    
    for(size_t i = 0;i < hppos.size();i++){
        hparticles.positions[i] = hpos[i];
        
        hparticles.previous_positions[i] = hppos[i];
        hparticles.gammaBeta[i] = hgb[i];
    }

    particles<scalar, device> bunch   (hppos.size(), total_charge / num_particles, total_mass / num_particles);
    bunch.update_from_hostcopy(hparticles);
    //std::cerr << "Initial variance z: " << bunch.zvariance()[2] << "\n";
    //std::cerr << "Initial böntsching factor: " << bunch.bunching_factor(undulator_lambda, frame_gamma) << "\n";
    
    return bunch;
}
template<typename scalar>
struct nondispersive{
    scalar a1;
    scalar a2;
    scalar a4;
    scalar a6;
    scalar a8;
};

struct do_nothing_boundary_conditions{
    template<typename value_type>
    void apply(field_state<value_type, device>& F, extract_value_t<value_type> dt){}
};
struct periodic_boundary_conditions{
    template<typename value_type>
    void apply(field_state<value_type, device>& F, extract_value_t<value_type> dt){
        const unsigned nghost = F.A_n.nghost();
        const ippl::Vector<uint32_t, 3> nr_m{
            F.A_n.m_imap.m - F.A_n.m_imap.nghost() * 2,
            F.A_n.m_imap.n - F.A_n.m_imap.nghost() * 2,
            F.A_n.m_imap.k - F.A_n.m_imap.nghost() * 2
        };
        auto A1 = F.A_np1;
        auto A2 = F.A_n;
        auto A3 = F.A_nm1;
        parallel_for_cuda([nghost, nr_m, A1, A2, A3] __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
            size_t write_to = A1.m_imap(i, j, k);
            i += nr_m[0] * uint32_t(i < nghost);
            i -= nr_m[0] * uint32_t(i >= nr_m[0] + nghost);

            j += nr_m[1] * uint32_t(j < nghost);
            j -= nr_m[1] * uint32_t(j >= nr_m[1] + nghost);

            k += nr_m[2] * uint32_t(k < nghost);
            k -= nr_m[2] * uint32_t(k >= nr_m[2] + nghost);

            A1[write_to] = A1(i, j, k);
            A2[write_to] = A2(i, j, k);
            A3[write_to] = A3(i, j, k);

        }, A1.getFullRange());
    }
};
KOKKOS_INLINE_FUNCTION int popcnt(uint32_t x){
    #ifdef __CUDA_ARCH__
    return __popc(x);
    #else
    return std::popcount(x);
    #endif
}
struct first_order_mur_boundary_conditions{
    template<typename value_type>
    void apply(field_state<value_type, device>& F, extract_value_t<value_type> dt){
        using scalar = extract_value_t<value_type>;
        const unsigned nghost = F.A_n.nghost();
        const ippl::Vector<scalar, 3> betaMur = ippl::Vector<scalar, 3>(dt) / F.A_n.m_mesh.spacing;
        assert_isreal((betaMur[0]));
        assert_isreal((betaMur[1]));
        assert_isreal((betaMur[2]));
        const ippl::Vector<uint32_t, 3> nr_m{
            F.A_n.m_imap.m - F.A_n.m_imap.nghost() * 2,
            F.A_n.m_imap.n - F.A_n.m_imap.nghost() * 2,
            F.A_n.m_imap.k - F.A_n.m_imap.nghost() * 2
        };
        const ippl::Vector<uint32_t, 3> true_nr{
            F.A_n.m_imap.m,
            F.A_n.m_imap.n,
            F.A_n.m_imap.k
        };
        auto A = F.A_n;
        auto A_np1 = F.A_np1;
        parallel_for_cuda([A, A_np1, betaMur, true_nr]KOKKOS_FUNCTION(uint32_t i, uint32_t j, uint32_t k)mutable{
            uint32_t val = uint32_t(i == 0) + (uint32_t(j == 0) << 1) + (uint32_t(k == 0) << 2)
                             + (uint32_t(i == true_nr[0] - 1) << 3) + (uint32_t(j == true_nr[1] - 1) << 4) + (uint32_t(k == true_nr[2] - 1) << 5);
            //assert(i != true_nr[0] - 1);
            int xoff = !!(val & 1) - !!(val & 8 );
            int yoff = !!(val & 2) - !!(val & 16);
            int zoff = !!(val & 4) - !!(val & 32);
            ippl::Vector<int, 3> offs{xoff, yoff, zoff};
            if(val != 0){
                value_type derivs(0);
                for(unsigned d = 0;d < 3;d++){
                    int ofs = offs[d];
                    if(ofs != 0){
                        //A_np1(i, j, k) = A(i, j, k) + (A(i + xoff * (d == 0), j + yoff * (d == 1), k + zoff * (d == 2)) - A(i, j, k)) * betaMur[d];
                        derivs += (A(i + xoff * (d == 0), j + yoff * (d == 1), k + zoff * (d == 2)) - A(i, j, k)) * betaMur[d];
                    }
                }
                A_np1(i, j, k) = A(i, j, k) + derivs;
            }
            /*val &= -val;
            if(false && __popc(val) == 1)
            {
                ippl::Vector<int, 3> off_vec{xoff, yoff, zoff};
                for(unsigned d = 0;d < 3;d++){

                }
                int ax = abs(yoff) + abs(zoff) + abs(zoff);
                assert(ax < 3);
                //for(unsigned int d = 0;d < 4;d++){
                    A(i, j, k)[even] = A(i, j, k)[!even] + (A(i + xoff, j + yoff, k + zoff)[!even] - A(i, j, k)[!even]) * betaMur[ax];
                    //assert_isreal((A(i, j, k)[even][d]));
                //}
            }*/

        }, A.getFullRange());
    }
};
struct second_order_mur_boundary_conditions{
    template<typename value_type>
    void apply(field_state<value_type, device>& F, extract_value_t<value_type> dt){
        using scalar = extract_value_t<value_type>;
        const unsigned nghost = F.A_n.nghost();
        const ippl::Vector<scalar, 3> betaMur = ippl::Vector<scalar, 3>(dt) / F.A_n.m_mesh.spacing;
        assert_isreal((betaMur[0]));
        assert_isreal((betaMur[1]));
        assert_isreal((betaMur[2]));
        const ippl::Vector<uint32_t, 3> nr_m{
            F.A_n.m_imap.m - F.A_n.m_imap.nghost() * 2,
            F.A_n.m_imap.n - F.A_n.m_imap.nghost() * 2,
            F.A_n.m_imap.k - F.A_n.m_imap.nghost() * 2
        };
        const ippl::Vector<uint32_t, 3> true_nr{
            F.A_n.m_imap.m,
            F.A_n.m_imap.n,
            F.A_n.m_imap.k
        };
        auto A_n   = F.A_n;
        auto A_np1 = F.A_np1;
        auto A_nm1 = F.A_nm1;
        parallel_for_cuda([A_np1, A_n, A_nm1, true_nr,betaMur, dt]KOKKOS_FUNCTION(uint32_t i, uint32_t j, uint32_t k)mutable{
            uint32_t val = uint32_t(i == 0) + (uint32_t(j == 0) << 1) + (uint32_t(k == 0) << 2)
                             + (uint32_t(i == true_nr[0] - 1) << 3) + (uint32_t(j == true_nr[1] - 1) << 4) + (uint32_t(k == true_nr[2] - 1) << 5);
            if(popcnt(val) == 1){
                if(i == 0){
                    second_order_abc_face<scalar, 0, 1, 2> soa(A_n.m_mesh.spacing, dt, 1);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                if(j == 0){
                    second_order_abc_face<scalar, 1, 0, 2> soa(A_n.m_mesh.spacing, dt, 1);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                if(k == 0){
                    second_order_abc_face<scalar, 2, 0, 1> soa(A_n.m_mesh.spacing, dt, 1);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                if(i == true_nr[0] - 1){
                    second_order_abc_face<scalar, 0, 1, 2> soa(A_n.m_mesh.spacing, dt, -1);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                if(j == true_nr[1] - 1){
                    second_order_abc_face<scalar, 1, 0, 2> soa(A_n.m_mesh.spacing, dt, -1);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                if(k == true_nr[2] - 1){
                    second_order_abc_face<scalar, 2, 0, 1> soa(A_n.m_mesh.spacing, dt, -1);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
            }
        }, A_n.getFullRange());
        parallel_for_cuda([A_np1, A_n, A_nm1, true_nr,betaMur, dt]KOKKOS_FUNCTION(uint32_t i, uint32_t j, uint32_t k)mutable{
            uint32_t val = uint32_t(i == 0) + (uint32_t(j == 0) << 1) + (uint32_t(k == 0) << 2)
                             + (uint32_t(i == true_nr[0] - 1) << 3) + (uint32_t(j == true_nr[1] - 1) << 4) + (uint32_t(k == true_nr[2] - 1) << 5);
            if(popcnt(val) == 2){ //Edge
                if(i == 0 && k == 0){
                    second_order_abc_edge<scalar, 1, 2, 0, true, true> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == 0 && j == 0){
                    second_order_abc_edge<scalar, 2, 0, 1, true, true> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(j == 0 && k == 0){
                    second_order_abc_edge<scalar, 0, 1, 2, true, true> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }

                else if(i == 0 && k == true_nr[2] - 1){
                    second_order_abc_edge<scalar, 1, 2, 0, false, true> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == 0 && j == true_nr[1] - 1){
                    second_order_abc_edge<scalar, 2, 0, 1, true, false> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(j == 0 && k == true_nr[2] - 1){
                    second_order_abc_edge<scalar, 0, 1, 2, true, false> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }

                else if(i == true_nr[0] - 1 && k == 0){
                    second_order_abc_edge<scalar, 1, 2, 0, true, false> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == true_nr[0] - 1 && j == 0){
                    second_order_abc_edge<scalar, 2, 0, 1, false, true> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(j == true_nr[1] - 1 && k == 0){
                    second_order_abc_edge<scalar, 0, 1, 2, false, true> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }

                else if(i == true_nr[0] - 1 && k == true_nr[2] - 1){
                    second_order_abc_edge<scalar, 1, 2, 0, false, false> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == true_nr[0] - 1 && j == true_nr[1] - 1){
                    second_order_abc_edge<scalar, 2, 0, 1, false, false> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(j == true_nr[1] - 1 && k == true_nr[2] - 1){
                    second_order_abc_edge<scalar, 0, 1, 2, false, false> soa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else{
                    assert(false);
                }
            }
        }, A_n.getFullRange());
        parallel_for_cuda([A_np1, A_n, A_nm1, true_nr,betaMur, dt]KOKKOS_FUNCTION(uint32_t i, uint32_t j, uint32_t k)mutable{
            uint32_t val = uint32_t(i == 0) + (uint32_t(j == 0) << 1) + (uint32_t(k == 0) << 2)
                             + (uint32_t(i == true_nr[0] - 1) << 3) + (uint32_t(j == true_nr[1] - 1) << 4) + (uint32_t(k == true_nr[2] - 1) << 5);
            
            if(popcnt(val) == 3){
                //printf("Corner: %d, %d, %d\n", i, j, k);
                if(i == 0 && j == 0 && k == 0){
                    second_order_abc_corner<scalar, 1, 1, 1> coa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == true_nr[0] - 1 && j == 0 && k == 0){
                    second_order_abc_corner<scalar, 0, 1, 1> coa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == 0 && j == true_nr[1] - 1 && k == 0){
                    second_order_abc_corner<scalar, 1, 0, 1> coa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == true_nr[0] - 1 && j == true_nr[1] - 1 && k == 0){
                    second_order_abc_corner<scalar, 0, 0, 1> coa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == 0 && j == 0 && k == true_nr[2] - 1){
                    second_order_abc_corner<scalar, 1, 1, 0> coa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == true_nr[0] - 1 && j == 0 && k == true_nr[2] - 1){
                    second_order_abc_corner<scalar, 0, 1, 0> coa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == 0 && j == true_nr[1] - 1 && k == true_nr[2] - 1){
                    second_order_abc_corner<scalar, 1, 0, 0> coa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else if(i == true_nr[0] - 1 && j == true_nr[1] - 1 && k == true_nr[2] - 1){
                    second_order_abc_corner<scalar, 0, 0, 0> coa(A_n.m_mesh.spacing, dt);
                    A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                }
                else{
                    assert(false);
                }
            }
        }, A_n.getFullRange());
    }
};
template<typename boundary_conditions, typename value_type>
void field_update(field_state<value_type, device>& F, extract_value_t<value_type> dt){
    using scalar = extract_value_t<value_type>;
    scalar dx = F.A_n.m_mesh.spacing[0];
    scalar dy = F.A_n.m_mesh.spacing[1];
    scalar dz = F.A_n.m_mesh.spacing[2];
    ippl::Vector<scalar, 3> hr_m = F.A_n.m_mesh.spacing;
    const scalar calA = 0.25 * (1 + 0.02 / (sq(hr_m[2] / hr_m[0]) + sq(hr_m[2] / hr_m[1])));
    ippl::Vector<uint32_t, 3> nr_m{
        F.A_n.m_imap.m - F.A_n.m_imap.nghost() * 2,
        F.A_n.m_imap.n - F.A_n.m_imap.nghost() * 2,
        F.A_n.m_imap.k - F.A_n.m_imap.nghost() * 2
    };
    //assert_isreal((a1));
    //assert_isreal((a2));
    //assert_isreal((a4));
    //assert_isreal((a6));
    //assert_isreal((a8));
    const ippl::Vector<scalar, 3> betaMur = ippl::Vector<scalar, 3>(dt) / F.A_n.m_mesh.spacing;
    assert_isreal((betaMur[0]));
    assert_isreal((betaMur[1]));
    assert_isreal((betaMur[2]));
    
    nondispersive<scalar> ndisp{
        .a1 = 2 * (1 - (1 - 2 * calA) * sq(dt / hr_m[0]) - (1 - 2*calA) * sq(dt / hr_m[1]) - sq(dt / hr_m[2])),
        .a2 = sq(dt / hr_m[0]),
        .a4 = sq(dt / hr_m[1]),
        .a6 = sq(dt / hr_m[2]) - 2 * calA * sq(dt / hr_m[0])  - 2 * calA * sq(dt / hr_m[1]),
        .a8 = sq(dt)
    };
    //parallel_for_cuda([=] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{

    //}, A.getRange());
    auto A_n   = F.A_n;
    auto A_np1 = F.A_np1;
    auto A_nm1 = F.A_nm1;
    auto source = F.source_n;
    cudaEvent_t b,e,eb;
    cudaEventCreate(&b);
    cudaEventCreate(&e);
    cudaEventCreate(&eb);

    cudaEventRecord(b);
    parallel_for_cuda([=] __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
            //typename value_type::value_type laplace_even = 
            //                          scalar(0.0 / 10) * (A(i + 1, j, k)  [!even] - A(i, j, k)[!even] * scalar(2) + A(i - 1, j, k)[!even])
            //                        + scalar(0.0 / 10) * (A(i, j + 1, k)  [!even] - A(i, j, k)[!even] * scalar(2) + A(i, j - 1, k)[!even])
            //                        + scalar(0.25 / 10) * (A(i, j, k + 1)[!even] - A(i, j, k)[!even] * scalar(2) + A(i, j, k - 1)[!even]);
            //printf("updating\n");
            //__shared__ value_type preload_n  [8 * 8 * 4];
            //__shared__ value_type preload_nm1[8 * 8 * 4];
            //ct_indexmap<8, 8, 4> ct_imap;
            //uint32_t tindex = ct_imap(threadIdx.x, threadIdx.y, threadIdx.z);
            //preload_n[tindex] = A_n(i,j,k);
            //preload_nm1[tindex] = A_nm1(i,j,k);
            //__syncthreads();
            A_np1(i, j, k) =        -A_nm1(i,j,k) + 
                          ndisp.a1 * A_n  (i,j,k)
                        + ndisp.a2 * (calA * A_n(i + 1, j, k - 1) + (1 - 2 * calA) * A_n(i + 1, j, k) + calA * A_n(i + 1, j, k + 1))
                        + ndisp.a2 * (calA * A_n(i - 1, j, k - 1) + (1 - 2 * calA) * A_n(i - 1, j, k) + calA * A_n(i - 1, j, k + 1))
                        + ndisp.a4 * (calA * A_n(i, j + 1, k - 1) + (1 - 2 * calA) * A_n(i, j + 1, k) + calA * A_n(i, j + 1, k + 1))
                        + ndisp.a4 * (calA * A_n(i, j - 1, k - 1) + (1 - 2 * calA) * A_n(i, j - 1, k) + calA * A_n(i, j - 1, k + 1))
                        + ndisp.a6 * A_n(i, j, k + 1) + ndisp.a6 * A_n(i, j, k - 1) + ndisp.a8 * source(i, j, k);// - laplace_even;
            
        }, A_n.getRange());
    
    
        //std::cout << A.getRange().begin << " to " << A.getRange().end << "\n";
        //cudaDeviceSynchronize();
        //auto average = A.sum(A.getFullRange()) / value_type(scalar(A.getFullRange().volume()));
        //value_type actual_average = (average[0] + average[1]) * scalar(0.5);
        //parallel_for_cuda([=] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
        //    //A(i,j,k)[0] -= actual_average;
        //    //A(i,j,k)[1] -= actual_average;
        //}, A.getFullRange());
        //cudaDeviceSynchronize();
        boundary_conditions bc{};
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        bc.apply(F, dt);
        cudaEventRecord(eb);
        cudaEventSynchronize(eb);
        {
            float ms = 0;
            float ms2 = 0;
            cudaEventElapsedTime(&ms, b, e);
            cudaEventElapsedTime(&ms2, b, eb);
            //std::cout << F.A_n.m_imap.k << "^3: " << ms << "\n";
            //std::cout << F.A_n.m_imap.k << "^3 total: " << ms2 << "\n";
        }
        cudaEventDestroy(b);
        cudaEventDestroy(e);
        cudaDeviceSynchronize();
        value_type* __restrict__ oldest = F.A_nm1.data;
        value_type* __restrict__ middle = F.A_n.data;
        value_type* __restrict__ newest = F.A_np1.data;
        const size_t amount = F.A_n.m_imap.size() * sizeof(value_type);
        cudaMemcpy(oldest, middle, amount, cudaMemcpyDeviceToDevice);
        cudaMemcpy(middle, newest, amount, cudaMemcpyDeviceToDevice);
        /*parallel_for_cuda([=] __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
            uint32_t val = uint32_t(i == 0) + (uint32_t(j == 0) << 1) + (uint32_t(k == 0) << 2)
                         + (uint32_t(i == nr_m[0] - 1) << 3) + (uint32_t(j == nr_m[1] - 1) << 4) + (uint32_t(k == nr_m[2] - 1) << 5);
            val &= -val;
            if(__popc(val) == 1){
                int xoff = !!(val & 1) - !!(val & 8 );
                int yoff = !!(val & 2) - !!(val & 16);
                int zoff = !!(val & 4) - !!(val & 32);
                int ax = abs(yoff) + abs(zoff) + abs(zoff);
                assert(ax < 3);
                for(unsigned int d = 0;d < 4;d++){
                    A(i, j, k)[even][d] = A(i, j, k)[!even][d] + (A(i + xoff, j + yoff, k + zoff)[!even][d] - A(i, j, k)[!even][d]) * betaMur[ax];
                    assert_isreal((A(i, j, k)[even][d]));
                }
                if(k < unsigned(nr_m[2] * 0.15)){
                    scalar exp_ = scalar(nr_m[2] * 0.15 - k) / (nr_m[2] * 0.15);
                    scalar fac = scalar(1) - scalar(0.0) * (exp_ * exp_);
                    //A(i,j,k)[0] *= fac;
                    //A(i,j,k)[1] *= fac;
                }
            }
            
            
        }, A.getFullRange());*/
}
template<typename boundary_conditions, typename value_type>
void field_update_standard(field_state<value_type, device>& F, extract_value_t<value_type> dt){
    using scalar = extract_value_t<value_type>;
    scalar dx = F.A_n.m_mesh.spacing[0];
    scalar dy = F.A_n.m_mesh.spacing[1];
    scalar dz = F.A_n.m_mesh.spacing[2];
    ippl::Vector<scalar, 3> hr_m = F.A_n.m_mesh.spacing;
    const scalar a1 = scalar(2) * (scalar(1) - sq(dt / dx) - sq(dt / dy) - sq(dt / dz));
    const scalar a2 = sq(dt / dx);
    const scalar a4 = sq(dt / dy);
    const scalar a6 = sq(dt / dz);
    const scalar a8 = sq(dt);
    ippl::Vector<uint32_t, 3> nr_m{
        F.A_n.m_imap.m - F.A_n.m_imap.nghost() * 2,
        F.A_n.m_imap.n - F.A_n.m_imap.nghost() * 2,
        F.A_n.m_imap.k - F.A_n.m_imap.nghost() * 2
    };
    assert_isreal((a1));
    assert_isreal((a2));
    assert_isreal((a4));
    assert_isreal((a6));
    assert_isreal((a8));
    auto A_n   = F.A_n;
    auto A_np1 = F.A_np1;
    auto A_nm1 = F.A_nm1;
    auto source = F.source_n;
    parallel_for_cuda([=] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{

            A_np1(i, j, k) = -A_nm1(i, j, k) +  A_n(i    , j, k) * a1
                                                         + (A_n(i + 1, j, k) + A_n(i - 1, j, k)) * a2
                                                         + (A_n(i, j + 1, k) + A_n(i, j - 1, k)) * a4
                                                         + (A_n(i, j, k + 1) + A_n(i, j, k - 1)) * a6
                                                         + source(i, j, k)                       * a8;
            
        }, A_n.getRange()
    );
    //std::cerr << A_n.getRange().end << "\n";
    //cudaDeviceSynchronize();
    //auto average = A.sum(A.getFullRange()) / ippl::Vector<scalar, 4>(scalar(A.getFullRange().volume()));
    //ippl::Vector<scalar, 4> actual_average = (average[0] + average[1]) * scalar(0.5);
    //parallel_for_cuda([=] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
    //    A(i,j,k)[0] -= actual_average;
    //    A(i,j,k)[1] -= actual_average;
    //}, A.getFullRange());
    //cudaDeviceSynchronize();
    boundary_conditions{}.apply(F, dt);
    value_type* __restrict__ oldest = F.A_nm1.data;
    value_type* __restrict__ middle = F.A_n.data;
    value_type* __restrict__ newest = F.A_np1.data;
    const size_t amount = F.A_n.m_imap.size() * sizeof(value_type);
    cudaMemcpy(oldest, middle, amount, cudaMemcpyDeviceToDevice);
    cudaMemcpy(middle, newest, amount, cudaMemcpyDeviceToDevice);
}
void teschtrand(){
    XORShift64 s(1198120);
    s.scramble(12312321);
    std::cout << s.next<double>() << "\n";
    std::cout << s.next<double>() << "\n";
    std::cout << s.next<double>() << "\n";
    std::cout << s.next<double>() << "\n";
    std::cout << s.next<double>() << "\n";
    std::cout << s.next<double>() << "\n";
    std::cout << convertToNormal(s.next<double>(), 1.0) << "\n";
    std::cout << convertToNormal(s.next<double>(), 1.0) << "\n";
    std::cout << convertToNormal(s.next<double>(), 1.0) << "\n";
    std::cout << convertToNormal(s.next<double>(), 1.0) << "\n";
    std::cout << convertToNormal(s.next<double>(), 1.0) << "\n";
    std::cout << convertToNormal(s.next<double>(), 1.0) << "\n";
}
void testmesh(const ghosted_indexmap& m){
    for(size_t i = 0;i < m.size();i++){
        auto dec = m.decompose(i);
        assert(i == m(dec));
    }
}
template<typename value_type, space sp>
field_state<value_type, sp> make_field_state(uint32_t m, uint32_t n, uint32_t k, ippl::Vector<extract_value_t<value_type>, 3> origin, ippl::Vector<extract_value_t<value_type>, 3> hr){
    using scalar = extract_value_t<value_type>;
    grid<value_type, sp> Ap1   (m,n,k);
    grid<value_type, sp> A     (m,n,k);
    grid<value_type, sp> Am1   (m,n,k);
    grid<value_type, sp> source(m,n,k);
    ippl::Vector<scalar, 3> ihr = ippl::Vector<scalar, 3>(1) / hr;
    Ap1   .m_mesh.origin = origin;
    A     .m_mesh.origin = origin;
    Am1   .m_mesh.origin = origin;
    source.m_mesh.origin = origin;

    Ap1   .m_mesh.spacing = hr;
    A     .m_mesh.spacing = hr;
    Am1   .m_mesh.spacing = hr;
    source.m_mesh.spacing = hr;

    Ap1   .m_mesh.inverse_spacing = ihr;
    A     .m_mesh.inverse_spacing = ihr;
    Am1   .m_mesh.inverse_spacing = ihr;
    source.m_mesh.inverse_spacing = ihr;

    field_state<value_type, sp> ret{Ap1, A, Am1, source};
    return ret;
}

using scalar = float;
scalar test_fieldsolve(uint32_t n){
    
    using gvt = scalar;

    ippl::Vector<scalar, 3> origin{-0.5,-0.5,-0.5};
    ippl::Vector<scalar, 3> hr_m(1);
    hr_m /= ippl::Vector<scalar, 3>{scalar(n), scalar(n), scalar(2 * n)};
    field_state<gvt, device> F = make_field_state<gvt, device>(n, n, 2 * n, origin, hr_m);
    grid<gvt, device> init(n, n, 2 * n, F.A_n.m_mesh);
    const scalar variance = scalar(1) / 10;
    const scalar dt = scalar(0.5) ** std::min_element(hr_m.begin(), hr_m.end());
    parallel_for_cuda([init, F, hr_m, origin, variance] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
        scalar x = (scalar(i) + scalar(0.5)) * hr_m[0] + origin[0];
        scalar y = (scalar(j) + scalar(0.5)) * hr_m[1] + origin[1];
        scalar z = (scalar(k) + scalar(0.5)) * hr_m[2] + origin[2];
        scalar value = gauss(scalar(0), variance, 0, 0, z);
        F.A_n(i, j, k) = value;
        F.A_nm1(i, j, k) = value;
        F.source_n(i, j, k) = 0;
        init(i, j, k) = value;
    }, F.A_n.getFullRange());

    for(uint32_t i = 0;i < 4 * n;i++){
        field_update_standard<periodic_boundary_conditions>(F, dt);
        //field_update_standard<periodic_boundary_conditions>(F, dt) ;
    }
    const gvt* __restrict__ data_i = init.data;
    const gvt* __restrict__ data_e = F.A_n.data;
    index_range<3> interior = F.A_n.getRange();
    const auto mimap = F.A_n.m_imap;
    gvt sum = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(interior.volume()),
        [data_e, data_i, interior, mimap]KOKKOS_FUNCTION(size_t x) -> gvt{
            //return T(1);
            auto first_idx = interior.decompose(x);
            gvt init = data_i[mimap(first_idx[0], first_idx[1], first_idx[2])];
            gvt end =  data_e[mimap(first_idx[0], first_idx[1], first_idx[2])];
            return abs(init - end);
        },
        gvt(0),
        []KOKKOS_FUNCTION(gvt x, gvt y){
            return x + y;
    });
    //if(n == 128){
    //    grid<gvt, host> Ah = F.A_n.hostCopy();
    //    std::ofstream line("line.txt");
    //    for(uint32_t i = 1;i <= 128;i++){
    //        line << i * hr_m[2] << " " << Ah(64, 64, i) << "\n";
    //    }
    //    Ah.destroy();
    //}
    F.destroy();
    init.destroy();
    return scalar(0.5) * (sum * init.m_mesh.volume());
}

scalar test_nsfd(uint32_t n, ippl::Vector<scalar, 3> dirvec = ippl::Vector<scalar, 3>{scalar(0),scalar(0),scalar(1)}){
    
    using gvt = scalar;
    //n &= ~uint32_t(1);
    
    ippl::Vector<scalar, 3> origin{-0.5,-0.5,-0.5};
    ippl::Vector<scalar, 3> hr_m(1);
    hr_m /= ippl::Vector<scalar, 3>{scalar(n), scalar(n), scalar(2 * n)};
    field_state<gvt, device> F = make_field_state<gvt, device>(n, n, 2 * n, origin, hr_m);
    grid<gvt, device> init(n, n, 2 * n, F.A_n.m_mesh);

    const scalar variance = scalar(1) / 10;
    const scalar dt = scalar(1.0) ** std::min_element(hr_m.begin(), hr_m.end());
    assert(dt == hr_m[2]);
    parallel_for_cuda([=] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
        scalar x = dirvec[0] * ((scalar(i) + scalar(0.5)) * hr_m[0] + origin[0]);
        scalar y = dirvec[1] * ((scalar(j) + scalar(0.5)) * hr_m[1] + origin[1]);
        scalar z = dirvec[2] * ((scalar(k) + scalar(0.5)) * hr_m[2] + origin[2]);
        scalar value = gauss(scalar(0), variance, x, y, z);
        F.A_n(i, j, k) = (value);
        F.A_nm1(i, j, k) = (value);
        init(i, j, k) = (value);
        F.source_n(i,j,k) = 0;

    }, F.A_n.getFullRange());

    for(uint32_t i = 0;i < n;i++){
        field_update<periodic_boundary_conditions>(F, dt);
        field_update<periodic_boundary_conditions>(F, dt);
    }
    const gvt* __restrict__ data_i = init.data;
    const gvt* __restrict__ data_e = F.A_n.data;
    index_range<3> interior = F.A_n.getRange();
    const auto mimap = F.A_n.m_imap;
    gvt sum = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(interior.volume()),
        [data_e, data_i, interior, mimap]KOKKOS_FUNCTION(size_t x) -> gvt{
            //return T(1);
            auto first_idx = interior.decompose(x);
            gvt init = data_i[mimap(first_idx[0], first_idx[1], first_idx[2])];
            gvt end =  data_e[mimap(first_idx[0], first_idx[1], first_idx[2])];
            return abs(init - end);
        },
        gvt(0),
        []KOKKOS_FUNCTION(gvt x, gvt y){
            return x + y;
    });
    F.destroy();
    init.destroy();
    return scalar(0.5) * (sum * init.m_mesh.volume());
}
void test_nsfd(){
    std::ofstream results("test_nsfd.txt");
    struct entry{
        scalar dx;
        scalar error;
        scalar transverse_error;
        scalar runtime_in_ms;
    };
    std::vector<entry> disc_error;
    const std::array sizes = {20, 30, 40, 48, 64, 80, 96, 128, 192, 256};
    for(uint32_t n : sizes){
        auto t1 = nanoTime();
        scalar error = test_nsfd(n);
        cudaDeviceSynchronize();
        auto t2 = nanoTime();
        scalar error_transverse = test_nsfd(n, ippl::Vector<scalar, 3>{scalar(1),scalar(0),scalar(0)});
        disc_error.push_back(entry{scalar(1) / n, error, error_transverse, scalar(t2 - t1) / scalar(1e6)});
    }
    for(auto [h, error, transverse_error, runtime] : disc_error){
        results << h << ' ' << error << ' ' << transverse_error << ' ' << runtime << "\n";
    }
}
void test_fieldsolve(){
    std::ofstream results("test_fieldsolve_results.txt");
    struct entry{
        scalar dx;
        scalar error;
        scalar runtime_in_ms;
    };
    std::vector<entry> disc_error;
    const std::array sizes = {20, 30, 40, 48, 64, 80, 96, 128, 192, 256};
    for(uint32_t n : sizes){
        auto t1 = nanoTime();
        scalar error = test_fieldsolve(n);
        std::cerr << "tfs(" << n << ")\n";
        cudaDeviceSynchronize();
        auto t2 = nanoTime();
        disc_error.push_back(entry{scalar(1) / n, error, scalar(t2 - t1) / scalar(1e6)});
    }
    for(size_t i = 1; i < disc_error.size();i++){
        scalar dx_ratio = disc_error[i].dx / disc_error[i - 1].dx;
        scalar error_ratio = disc_error[i].error / disc_error[i - 1].error;
        if(std::log(error_ratio) / std::log(dx_ratio) < 2.8){
            //std::cerr << "Warning: No cubic convergence here, but order " << std::log(error_ratio) / std::log(dx_ratio) << "\n";
            //std::cerr << disc_error[i].dx << " and " << disc_error[i - 1].dx << "\n";
            //std::cerr << disc_error[i].error << " and " << disc_error[i - 1].error << "\n";
        }
        
    }
    for(auto [h, error, runtime] : disc_error){
        results << h << ' ' << error << ' ' << runtime << "\n";
    }
}
scalar test_abcs(uint32_t n){
    using gvt = ippl::Vector<scalar, 4>;
    using source_vt = ippl::Vector<scalar, 4>;
    using EB_vt = ippl::Vector<ippl::Vector<scalar, 3>, 2>;

    //n &= ~uint32_t(1);

    
    ippl::Vector<scalar, 3> origin{-0.5,-0.5,-0.5};
    ippl::Vector<scalar, 3> hr_m{1,1,1};
    uint32_t nz = n;
    hr_m /= ippl::Vector<scalar, 3>{scalar(n), scalar(n), scalar(nz)};
    field_state<gvt, device> F = make_field_state<gvt, device>(n, n, nz, origin, hr_m);
    //grid<gvt, device> init(n, n, nz, F.A_n.m_mesh);
    grid<EB_vt, device> EB(n, n, nz, F.A_n.m_mesh);
    //grid<EB_vt, device> EB    (n, n, 2 * n, m_mesh);
    const scalar stddev = scalar(1) / 15;
    const scalar dt = scalar(0.5) ** std::min_element(hr_m.begin(), hr_m.end());
    std::cerr << "Picked dt = " << dt << "\n";
    //assert(dt == hr_m[2]);
    parallel_for_cuda([=] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
        scalar x = ((scalar(i) + scalar(0.5)) * hr_m[0] + origin[0]);
        scalar y = ((scalar(j) + scalar(0.5)) * hr_m[1] + origin[1]);
        scalar z = ((scalar(k) + scalar(0.5)) * hr_m[2] + origin[2]);
        scalar value = gauss(scalar(0), stddev, x, y, z);
        F.A_np1(i, j, k).fill(0);
        F.A_n  (i, j, k).fill(0);
        F.A_nm1(i, j, k).fill(0);
        F.A_np1(i, j, k)[0] = value;
        F.A_n  (i, j, k)[0] = value;
        F.A_nm1(i, j, k)[0] = value;
        //init(i, j, k) = value;
        F.source_n(i, j, k).fill(0);

    }, F.A_n.getFullRange());
    
    for(uint32_t i = 0;i < uint32_t(std::ceil(1.7 * n));i++){
        //std::cout << "Updatinf field goddanit\n";
        field_update_standard<second_order_mur_boundary_conditions>(F, dt);
        //gpuErrchk(cudaDeviceSynchronize());
        //field_update_standard<first_order_mur_boundary_conditions>(F, dt);
        //gpuErrchk(cudaDeviceSynchronize());
        //field_update<false, do_nothing_boundary_conditions>(A, source, dt);
        //field_update<true , do_nothing_boundary_conditions>(A, source, dt);
    }
    evaluate_EB(F, EB, dt);
    gpuErrchk(cudaDeviceSynchronize());
    scalar energy = EB.volumeIntegral_energy(); 
    //EB.volumeIntegral([]KOKKOS_FUNCTION(EB_vt x) -> scalar{
    //    return x[0].squaredNorm() + x[1].squaredNorm();
    //});
    //if(n == 128){
    //    std::ofstream line("line.txt");
    //    grid<EB_vt, host> EBh = EB.hostCopy();
    //    for(int i = 1;i < 511;i++){
    //        line << i * hr_m[2] << " " << EBh(64, 64, i)[0] << "\n";
    //    }
    //    EBh.destroy();
    //}
    //scalar Asum = F.A_n.volumeIntegral().sum();
    F.destroy();
    //init.destroy();
    EB.destroy();

    return energy;
}
scalar test_gauss_law(uint32_t n){
    particles<scalar, device> bunch(1 << 20);
    bunch.charge_per_particle = scalar(coulomb_in_unit_charges) / bunch.count_m;
    ippl::Vector<scalar, 3> origin{-0.5 * meter_in_unit_lengths,-0.5 * meter_in_unit_lengths,-0.5 * meter_in_unit_lengths};
    ippl::Vector<scalar, 3> hr_m(meter_in_unit_lengths);
    //constexpr scalar vy = meter_in_unit_lengths / second_in_unit_times;
    hr_m /= ippl::Vector<scalar, 3>{scalar(n), scalar(n), scalar(n)};
    scalar dt = 0.5 ** std::min_element(hr_m.begin(), hr_m.end());
    
    parallel_for_cuda([=]KOKKOS_FUNCTION(uint32_t i){
        //bunch.gammaBeta[i].fill(scalar(0));
        XORShift64 gen(i);
        gen.scramble(0xffafdfadf111);

        ippl::Vector<scalar, 3> pos;//, ppos;
        pos.fill(0);
        pos[0] = convertToNormal(gen.next<scalar>(), scalar(1)) * scalar(0.01 * meter_in_unit_lengths);
        pos[1] = convertToNormal(gen.next<scalar>(), scalar(1)) * scalar(0.01 * meter_in_unit_lengths);
        pos[2] = convertToNormal(gen.next<scalar>(), scalar(1)) * scalar(0.01 * meter_in_unit_lengths);
        //ppos.fill(0);
        //pos[1] = origin[1] +  meter_in_unit_lengths * scalar(i) / (bunch.count_m - 1);
        //ppos[1] = origin[1] + meter_in_unit_lengths * scalar(i) / (bunch.count_m - 1);
        bunch.positions[i] = pos;
        bunch.previous_positions[i] = pos;
    }, bunch.getRange());
    cudaDeviceSynchronize();
    using gvt = ippl::Vector<ippl::Vector<scalar, 4>, 2>;
    using source_vt = ippl::Vector<scalar, 4>;
    using EB_vt = ippl::Vector<ippl::Vector<scalar, 3>, 2>;
    
    
    mesh<scalar> m_mesh;
    m_mesh.origin = origin;
    m_mesh.spacing = hr_m;
    m_mesh.inverse_spacing = ippl::Vector<scalar, 3>(1) / hr_m;
    field_state<ippl::Vector<scalar, 4>, device> F = make_field_state<ippl::Vector<scalar, 4>, device>(n, n, n, origin, hr_m);
    //grid<source_vt, device> source(n, n, n, m_mesh);
    grid<EB_vt, device> EB(n, n, n, m_mesh);
    F.A_n.setZero();
    F.A_nm1.setZero();
    F.A_np1.setZero();
    //grid<EB_vt, device> EB    (n, n, 2 * n, m_mesh);
    //source.setZero();
    //EB.setZero();
    bunch.scatter(F.source_n, scalar(1 /*should be irrelevant*/));
    bunch.scatterCurrent(F.source_n, dt);
    for(unsigned int i = 0;i < 4 * n;i++){
        field_update_standard<second_order_mur_boundary_conditions>(F, dt);
    }
    evaluate_EB(F, EB, dt);
    {
        std::ofstream line("gauss_line.txt");

        grid<EB_vt, host> EBh = EB.hostCopy();
        //grid<source_vt, host> Sh = F.source_n.hostCopy();
        //grid<source_vt, host> Ah = F.A_n.hostCopy();
        for(uint32_t i = 1;i < n - 1;i++){
            auto p = EB.m_mesh.positionOf(ippl::Vector<uint32_t, 3>{i, n / 2, n / 2});
            //std::cout << p << "\n";
            //This is the Z component of B
            //line << p[0] * unit_length_in_meters << " " << (Ah(i, ny / 2, n / 2)[2]) << "\n";
            line << p.norm() * unit_length_in_meters << " " << (EBh(i, n / 2, n / 2)[0].norm()) * unit_electric_fieldstrength_in_voltpermeters << "\n";
            //line << p[0] * unit_length_in_meters << " " << (Sh(i, n / 2, n / 2)[0]) * unit_electric_fieldstrength_in_voltpermeters << "\n";
        }
        ippl::Vector<double, 3> e_at_point = gatherHost(EBh, ippl::Vector<scalar, 3>{0.07 * meter_in_unit_lengths, 0.07 * meter_in_unit_lengths, 0.07 * meter_in_unit_lengths})[0].cast<double>();
        e_at_point *= coulomb_in_unit_charges;
        std::cout << "Force: " << e_at_point.norm() * unit_force_in_newtons << "\n";
        std::cout << "Expected force: " << 1.0 / (4 * M_PI * 8.854e-12 * sq(0.07 * std::sqrt(3))) << "\n";
        EBh.destroy();
        //Sh.destroy();
        //Ah.destroy();
    }
    bunch.destroy();
    EB.destroy();
    F.destroy();

    return 0;
}

scalar test_amperes_law(uint32_t n){
    particles<scalar, device> bunch(1 << 24);
    bunch.charge_per_particle = scalar(4 * coulomb_in_unit_charges) / bunch.count_m;
    ippl::Vector<scalar, 3> origin{-0.5 * meter_in_unit_lengths,-2.0 * meter_in_unit_lengths,-0.5 * meter_in_unit_lengths};
    ippl::Vector<scalar, 3> hr_m{meter_in_unit_lengths, 4.0 * meter_in_unit_lengths, meter_in_unit_lengths};
    constexpr scalar vy = meter_in_unit_lengths / second_in_unit_times;
    uint32_t ny = n * 4;
    hr_m /= ippl::Vector<scalar, 3>{scalar(n), scalar(ny), scalar(n)};
    scalar dt = 0.5 ** std::min_element(hr_m.begin(), hr_m.end());
    parallel_for_cuda([=]KOKKOS_FUNCTION(uint32_t i){
        //bunch.gammaBeta[i].fill(scalar(0));
        XORShift64 gen(i);
        gen.scramble(0xffafdfadf111);

        ippl::Vector<scalar, 3> pos, ppos;
        pos.fill(0);
        ppos.fill(0);
        pos[0] = convertToNormal(gen.next<scalar>(), 1) * 0.01 * meter_in_unit_lengths;
        pos[2] = convertToNormal(gen.next<scalar>(), 1) * 0.01 * meter_in_unit_lengths;
        ppos = pos;
        pos[1] = origin[1] +  4.0 * meter_in_unit_lengths * scalar(i) / (bunch.count_m - 1);
        ppos[1] = origin[1] + 4.0 * meter_in_unit_lengths * scalar(i) / (bunch.count_m - 1) - vy * dt;
        bunch.positions[i] = pos;
        bunch.previous_positions[i] = ppos;
    }, bunch.getRange());
    cudaDeviceSynchronize();
    using gvt = ippl::Vector<ippl::Vector<scalar, 4>, 2>;
    using source_vt = ippl::Vector<scalar, 4>;
    using EB_vt = ippl::Vector<ippl::Vector<scalar, 3>, 2>;
    
    
    mesh<scalar> m_mesh;
    m_mesh.origin = origin;
    m_mesh.spacing = hr_m;
    m_mesh.inverse_spacing = ippl::Vector<scalar, 3>(1) / hr_m;
    field_state<ippl::Vector<scalar, 4>, device> F = make_field_state<ippl::Vector<scalar, 4>, device>(n, ny, n, origin, hr_m);
    //grid<source_vt, device> source(n, n, n, m_mesh);
    grid<EB_vt, device> EB(n, ny, n, m_mesh);
    F.A_n.setZero();
    F.A_nm1.setZero();
    F.A_np1.setZero();
    //grid<EB_vt, device> EB    (n, n, 2 * n, m_mesh);
    //source.setZero();
    //EB.setZero();
    bunch.scatter(F.source_n, scalar(1.0));
    bunch.scatterCurrent(F.source_n, dt);
    for(unsigned int i = 0;i < 8 * n;i++){
        field_update_standard<second_order_mur_boundary_conditions>(F, dt);
    }
    evaluate_EB(F, EB, dt);
    //if(n == 200)
    {
        std::ofstream line("ampere_line.txt");

        grid<EB_vt, host> EBh = EB.hostCopy();
        //grid<source_vt, host> Ah = F.A_n.hostCopy();
        for(uint32_t i = 1;i < n - 1;i++){
            auto p = EB.m_mesh.positionOf(ippl::Vector<uint32_t, 3>{i, ny / 2, n / 2});
            //std::cout << p << "\n";
            //This is the Z component of B
            line << p[0] * unit_length_in_meters << " " << (EBh(i, ny / 2, n / 2)[1][2]) * unit_magnetic_fluxdensity_in_tesla << "\n";
            //line << p[0] * unit_length_in_meters << " " << (Ah(i, ny / 2, n / 2)[2]) << "\n";
        }
        ippl::Vector<double, 3> e_at_point = gatherHost(EBh, ippl::Vector<scalar, 3>{0.07 * meter_in_unit_lengths, 0.07 * meter_in_unit_lengths, 0.07 * meter_in_unit_lengths})[0].cast<double>();
        ippl::Vector<double, 3> b_at_point = gatherHost(EBh, ippl::Vector<scalar, 3>{0.07 * meter_in_unit_lengths, 0.07 * meter_in_unit_lengths, 0.07 * meter_in_unit_lengths})[1].cast<double>();
        
        e_at_point *= coulomb_in_unit_charges;
        b_at_point *= coulomb_in_unit_charges;

        //std::cout << "test_amperes_law::Electric Force: " << e_at_point.norm() * unit_force_in_newtons << "\n";
        std::cout << "test_amperes_law::Magnetic Force: " << ippl::Vector<double, 3>{0, 1, 0}.cross(b_at_point).norm() * unit_force_in_newtons << " N\n";
        std::cout << "Expected magnetic force: " << (4 * M_PI * 1e-7) / (2 * M_PI * (0.07 * std::sqrt(2))) * (1/*Coulomb^2*/ * 1) * 299792458 << " N\n";
        EBh.destroy();
        //Ah.destroy();
    }
    F.destroy();
    //source.destroy();
    EB.destroy();
    bunch.destroy();
    return 0;
}
void test_radiation_reaction(uint32_t n){
    particles<scalar, device> bunch(1 << 23);
    //particles<scalar, host> hbunch(1 << 16);
    bunch.charge_per_particle = scalar(coulomb_in_unit_charges) / bunch.count_m;
    constexpr scalar extents_in_meters = 3.0;
    constexpr scalar measurement_radius = 1.25;
    ippl::Vector<scalar, 3> origin{-(extents_in_meters / 2) * meter_in_unit_lengths, -(extents_in_meters / 2) * meter_in_unit_lengths, -(extents_in_meters / 2) * meter_in_unit_lengths};
    ippl::Vector<scalar, 3> hr_m(extents_in_meters * meter_in_unit_lengths);
    hr_m /= ippl::Vector<scalar, 3>{scalar(n), scalar(n), scalar(n)};
    scalar dt = 0.5 ** std::min_element(hr_m.begin(), hr_m.end());
    constexpr scalar beta = 0.7;
    parallel_for_cuda([=]KOKKOS_FUNCTION(uint32_t i){
        //bunch.gammaBeta[i].fill(scalar(0));
        XORShift64 gen(i);
        gen.scramble(0xffafdfadf111 * 17 + i * 13);

        ippl::Vector<scalar, 3> pos, ppos;
        pos.fill(0) ;
        ppos.fill(0);
        pos[0] = meter_in_unit_lengths + convertToNormal(gen.next<scalar>(), scalar(0.005 * meter_in_unit_lengths));
        pos[0] = min(pos[0], meter_in_unit_lengths / beta);
        pos[1] = convertToNormal(gen.next<scalar>(),                         scalar(0.005 * meter_in_unit_lengths));
        pos[2] = convertToNormal(gen.next<scalar>(),                         scalar(0.005 * meter_in_unit_lengths));
        ppos = pos;
        bunch.positions[i] = pos;
        bunch.previous_positions[i] = ppos;
    }, bunch.getRange());
    cudaDeviceSynchronize();
    using gvt = ippl::Vector<ippl::Vector<scalar, 4>, 2>;
    using source_vt = ippl::Vector<scalar, 4>;
    using EB_vt = ippl::Vector<ippl::Vector<scalar, 3>, 2>;
    
    
    mesh<scalar> m_mesh;
    m_mesh.origin = origin;
    m_mesh.spacing = hr_m;
    m_mesh.inverse_spacing = ippl::Vector<scalar, 3>(1) / hr_m;
    field_state<ippl::Vector<scalar, 4>, device> F = make_field_state<ippl::Vector<scalar, 4>, device>(n, n, n, origin, hr_m);
    //grid<source_vt, device> source(n, n, n, m_mesh);
    grid<EB_vt, device> EB(n, n, n, m_mesh);
    F.A_n.setZero();
    F.A_nm1.setZero();
    F.A_np1.setZero();
    //std::cout << "Radiation test";
    std::ofstream radline("orbit_rad.txt");
    for(uint32_t i = 0;i < 15 * n;i++){
        //std::cout << "\rStep " << i << "                " << std::endl;
        const scalar angle = beta / meter_in_unit_lengths * dt;
        parallel_for_cuda([=]KOKKOS_FUNCTION(uint32_t i){
            ippl::Vector<scalar, 3> pos = bunch.positions[i];
            bunch.previous_positions[i] = pos;
            scalar newx = pos[0] * cos(angle) - pos[1] * sin(angle);
            scalar newy = pos[0] * sin(angle) + pos[1] * cos(angle);
            pos[0] = newx;
            pos[1] = newy;
            bunch.positions[i] = pos;
            if(i == 0){
                //printf("x, y, vel: %.3e, %.3e, %.3e\n", pos[0], pos[1], (pos - bunch.previous_positions[i]).norm() / dt);
            }
            //assert(false);
            assert((pos - bunch.previous_positions[i]).norm() / dt < 1);
        }, bunch.getRange());
        F.source_n.setZero();
        bunch.scatter(F.source_n, dt);
        bunch.scatterCurrent(F.source_n, dt);
        field_update_standard<second_order_mur_boundary_conditions>(F, dt);
        std::cout << F.source_n.volumeIntegral()[0] / coulomb_in_unit_charges << " C and ";
        std::cout << F.source_n.volumeIntegral().tail<3>() * unit_current_length_in_ampere_meters << " Am\n";
        evaluate_EB(F, EB, dt);
        //EB_vt ebv;
        const size_t root_sample_count = 10000;
        double radint = thrust::transform_reduce(thrust::make_counting_iterator(size_t(0)), thrust::make_counting_iterator(root_sample_count * root_sample_count),
        [=]__device__(size_t i){
            size_t xi = i % root_sample_count;
            size_t yi = i / root_sample_count;
            scalar phi = (2 * M_PI) * scalar(xi) / root_sample_count;
            scalar theta = acos(2 * scalar(yi + 1) / (root_sample_count + 2) - 1);
            ippl::Vector<double, 3> pos{
                double(cos(phi) * sin(theta)),  
                double(sin(phi) * sin(theta)),  
                double(cos(theta)),
            };
            ippl::Vector<double, 3> normal = pos;
            pos *= scalar(measurement_radius * meter_in_unit_lengths);
            EB_vt gatherval = gather(EB, pos.cast<scalar>());
            return normal.dot(gatherval[0].cast<double>().cross(gatherval[1].cast<double>()));
        },
        double(0),
        []KOKKOS_FUNCTION(double x, double y){
            return x + y;
        });
        
        //cudaMemcpy(&ebv, EB.data + EB.imap(EB.m_mesh.gridCoordinatesOf(ippl::Vector<scalar, 3>{0,1.8 * meter_in_unit_lengths,0}).first.cast<uint32_t>()), sizeof(EB_vt), cudaMemcpyDeviceToHost);
        //radline << i * dt * unit_time_in_seconds << " " << ebv[0].cross(ebv[1])[2] * unit_magnetic_fluxdensity_in_tesla * unit_electric_fieldstrength_in_voltpermeters << std::endl;
        radline << i * dt * unit_time_in_seconds << " " << (4 * M_PI * measurement_radius * measurement_radius) * (radint * unit_powerdensity_in_watt_per_square_meter) / (root_sample_count * root_sample_count) << " " << bunch.covariance_matrix().first.head<3>() << std::endl;

        if(i % 4 == 7){
            using vec3 = rm::Vector<float, 3>;
            ClearFrame();
            rc.clear();
            glDisable(GL_CULL_FACE);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            LookAt(vec3{0.0, 0.0, scalar(2.0)}, vec3{0, 0, 0});
            //std::cout << rc.cam.look_dir() << "\n";
            //DrawText("klonk", 0, 100, 1, 1, 1, 1);
            //bunch.update_hostcopy<false>(hbunch);
            //Xoshiro128 pgen(332422211111732ul);
            //for(size_t pi = 0;pi < hbunch.count_m;pi++){
            //    ippl::Vector<scalar, 3> pos = hbunch.positions[pi] / float(cfg.extents[1]);
            //    //if((pgen() & 15) == 0){
            //        rc.draw_sphere(sphere_info{
            //            .pos = vec3{(float)pos[0], (float)pos[1], (float)pos[2]},
            //            .color = vec3{0,1,0},
            //            .radius = 0.0005f
            //        });
            //    //}
            //}
            //Xoshiro128 gen(132412341);
            grid<EB_vt, host> ebh = EB.hostCopy();
            scalar maxnorm = -1;
            serial_for<3>([&](size_t i, size_t j, size_t k){
                if(((i & 3) | (j & 3) | (k & 3)) == 0){
                    ippl::Vector<scalar, 3> poynting = ebh(i,j,k)[0].cross(ebh(i,j,k)[1]) * 1e-18;
                    if(poynting.norm() > 0.3){
                        scalar fac = 0.3 / poynting.norm();
                        poynting *= fac;
                    }
                    int acc = poynting.norm() * 2550;
                    acc = std::max(std::min(acc, 255), 0);
                    float fillr = turbo_cm[acc][0];
                    float fillg = turbo_cm[acc][1];
                    float fillb = turbo_cm[acc][2];
                    //if(poynting.norm() > 0.01f){
                    //    poynting /= (100.0f * poynting.norm());
                    //}
                    ippl::Vector<scalar, 3> pos = ebh.m_mesh.positionOf(ippl::Vector<size_t, 3>{i, j, k}) / scalar(EB.m_mesh.spacing[1] * EB.m_imap.n);
                    maxnorm = std::max(poynting.norm(), maxnorm);
                    //std::cout << pos << "\n";
                    rc.draw_line(line_info{
                        .from = vec3{(float)pos[0], (float)pos[1], (float)pos[2]},
                        .fcolor = vec3{fillr,fillg,fillb},
                        .to = vec3{(float)(pos[0] + poynting[0]), (float)(pos[1] + poynting[1]), (float)(pos[2] + poynting[2])},
                        .tcolor = vec3{fillr,fillg,fillb},
                    });
                }
            }, ebh.m_imap.getRange().begin, ebh.m_imap.getRange().end);
            //std::cout << maxnorm << " maxn" << std::endl;
            rc.draw();

        
            ebh.destroy();
            const unsigned width = 3840, height = 2160;
            std::vector<unsigned char> pixels(3 * width * height, 0);
            //std::fill(imagedata, imagedata + img_width * img_height * 3, 0);
            glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
            for(int _i = 0;_i < height / 2;_i++){
                for(int _j = 0;_j < width;_j++){
                    int i1 = (_i * width + _j) * 3;
                    int i2 = ((height - _i - 1) * width + _j) * 3;
                    //{
                    //    pixels[i1+0] = (unsigned char)(pixels[i2+0] * 255);
                    //    pixels[i1+1] = (unsigned char)(pixels[i2+1] * 255);
                    //    pixels[i1+2] = (unsigned char)(pixels[i2+2] * 255);
                    //}
                    //pixels[i1] = (unsigned char)(dpixels[i1 / 3] * 255);
                    //pixels[i1+1] = (unsigned char)(dpixels[i1 / 3] * 255);
                    //pixels[i1+2] = (unsigned char)(dpixels[i1 / 3] * 255);
                    std::iter_swap(pixels.begin() + i1 + 0, pixels.begin() + i2 + 0);
                    std::iter_swap(pixels.begin() + i1 + 1, pixels.begin() + i2 + 1);
                    std::iter_swap(pixels.begin() + i1 + 2, pixels.begin() + i2 + 2);
                }
            }
            char output[1024] = {0};
            snprintf(output, 1023, "../data/outimage%.05d.bmp", i);
            //std::transform(imagedata, imagedata + img_height * img_width * 3, imagedata_final, [](float x){return (unsigned char)x;});
            //stbi_write_bmp(output, img_width, img_height, 3, pixels.data());
            stbi_write_bmp(output, width, height, 3, pixels.data());
        }
    }
}
void test_amperes_law(){
    std::ofstream results("test_amperes_results.txt");
    struct entry{
        scalar dx;
        scalar error;
        scalar runtime_in_ms;
    };
    std::vector<entry> disc_error;
    const std::array sizes = {100};
    for(uint32_t n : sizes){
        auto t1 = nanoTime();
        scalar error = test_amperes_law(n);
        cudaDeviceSynchronize();
        auto t2 = nanoTime();
        disc_error.push_back(entry{scalar(1) / n, error, scalar(t2 - t1) / scalar(1e6)});
    }
    for(auto [h, error, runtime] : disc_error){
        results << h << ' ' << error << ' ' << runtime << "\n";
    }
}
void test_abcs(){
    std::ofstream results("test_abcs_results.txt");
    struct entry{
        scalar dx;
        scalar error;
        scalar runtime_in_ms;
    };
    std::vector<entry> disc_error;
    const std::array sizes = {24, 32, 40, 48, 64, 80, 96, 128, 192, 256};
    for(uint32_t n : sizes){
        auto t1 = nanoTime();
        scalar error = test_abcs(n);
        cudaDeviceSynchronize();
        auto t2 = nanoTime();
        disc_error.push_back(entry{scalar(1) / n, error, scalar(t2 - t1) / scalar(1e6)});
    }
    for(size_t i = 1; i < disc_error.size();i++){
        scalar dx_ratio = disc_error[i].dx / disc_error[i - 1].dx;
        scalar error_ratio = disc_error[i].error / disc_error[i - 1].error;
        if(std::log(error_ratio) / std::log(dx_ratio) < 2.8){
            //std::cerr << "Warning: No cubic convergence here, but order " << std::log(error_ratio) / std::log(dx_ratio) << "\n";
            //std::cerr << disc_error[i].dx << " and " << disc_error[i - 1].dx << "\n";
            //std::cerr << disc_error[i].error << " and " << disc_error[i - 1].error << "\n";
        }
        
    }
    for(auto [h, error, runtime] : disc_error){
        results << h << ' ' << error << ' ' << runtime << "\n";
    }
}
void test_bunch_operations(uint32_t n = 128){
    particles<scalar, device> bunch(1 << 20);
    ippl::Vector<scalar, 3> origin{-0.5,-0.5,-0.5};
    parallel_for_cuda([=]KOKKOS_FUNCTION(uint32_t i){
        bunch.gammaBeta[i].fill(scalar(0));
        XORShift64 gen(i);
        gen.scramble(0xffafdfadf111);
        ippl::Vector<scalar, 3> pos, ppos;
        for(uint32_t d = 0;d < 3;d++){
            scalar p = origin[d] + gen.next<scalar>() * scalar(0.8) + scalar(0.1);
            pos[d]  = p;
            ppos[d] = p - scalar(d == 1) * scalar(0.001);
        }
        bunch.positions[i] = pos;
        bunch.previous_positions[i] = ppos;
    }, bunch.getRange());
    cudaDeviceSynchronize();
    using gvt = ippl::Vector<ippl::Vector<scalar, 4>, 2>;
    using source_vt = ippl::Vector<scalar, 4>;
    using EB_vt = ippl::Vector<ippl::Vector<scalar, 3>, 2>;
    ippl::Vector<scalar, 3> hr_m(1);
    hr_m /= ippl::Vector<scalar, 3>{scalar(n), scalar(n), scalar(2 * n)};
    mesh<scalar> m_mesh;
    m_mesh.origin = origin;
    m_mesh.spacing = hr_m;
    m_mesh.inverse_spacing = ippl::Vector<scalar, 3>(1) / hr_m;
    grid<gvt, device> A     (n, n, 2 * n, m_mesh);
    grid<source_vt, device> source(n, n, 2 * n, m_mesh);
    //grid<EB_vt, device> EB    (n, n, 2 * n, m_mesh);
    A.setZero();
    source.setZero();
    //EB.setZero();
    bunch.scatter(source, scalar(1.0));
    bunch.scatterCurrent(source, scalar(1.0));
    source_vt source_int = source.volumeIntegral();
    //std::cerr << source_int[0] << ", " <<  bunch.count_m * bunch.charge_per_particle << "\n";
    std::cout << "Scattered current: " << source_int.tail<3>() << " vs " <<ippl::Vector<scalar, 3>{scalar(0),scalar(0.001),scalar(0)} * bunch.count_m * bunch.charge_per_particle << "\n";
    std::cout << "Scattered charge: " << source_int[0] << " vs " << bunch.count_m * bunch.charge_per_particle << "\n";
    //assert(std::abs(source_int[0] - bunch.count_m * bunch.charge_per_particle) < 1e-5);
    //assert(std::abs((source_int.tail<3>() - ippl::Vector<scalar, 3>{scalar(0),scalar(0.01),scalar(0)} * bunch.count_m * bunch.charge_per_particle).average()) < 1e-5);
    A.destroy();
    source.destroy();
    //EB.destroy();
    bunch.destroy();
}
std::pair<double, double> bench_scatter_gather(uint32_t n){
    particles<scalar, device> bunch(n * n * n * 4);
    bunch.charge_per_particle = 1.0f;
    ippl::Vector<scalar, 3> origin{-0.5,-0.5,-0.5};
    ippl::Vector<scalar, 3> hr_m(1);
    hr_m /= ippl::Vector<scalar, 3>{scalar(n), scalar(n), scalar(n)};
    field_state<ippl::Vector<scalar, 4>, device> F = make_field_state<ippl::Vector<scalar, 4>, device>(n,n,n, origin, hr_m);
    //grid<ippl::Vector<ippl::Vector<scalar, 3>, 2>, device> EB(n, n, n);
    //EB.setZero();
    parallel_for_cuda([=]KOKKOS_FUNCTION(uint32_t i){
        bunch.gammaBeta[i].fill(scalar(0));
        XORShift64 gen(i);
        gen.scramble(0xffafdfadf111);
        ippl::Vector<scalar, 3> pos, ppos;
        for(uint32_t d = 0;d < 3;d++){
            scalar p = origin[d] + gen.next<scalar>() * scalar(0.8) + scalar(0.1);
            pos[d]  = p;
            ppos[d] = p;
            
        }
        bunch.positions[i] = pos;
        bunch.previous_positions[i] = ppos;
        bunch.gammaBeta[i] = ippl::Vector<scalar, 3>(0);
    }, bunch.getRange());
    auto undulator_field = []KOKKOS_FUNCTION(const ippl::Vector<scalar, 3>& position_in_lab_frame){
        return pear<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>>{ippl::Vector<scalar, 3>(0), ippl::Vector<scalar, 3>(0)};
    };
    auto truu = []KOKKOS_FUNCTION(const ippl::Vector<scalar, 3>& particle_labpos){
        return true;
    };
    LorentzFrame<scalar> lf(ippl::Vector<scalar, 3>{0.0f,0.0f,1.0f});
    auto t1 = nanoTime();
    for(int i = 0;i < 10;i++){
        //bunch.update(EB, 0.1f, 0.0f, lf, undulator_field, truu, 10);
        bunch.scatter(F.source_n, 1.0f);
    }
    cudaDeviceSynchronize();
    auto t2 = nanoTime();
    double unsorted = double(t2 - t1);
    using bvec3 = decltype(bunch)::vec3;
            thrust::device_ptr<bvec3> bpp(bunch.previous_positions), bp(bunch.positions), bgb(bunch.gammaBeta);
            thrust::sort(
            thrust::make_zip_iterator(bpp, bp, bgb),
            thrust::make_zip_iterator(bpp + bunch.count_m, bp + bunch.count_m, bgb + bunch.count_m),
            [F]KOKKOS_FUNCTION(const thrust::tuple<bvec3, bvec3, bvec3>& x, const thrust::tuple<bvec3, bvec3, bvec3>& y){
                bvec3 px = thrust::get<1>(x);
                bvec3 py = thrust::get<1>(y);
                pear<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> pxi = F.A_n.m_mesh.gridCoordinatesOf(px);
                pear<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> pyi = F.A_n.m_mesh.gridCoordinatesOf(py);
                size_t xi1d = F.A_n.m_imap(pxi.first.cast<uint32_t>());
                size_t yi1d = F.A_n.m_imap(pyi.first.cast<uint32_t>());
                return xi1d < yi1d;
            });
    t1 = nanoTime();
    for(int i = 0;i < 10;i++){
        //bunch.update(EB, 0.1f, 0.0f, lf, undulator_field, truu, 10);
        bunch.scatter(F.source_n, 1.0f);
    }
    cudaDeviceSynchronize();
    t2 = nanoTime();
    double sorted = double(t2 - t1);
    F.destroy();
    bunch.destroy();
    //EB.destroy();
    return {sorted, unsorted};
}
template<typename scalar>
void draw_bunch(const particles<scalar, host>& hbunch, scalar scale){
    using vec3 = rm::Vector<float, 3>;
    //bunch.update_hostcopy<false>(hbunch);
    Xoshiro128 pgen(332422211111732ul);
    for(size_t pi = 0;pi < hbunch.count_m;pi++){
        ippl::Vector<scalar, 3> pos = hbunch.positions[pi] * float(scale);
        //if((pgen() & 15) == 0){
            rc.draw_sphere(sphere_info{
                .pos = vec3{(float)pos[0], (float)pos[1], (float)pos[2]},
                .color = vec3{0,1,0},
                .radius = 0.0005f
            });
        //}
    }
    
}
template<typename scalar>
void draw_field(const grid<ippl::Vector<ippl::Vector<scalar, 3>, 2>, device>& EB, scalar scale){
    using EB_vt = ippl::Vector<ippl::Vector<scalar, 3>, 2>;
    using vec3 = rm::Vector<float, 3>;
    Xoshiro128 gen(132412341);
    grid<EB_vt, host> ebh = EB.hostCopy();
    serial_for<3>([&](size_t i, size_t j, size_t k){
        if((gen() & 255) == 0){
            ippl::Vector<scalar, 3> poynting = ebh(i,j,k)[0].cross(ebh(i,j,k)[1]) * 0.00001f;
            if(poynting.norm() > 0.01f){
                poynting /= (100.0f * poynting.norm());
            }
            ippl::Vector<scalar, 3> pos = ebh.m_mesh.positionOf(ippl::Vector<size_t, 3>{i, j, k}) * scale;
            //std::cout << pos << "\n";
            rc.draw_line(line_info{
                .from = vec3{(float)pos[0], (float)pos[1], (float)pos[2]},
                .fcolor = vec3{1,0,0},
                .to = vec3{(float)(pos[0] + poynting[0]), (float)(pos[1] + poynting[1]), (float)(pos[2] + poynting[2])},
                .tcolor = vec3{1,0,0},
            });
        }
    }, ebh.m_imap.getRange().begin, ebh.m_imap.getRange().end);
    rc.draw();
    ebh.destroy();
    
}
int main(int argc, char** argv){
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    //test_bunch_operations();
    //test_fieldsolve();
    //test_nsfd();
    //test_abcs();
    //teschtrand();
    //test_amperes_law();
    std::ofstream scatterbench("scatterbench.txt");
    for(unsigned n = 64;n < 256; n += 32){
        auto [sorted, unsorted] = bench_scatter_gather(n);
        scatterbench << n << " " << unsorted / 10.0 << " " << sorted / 10.0 << std::endl;
    }
    //test_radiation_reaction(350);
    //test_gauss_law(150);
    //return 0;
    #define klonk_the_rest
    #ifndef klonk_the_rest
    const unsigned width = 3840, height = 2160;
    //egl_config econfig = egl_default_config();
    //config.samples = 0;
    //config.surface_type = EGL_PIXMAP_BIT;
    //load_context(width, height, econfig);
    //rc.init(width, height); //Yikes
    //rc.load_default_font();
    using gvt = ippl::Vector<scalar, 4>;
    using source_vt = ippl::Vector<scalar, 4>;
    using EB_vt = ippl::Vector<ippl::Vector<scalar, 3>, 2>;
    //cfg.extents is not gamma-corrected here, but will be later down
    config cfg = read_config("../config.json");
    const scalar frame_gamma = std::max(decltype(cfg)::scalar(1), cfg.bunch_gamma / std::sqrt(1.0 + cfg.undulator_K * cfg.undulator_K * config::scalar(0.5)));
    cfg.extents[2] *= frame_gamma;
    const scalar frame_beta = std::sqrt(1.0 - 1.0 / (frame_gamma * frame_gamma));
    assert_isreal(frame_gamma);
    assert_isreal(frame_beta);
    LorentzFrame<scalar> frame_boost = LorentzFrame<scalar>::uniaxialGamma<'z'>(frame_gamma);
    particles<scalar, host> mithrabunch;
    //std::ofstream initialpos("initialpos.txt");
    //std::cerr << "Something\n";
    //std::cerr << cfg.total_time << "\n";
    
    grid<EB_vt, device> EB(cfg.resolution[0], cfg.resolution[1], cfg.resolution[2]);
    grid<source_vt, device> source(cfg.resolution[0], cfg.resolution[1], cfg.resolution[2]);
    //std::cout << "K " << cfg.undulator_K << "\n";
    const undulator_parameters<scalar> uparams(cfg.undulator_K, cfg.undulator_period, cfg.undulator_length);
    
    
    const scalar k_u                 = scalar(2.0 * M_PI) / uparams.lambda;

    //Lab frame distance to undulator entry
    const scalar distance_to_entry   = std::max(0.0 * uparams.lambda, 2.0 * cfg.sigma_position[2] * frame_gamma * frame_gamma);
    std::cout << "Distance to entry: " << distance_to_entry << "\n";
    auto undulator_field = [uparams, k_u, distance_to_entry]KOKKOS_FUNCTION(const ippl::Vector<scalar, 3>& position_in_lab_frame){
        pear<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> ret;
        //printf("Trigo arg: %.4e\n", k_u * position_in_lab_frame[2]);
        //printf("Y = %f * %f\n", k_u, position_in_lab_frame[2]);
        //printf("cosharg: %f\n", k_u * position_in_lab_frame[1]);
        //printf("D: %f\n", distance_to_entry);
        //printf("bmag: %.3e\n", uparams.B_magnitude);
        ret.first.fill(0);
        ret.second.fill(0);
        if(position_in_lab_frame[2] < distance_to_entry){
            //assert(false);
            scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
            assert(z_in_undulator < 0);
            scalar scal = exp(-((k_u * z_in_undulator) * (k_u * z_in_undulator) * 0.5));
            //printf("Discard: %.4e\n", scal);
            ret.second[0] = 0;
            ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1]) * z_in_undulator * k_u * scal;
            ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1]) * scal;
        }
        else if(position_in_lab_frame[2] > distance_to_entry && position_in_lab_frame[2] < distance_to_entry + uparams.length){
            scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
            assert(z_in_undulator >= 0);
            ret.second[0] = 0;
            ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1]) * sin(k_u * z_in_undulator);
            ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1]) * cos(k_u * z_in_undulator);
        }
        //if(ret.second.squaredNorm() > 0){
        //    printf("Returning: %.3e, %.3e, %.3e, Bmag: %.3e\n", ret.second[0], ret.second[1], ret.second[2], uparams.B_magnitude);
        //}
        return ret;

    };

    //cudaMemset(gr.data, 0, gr.m_imap.size() * sizeof(float));
    
    //std::cout << A.getRange().begin[0] << "\n";
    //std::cout << A.getRange().begin[1] << "\n";
    //std::cout << A.getRange().begin[2] << "\n";
    //std::cout << A.getRange().end[0] << "\n";
    //std::cout << A.getRange().end[1] << "\n";
    //std::cout << A.getRange().end[2] << "\n\n";
    cudaDeviceSynchronize();
    
    auto t1 = nanoTime();
    ippl::Vector<uint32_t, 3> nr_m{(uint32_t)cfg.resolution[0], (uint32_t)cfg.resolution[1], (uint32_t)cfg.resolution[2]};
    ippl::Vector<scalar, 3>   hr_m{(scalar)cfg.extents [0] / nr_m[0], (scalar)cfg.extents[1] / nr_m[1], (scalar)cfg.extents[2] / nr_m[2]};
    ippl::Vector<scalar, 3> origin{-(scalar)cfg.extents[0] / 2, -(scalar)     cfg.extents[1] / 2,      -(scalar)cfg.extents[2] / 2};
    ippl::Vector<scalar, 3> ext_fp{(scalar)cfg.extents [0], (scalar)          cfg.extents[1],           (scalar)cfg.extents[2]};
    field_state<gvt, device> F = make_field_state<gvt, device>(cfg.resolution[0], cfg.resolution[1], cfg.resolution[2], origin, hr_m);
    //A.m_mesh.origin = origin;
    //A.m_mesh.spacing = hr_m;
    //A.m_mesh.inverse_spacing = ippl::Vector<scalar, 3>(scalar(1)) / hr_m;
    testmesh(F.A_n.m_imap);
    source.m_mesh.origin = origin;
    source.m_mesh.spacing = hr_m;
    source.m_mesh.inverse_spacing = ippl::Vector<scalar, 3>(scalar(1)) / hr_m;
    EB.m_mesh.origin = origin;
    EB.m_mesh.spacing = hr_m;
    EB.m_mesh.inverse_spacing = ippl::Vector<scalar, 3>(scalar(1)) / hr_m;
    assert(!std::isnan(EB.m_mesh.spacing[0] * EB.m_mesh.spacing[1] * EB.m_mesh.spacing[2]));
    scalar _dt = scalar(cfg.timestep_ratio) ** std::min_element(hr_m.begin(), hr_m.end());
    cfg.total_time /= frame_gamma;
    size_t itercount = std::ceil(cfg.total_time / _dt);
    const scalar dt = _dt;//cfg.total_time / itercount;
    const scalar dx = hr_m[0];
    const scalar dy = hr_m[1];
    const scalar dz = hr_m[2];
    if(sq(dz / dx) + sq(dz / dy) >= 1){
        std::cerr << "Dispersion relation not satisfiable\n";
        abort();
    }
    
    //std::cout << *std::min_element(ext_fp.begin(), ext_fp.end()) / 10 << "\n";
    scalar variance = *std::min_element(ext_fp.begin(), ext_fp.end()) / 10;
    parallel_for_cuda([=] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
        scalar x = (scalar(i) + scalar(0.5)) * hr_m[0] + origin[0];
        scalar y = (scalar(j) + scalar(0.5)) * hr_m[1] + origin[1];
        scalar z = (scalar(k) + scalar(0.5)) * hr_m[2] + origin[2];
        //A(i, j, k)[0].fill(0);
        //A(i, j, k)[1].fill(0);
        //scalar value = 1e5f * (i == nr_m[0] / 2 && j == nr_m[1] / 2 && k == nr_m[2] / 2);// gauss(x, y, z, scalar(0), variance);
        scalar value = (cfg.num_particles == 0) * 1e4f * gauss(scalar(0), variance, x, y, z);
        //printf("Coords: %f, %f, %f, %f\n", x, y, z, value);
        F.A_np1(i, j, k).fill(0);
        F.A_n  (i, j, k).fill(0);
        F.A_nm1(i, j, k).fill(0);
        F.A_np1(i, j, k)[2] = value;
        F.A_n  (i, j, k)[2] = value;
        F.A_nm1(i, j, k)[2] = value;
        //init(i, j, k) = value;
        F.source_n(i, j, k).fill(0);

    }, F.A_n.getFullRange());
    cudaDeviceSynchronize();
    //return 0;
    BunchInitialize<scalar> mithra_config = generate_mithra_config(cfg, frame_boost);
    std::cout << "Sigma_P: \n" << mithra_config.sigmaGammaBeta_ << "\n\n";
    std::cout << "Sigma_R: \n" << mithra_config.sigmaPosition_ * unit_length_in_meters << "\n\n";
    //ChargeVector<scalar> mithra_cl;
    //initializeBunchEllipsoid(mithra_config, mithra_cl, 0, 1, 0);
    //{
    //    size_t c = 0;
    //    for(auto it = mithra_cl.begin();it != mithra_cl.end();it++){
    //        if(c++ > 100)break;
    //        std::cout << it->rnp << "\n";
    //    }
    //}
    //return 0;
    particles<scalar, device> bunch = initialize_bunch_mithra(mithra_config, frame_gamma);
    //std::cout << "Mean: " << bunch.covariance_matrix().first << "\n\n";
    //std::cout << "Cov: " << std::sqrt(bunch.covariance_matrix().second.data[0][0]) * unit_length_in_meters << "\n\n";
    //std::cout << "Cov: " << std::sqrt(bunch.covariance_matrix().second.data[1][1]) * unit_length_in_meters << "\n\n";
    //std::cout << "Cov: " << std::sqrt(bunch.covariance_matrix().second.data[2][2]) * unit_length_in_meters << "\n\n";
    //return 0;
    // = initialize_bunch<scalar>(
    //    cfg.num_particles, 
    //    cfg.charge, cfg.mass, 
    //    cfg.mean_position.cast<scalar>(), 
    //    cfg.sigma_position.cast<scalar>(), 
    //    cfg.position_truncations.cast<scalar>(), 
    //    frame_boost,
    //    cfg.sigma_momentum.cast<scalar>(),
    //    cfg.bunch_gamma,
    //    cfg.undulator_period
    //)
    ;
    cudaDeviceSynchronize();
    bunch.charge_per_particle = cfg.charge / bunch.count_m;
    bunch.mass_per_particle = cfg.mass / bunch.count_m;
    particles<scalar, host> hbunch = bunch.hostCopy();
    //bunch.for_each([dt]KOKKOS_FUNCTION(ippl::Vector<scalar, 3>& prev_pos, ippl::Vector<scalar, 3>& pos, ippl::Vector<scalar, 3>& gb){
    //    printf("Particlepos: %f, %f, %f\n", pos[0], pos[1], pos[2]);
    //});
    
    cudaDeviceSynchronize();

    //std::cerr << "Spacing: " << hr_m << "\n";
    //std::cerr << "Resolution: " << nr_m << "\n";
    //std::cerr << "Timestep: " << dt << "\n";
    //std::ofstream maxel("dat.txt");
    const scalar a1 = scalar(2) * (scalar(1) - sq(dt / dx) - sq(dt / dy) - sq(dt / dz));
    const scalar a2 = sq(dt / dx);
    const scalar a4 = sq(dt / dy);
    const scalar a6 = sq(dt / dz);
    const scalar a8 = sq(dt);
    constexpr int bts_divider = 3;
    assert_isreal(a1);
    assert_isreal(a2);
    assert_isreal(a4);
    assert_isreal(a6);
    assert_isreal(a8);
    const ippl::Vector<scalar, 3> betaMur = ippl::Vector<scalar, 3>(dt) / hr_m;
    assert_isreal((betaMur[0]));
    assert_isreal((betaMur[1]));
    assert_isreal((betaMur[2]));

    //ippl::Vector<scalar, 3> b0{(dt - dx) / (dt + dx),
    //                           (dt - dy) / (dt + dy),
    //                           (dt - dz) / (dt + dz)};
//
    //ippl::Vector<scalar, 3> b1{(dx * 2) / (dt + dx),
    //                           (dy * 2) / (dt + dy),
    //                           (dz * 2) / (dt + dz)};
    //constexpr scalar b2 = -1;
    std::ofstream ppos("ppos.txt");
    std::ofstream radiation("radiation.txt");
    
    int img_height = 500;
    int img_width = int(500.0 * cfg.extents[2] / cfg.extents[0]);
    float* imagedata = cuda_malloc_host_helper<float>(img_width * img_height * 3);
    unsigned char* imagedata_final = cuda_malloc_host_helper<unsigned char>(img_width * img_height * 3);
    source.setZero();
    cufftHandle intoout;
    scalar* fft_in;
    std::complex<scalar>* fft_out;
    std::complex<scalar>* fft_outh;
    //auto cufft_exec_funkschen = std::is_same_v<scalar, float> ? cufftExecR2C : cufftExecD2Z;
    {
        int n = nr_m[2];
        fft_in = cuda_malloc_helper<scalar>(n);
        fft_out = cuda_malloc_helper<std::complex<scalar>>(n/2+1);
        fft_outh = cuda_malloc_host_helper<std::complex<scalar>>(n/2+1);
        cufftSafeCall(cufftPlan1d(&intoout, n, CUFFT_R2C, 1));
        //cufftSafeCall(cufftPlanMany(&intoout, 1, &n, nullptr, F.A_n.m_imap.m * F.A_n.m_imap.n * 4, 0, nullptr, 1, 0, CUFFT_R2C, 1));
        //cufftExecR2C(ez_batchtransform, (scalar*)F.A_n.data + F.A_n.imap(F.A_n.m_imap.m / 2, F.A_n.m_imap.n / 2, 1), (cufftComplex*)fft_out);
    }
    
    //bunch.scatter(source, dt);
    //for(int i = 0;i < 1000;i++){
    //    field_update<false>(A, source, dt);
    //    field_update<true>(A, source, dt);
    //}
    //std::cerr << "Initial covariance matrix: " << bunch.covariance_matrix().second << "\n";
    std::cerr << "Grid extents: " << cfg.extents * unit_length_in_meters << " meters\n";
    std::cerr << "Bunch charge: " << bunch.count_m * bunch.charge_per_particle * (1.0 / coulomb_in_unit_charges) << " coulombs\n";
    std::cerr << "Bunch current: " << bunch.count_m * bunch.charge_per_particle * (1.0 / coulomb_in_unit_charges) * 299792458 / (cfg.sigma_position[2] * 2 * unit_length_in_meters) << " amperes\n";
    scalar fac = 1.0;
    if(cfg.experiment_options.contains("stretch-factor")){
        fac = cfg.experiment_options["stretch-factor"];
    }
    //bunch.for_each([=]KOKKOS_FUNCTION(ippl::Vector<scalar, 3>& ppos, ippl::Vector<scalar, 3>& pos, ippl::Vector<scalar, 3>& gb){
    //    pos[2] *= fac;
    //    ppos[2] *= fac;
    //});
    std::cout << "Initial covariance matrix:\n" << bunch.covariance_matrix().second << "\n";
    std::cout << "zvariance:\n" << bunch.covariance_matrix().second.data[2][2] * unit_length_in_meters * unit_length_in_meters << "\n";

    for(int i = 0;i < itercount;i++){
        if(i % 100 == 0 && cfg.experiment_options.contains("resort")){
            using bvec3 = decltype(bunch)::vec3;
            thrust::device_ptr<bvec3> bpp(bunch.previous_positions), bp(bunch.positions), bgb(bunch.gammaBeta);
            thrust::sort(
            thrust::make_zip_iterator(bpp, bp, bgb),
            thrust::make_zip_iterator(bpp + bunch.count_m, bp + bunch.count_m, bgb + bunch.count_m),
            [F]KOKKOS_FUNCTION(const thrust::tuple<bvec3, bvec3, bvec3>& x, const thrust::tuple<bvec3, bvec3, bvec3>& y){
                bvec3 px = thrust::get<1>(x);
                bvec3 py = thrust::get<1>(y);
                pear<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> pxi = F.A_n.m_mesh.gridCoordinatesOf(px);
                pear<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> pyi = F.A_n.m_mesh.gridCoordinatesOf(py);
                size_t xi1d = F.A_n.m_imap(pxi.first.cast<uint32_t>());
                size_t yi1d = F.A_n.m_imap(pyi.first.cast<uint32_t>());
                return xi1d < yi1d;
            });
        }
        using bc_type = second_order_mur_boundary_conditions;
        F.source_n.setZero();
        if(cfg.space_charge){
            bunch.scatter(F.source_n, dt);
        }
        bunch.scatterCurrent(F.source_n, dt);
        field_update<bc_type>(F, dt);
        evaluate_EB(F, EB, dt);
        cudaMemcpy(bunch.previous_positions, bunch.positions, sizeof(decltype(bunch)::vec3) * bunch.count_m, cudaMemcpyDeviceToDevice);
        {
            scalar bts = dt / bts_divider;
            bunch.update(EB, bts, dt * i, frame_boost, undulator_field, [distance_to_entry]KOKKOS_FUNCTION(const ippl::Vector<scalar, 3>& particle_labpos){return particle_labpos[2] > distance_to_entry;}, bts_divider);
        }
        //bunch.update(EB, dt, dt * i * 2, frame_boost, undulator_field);
        #ifndef NDEBUG
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        #else
        cudaDeviceSynchronize();
        #endif

        {
            uint32_t nx = F.A_n.m_imap.m;
            uint32_t ny = F.A_n.m_imap.n;
            uint32_t nz = F.A_n.m_imap.k;
            using vec3 = ippl::Vector<scalar, 3>;
            thrust::device_ptr<EB_vt> EB_dptr((EB_vt*)EB.data);
            auto red = thrust::transform_reduce(
                thrust::make_zip_iterator(EB_dptr, thrust::make_counting_iterator<size_t>(0)),
                thrust::make_zip_iterator(EB_dptr + EB.m_imap.size(), thrust::make_counting_iterator<size_t>(EB.m_imap.size())),
                [EB, nr_m, frame_boost]KOKKOS_FUNCTION(thrust::tuple<EB_vt, size_t> x) -> vec3{
                    auto first_idx = EB.m_imap.decompose(thrust::get<1>(x));
                    vec3 Elab = frame_boost.inverse_transform_EB(pear{thrust::get<0>(x)[0], thrust::get<0>(x)[1]}).first;
                    vec3 Blab = frame_boost.inverse_transform_EB(pear{thrust::get<0>(x)[0], thrust::get<0>(x)[1]}).second;
                    vec3 first = Elab.cross(Blab);
                    //vec3 first = thrust::get<0>(x)[0].cross(thrust::get<0>(x)[1]);
                    return first * scalar(first_idx[2] == nr_m[2] - 3 
                    && first_idx[1] > 1
                    && first_idx[1] < nr_m[1] - 1
                    && first_idx[0] > 1
                    && first_idx[0] < nr_m[0] - 1
                    );
                },
                vec3(0),
                [EB, nz]KOKKOS_FUNCTION(vec3 x, vec3 y){
                    return x + y;
            });
            auto _eb_n = EB;
            thrust::for_each(thrust::make_counting_iterator(1u),thrust::make_counting_iterator(nz - 1), [_eb_n, fft_in, nx, ny, nz]KOKKOS_FUNCTION(uint32_t i){
                fft_in[i - 1] = _eb_n(nx / 2, ny / 2, i)[0].cross(_eb_n(nx / 2, ny / 2, i)[1])[2];
            });
            cudaDeviceSynchronize();
            
            cudaDeviceSynchronize();
            const scalar zpos_labframe = (frame_boost.primedToUnprimed() * prepend_t(EB.m_mesh.positionOf(ippl::Vector<uint32_t, 3>{nr_m[0] / 2, nr_m[1] / 2, nr_m[2] / 2}), scalar(dt * i * 2)))[3];
            const double radiation_value_in_whatever_unit_the_simulation_is_running = double(red[2]) * double(hr_m[0]) * double(hr_m[1]);
            const double radiation_in_watt = radiation_value_in_whatever_unit_the_simulation_is_running * 3.628e52 / sq(double(alpha_scaling_factor));
            radiation << zpos_labframe * unit_length_in_meters << " " << radiation_in_watt << std::endl;
            std::fill(fft_outh, fft_outh + F.A_n.m_imap.k, scalar(0));
            cufftSafeCall(cufftExecR2C(intoout, fft_in, (cufftComplex*)fft_out));
            cudaMemcpy(fft_outh, fft_out, (nr_m[2]/2 + 1) * sizeof(std::complex<scalar>), cudaMemcpyDeviceToHost);
            if(i % 500 == 0 && i > 10){
                std::ofstream spectrum("spectrum" + std::to_string(i) + ".txt");
                for(int w = 0;w < nr_m[2] / 2 + 1;w++){
                    spectrum << std::abs(fft_outh[w]) << "\n";// * std::conj(fft_out_as_complex[w])).real() << "\n";
                }
            }
            
            //auto EBpol = 
        }

        
        //Save output image dump file
        if((cfg.output_rhythm != 0) && (i % cfg.output_rhythm == 0)){
            using vec3 = rm::Vector<float, 3>;
            if(false){
                ClearFrame();
                rc.clear();
                glDisable(GL_CULL_FACE);
                glDisable(GL_DEPTH_TEST);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                bunch.update_hostcopy<false>(hbunch);
                draw_bunch<scalar>(hbunch, 1.0 / cfg.extents[1]);
                draw_field<scalar>(EB, 1.0 / cfg.extents[1]);
                std::vector<unsigned char> pixels(width * height * 3, 70);
                glFinish();
                glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
                glFinish();
                for(int _i = 0;_i < height / 2;_i++){
                    for(int _j = 0;_j < width;_j++){
                        int i1 = (_i * width + _j) * 3;
                        int i2 = ((height - _i - 1) * width + _j) * 3;
                        std::iter_swap(pixels.begin() + i1 + 0, pixels.begin() + i2 + 0);
                        std::iter_swap(pixels.begin() + i1 + 1, pixels.begin() + i2 + 1);
                        std::iter_swap(pixels.begin() + i1 + 2, pixels.begin() + i2 + 2);
                    }
                }
                char output[1024] = {0};
                snprintf(output, 1023, "../data/outimage%.05d.bmp", i);
                stbi_write_bmp(output, width, height, 3, pixels.data());
            }

            //Will be required later as well
            //glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, dpixels.data());

            if(true){
                std::memset(imagedata, 0, img_width * img_height * 3 * sizeof(float));
                bunch.update_hostcopy<false>(hbunch);
                for(size_t hi = 0;hi < hbunch.count_m;hi++){
                    particles<scalar, host>::vec3 ppos = hbunch.positions[hi];
                    ppos -= F.A_n.m_mesh.origin;
                    ppos /= cfg.extents.cast<scalar>();
                    int x_imgcoord = ppos[2] * img_width;
                    int y_imgcoord = ppos[0] * img_height;
                    //printf("%d, %d\n", x_imgcoord, y_imgcoord);
                    if(y_imgcoord >= 0 && x_imgcoord >= 0 && x_imgcoord < img_width && y_imgcoord < img_height){
                        const float intensity = std::min(255.f, (img_width * img_height * 15.f) / cfg.num_particles);
                        //std::cout << intensity << "\n";
                        imagedata[(y_imgcoord * img_width + x_imgcoord) * 3 + 1] = 
                        std::min(255.f, imagedata[(y_imgcoord * img_width + x_imgcoord) * 3 + 1] + intensity);
                    }
                };
                grid<EB_vt, host> ebh = EB.hostCopy();
                ippl::Vector<uint32_t, 3> beg{1, nr_m[1] / 2, 1};
                ippl::Vector<uint32_t, 3> end{nr_m[0] - 1, nr_m[1] / 2 + 1, nr_m[2] - 1};
                double exp_sum = 0;
                uint64_t acount = 0;
                serial_for<3>([&](size_t i, size_t j, size_t k){
                    ippl::Vector<scalar, 3> poynting = ebh(i, j, k)[0].cross(ebh(i,j,k)[1]);
                    scalar nrm = std::sqrt(poynting.squaredNorm());
                    if(nrm > 0.0f){
                        exp_sum += std::log2(nrm);
                        acount++;
                    }
                }, beg, end);
                double exp_avg = double(exp_sum) / double(acount);
                {
                    for(int i = 1;i < img_width;i++){
                        for(int j = 1;j < img_height;j++){
                            int i_remap = (double(i) / (img_width - 1)) * (nr_m[2] - 2);
                            int j_remap = (double(j) / (img_height - 1)) * (nr_m[0] - 2);
                            EB_vt acc = ebh(j_remap + 1, nr_m[1] / 2, i_remap + 1);
                            ippl::Vector<scalar, 3> poynting = acc[0].cross(acc[1]);
                            imagedata[(j * img_width + i) * 3 + 0] = (unsigned char)(std::min(255u, (unsigned int)(0.5f * std::sqrt(std::sqrt(poynting.squaredNorm())))));
                        }                
                    }
                }
                char output[1024] = {0};
                snprintf(output, 1023, "../data/outimage%.05d.bmp", i);
                std::transform(imagedata, imagedata + img_height * img_width * 3, imagedata_final, [](float x){return (unsigned char)x;});
                stbi_write_bmp(output, img_width, img_height, 3, imagedata_final);
                ebh.destroy();
            }
            
        }
        std::cerr << "Final variance x: " << bunch.zvariance()[0] * unit_length_in_meters * unit_length_in_meters * 1e12 << "\n";
        std::cerr << "Final variance y: " << bunch.zvariance()[1] * unit_length_in_meters * unit_length_in_meters * 1e12 << "\n";
        std::cerr << "Final variance z: " << bunch.zvariance()[2] * unit_length_in_meters * unit_length_in_meters * 1e12 << "\n";
        printf("Step %d of %lu\n", i, itercount);
    }
    std::cerr << "Final variance z: " << bunch.zvariance()[2] << "\n";
    std::cerr << "Final covariance matrix: " << bunch.covariance_matrix().second << "\n";
        //thrust::device_ptr<gvt> d_ptr = thrust::device_pointer_cast(A.data);
        //gvt z;z.fill(ippl::Vector<float, 4>{0,0,0,0});
        //gvt sumv = thrust::reduce(d_ptr, d_ptr + A.m_imap.size(),z, []KOKKOS_FUNCTION(const gvt& x, const gvt& y){
        //    return x + y;
        //});
        //maxel << (2 * dt) * i << " " << sumv[0][0] << std::endl;

        //parallel_for_cuda([A] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
        //    const uint32_t index = A.imap(i,j,k);
        //    const uint32_t bstride = A.m_imap.n * A.m_imap.k;
        //    const uint32_t sstride = A.m_imap.k;
        //    for(unsigned int d = 0;d < 4;d++){
        //        //if( blockIdx.x < 0 ) { __syncthreads(); }
        //        A[index][d+4] = -A[index][d+4]   +  A[index][d] * a1
        //                                             + (A[index+bstride][d] + A[index-bstride][d]) * a2
        //                                             + (A[index+sstride][d] + A[index-sstride][d]) * a4
        //                                             + (A[index+1]      [d] + A[index-1][d]) * a6;
        //    }
        //    
        //}, A.getRange());
        //parallel_for_cuda([A] __host__ __device__(uint32_t i, uint32_t j, uint32_t k)mutable{
        //    const uint32_t index = A.imap(i,j,k);
        //    const uint32_t bstride = A.m_imap.n * A.m_imap.k;
        //    const uint32_t sstride = A.m_imap.k;
        //    float* __restrict__ Adata = (float*)A.data;
        //    for(unsigned int d = 0;d < 4;d++){
        //        //if( blockIdx.x < 0 ) { __syncthreads(); }
        //        A[index][d] = -A[index][d]   +  A[index][d+4] * a1
        //                                             + (A[index+bstride][d+4] + A[index-bstride][d+4]) * a2
        //                                             + (A[index+sstride][d+4] + A[index-sstride][d+4]) * a4
        //                                             + (A[index+1]      [d+4] + A[index-1][d+4]) * a6;
        //    }
        //}, A.getRange());
    grid<EB_vt, host> hgr = EB.hostCopy();
    cudaDeviceSynchronize();
    auto t2 = nanoTime();
    //std::cerr << "Took " << (t2 - t1) / 1e9 << " seconds\n";
    //std::cerr << A.m_imap.size()  << std::endl;
    //std::cerr << hgr.m_imap.size() << std::endl;
    //std::ofstream line("line.txt");
    //serial_for<3>([&line, hgr](auto... args){
    //    line << hgr.m_mesh.positionOf(ippl::Vector<int, 3>{static_cast<int>(args)...}) << " " << hgr(args...)[0][0] << "\n";
    //}, ippl::Vector<uint32_t, 3>{0, nr_m[1] / 2, nr_m[2] / 2}, ippl::Vector<uint32_t, 3>{nr_m[0] - 1, nr_m[1] / 2 + 1, nr_m[2] / 2 + 1});
    return 0;
    #endif
}
