#ifndef GRID_CU
#define GRID_CU

#include "common.cu"
#include "cfg.hpp"
#include <system_error>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <cuda.h>
#include <cstring>
#include <concepts>
#include <stdio.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

template<typename... Args>
KOKKOS_INLINE_FUNCTION constexpr auto product(Args... args) {
    return (... * args);
}
template<typename... Args>
KOKKOS_INLINE_FUNCTION constexpr auto sum(Args... args) {
    return (... + args);
}
template<unsigned int Dim, typename callable, typename... Ts>
KOKKOS_INLINE_FUNCTION void serial_for(callable c, ippl::Vector<uint32_t, Dim> from, ippl::Vector<uint32_t, Dim> to, Ts... args){
    if constexpr(sizeof...(Ts) == Dim){
        c(args...);
    }
    else{
        for(uint32_t i = from[sizeof...(Ts)];i < to[sizeof...(Ts)];i++){
            serial_for(c, from, to, args..., i);
        }
    }
}

enum space{
    host, device, managed
};
template<unsigned Dim>
struct index_range{
    ippl::Vector<uint32_t, Dim> begin;
    ippl::Vector<uint32_t, Dim> end;
    KOKKOS_FUNCTION size_t volume()const noexcept{
        size_t prod = 1;
        for(unsigned d = 0;d < Dim;d++){
            prod *= size_t(end[d] - begin[d]);
        }
        return prod;
    }
    KOKKOS_FUNCTION ippl::Vector<uint32_t, 3> decompose(size_t i) const noexcept requires(Dim == 3){
        assert(i < volume());
        size_t m = end[0] - begin[0];
        size_t n = end[1] - begin[1];
        size_t k = end[2] - begin[2];
        ippl::Vector<uint32_t, 3> ret;
        ret[2] = i / (size_t(m) * size_t(n));
        size_t rem = i % (size_t(m) * size_t(n));
        ret[0] = rem % m;
        ret[1] = rem / m;
        ret += begin.template cast<uint32_t>();
        return ret;
    }
};
template<unsigned int Dim, typename callable, typename... Ts>
KOKKOS_INLINE_FUNCTION void serial_for(callable c, const index_range<Dim>& ir, Ts... args){
    if constexpr(sizeof...(Ts) == Dim){
        c(args...);
    }
    else{
        for(uint32_t i = ir.begin[sizeof...(Ts)];i < ir.end[sizeof...(Ts)];i++){
            serial_for(c, ir, args..., i);
        }
    }
}
KOKKOS_INLINE_FUNCTION constexpr unsigned int idiv_roundup(unsigned int n, unsigned int d){
    return (n + d - 1) / d;
}
template<typename callable>
__global__ void kernel_dispatch3(callable calla, index_range<3> limits){
    ippl::Vector<uint32_t, 3> c{blockDim.x * blockIdx.x + threadIdx.x + limits.begin[0],
                                blockDim.y * blockIdx.y + threadIdx.y + limits.begin[1],
                                blockDim.z * blockIdx.z + threadIdx.z + limits.begin[2]};
    if(c[0] < limits.end[0] && c[1] < limits.end[1] && c[2] < limits.end[2]){
        //printf("Dispatsching: %d, %d, %d\n", c[0], c[1], c[2]);
        if constexpr(std::is_invocable_v<callable, uint32_t, uint32_t, uint32_t>){
            calla(c[0], c[1], c[2]);
        } else if constexpr(std::is_invocable_v<callable, ippl::Vector<uint32_t, 3>>){
            calla(c);
        }
    }
}
template<typename callable>
__global__ void kernel_dispatch(callable calla, index_range<1> limits){
    uint32_t c = blockDim.x * blockIdx.x + threadIdx.x + limits.begin[0];
    if(c < limits.end[0]){
        calla(c);
    }
}
template<typename callable>
void parallel_for_cuda(callable c, const index_range<3>& ir){
    //dim3 begin{ir.begin[0], ir.begin[1], ir.begin[2]};
    //dim3 end  {ir.end[0]  , ir.end[1]  , ir.end[2]  };
    dim3 threadRange{8, 8, 4};
    dim3 block_range  {idiv_roundup(ir.end[0] - ir.begin[0], 8),
                 idiv_roundup(ir.end[1] - ir.begin[1], 8),
                 idiv_roundup(ir.end[2] - ir.begin[2], 4)
                };
    cudaFuncAttributes attrs;
    cudaFuncGetAttributes(&attrs, kernel_dispatch3<callable>);
    kernel_dispatch3<<<block_range, threadRange>>>(c, ir);
    //std::cout << "Called kernel with " << attrs.maxThreadsPerBlock << " maxThreadsPerBlock\n";
    //std::cout << "Called kernel with " << attrs.numRegs << " registers\n";
    #ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        cudaFuncAttributes attrs;
        cudaFuncGetAttributes(&attrs, kernel_dispatch3<callable>);
        std::cerr << "Called kernel with " << attrs.maxThreadsPerBlock << " maxThreadsPerBlock\n";
    }
    #endif
}
template<typename callable>
void parallel_for_cuda(callable c, const index_range<1>& ir){
    if(ir.begin[0] == ir.end[0])return;
    //dim3 begin{ir.begin[0], ir.begin[1], ir.begin[2]};
    //dim3 end  {ir.end[0]  , ir.end[1]  , ir.end[2]  };
    dim3 block_range  {idiv_roundup(ir.end[0] - ir.begin[0], 256)};
    
    dim3 threadRange{256,1,1};

    kernel_dispatch<<<block_range, threadRange>>>(c, ir);
    #ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        cudaFuncAttributes attrs;
        cudaFuncGetAttributes(&attrs, kernel_dispatch<callable>);
        std::cerr << "Called kernel with " << attrs.maxThreadsPerBlock << " maxThreadsPerBlock\n";
    }
    #endif
}
template<uint32_t args0, uint32_t... argrest>
struct ct_indexmap{
    template<std::integral T, std::integral... Ts>
    KOKKOS_INLINE_FUNCTION size_t operator()(T argument, Ts... arguments)const{
        if constexpr(sizeof...(arguments) > 0){
            static_assert(std::is_integral_v<T>);
            static_assert(std::conjunction_v<std::is_integral<Ts>...>);
            assert(argument < args0);
            return argument * product(argrest...) + ct_indexmap<argrest...>()(arguments...);
        }
        else{
            return argument;
        }
    }

    KOKKOS_INLINE_FUNCTION constexpr size_t total_size(){
        return args0 * (argrest * ...);
    }
    KOKKOS_INLINE_FUNCTION constexpr size_t nDims(){
        return sizeof...(argrest) + 1;
    }
};
struct ghosted_indexmap {
    /**
     * @brief Those are the actual underlying extents WITH ghost cells
     * 
     */
    uint32_t m;
    uint32_t n;
    uint32_t k;
    KOKKOS_FUNCTION constexpr ghosted_indexmap(size_t _m, size_t _n, size_t _k) : m(_m + 2), n(_n + 2), k(_k + 2){}
    KOKKOS_INLINE_FUNCTION size_t operator()(uint32_t i, uint32_t j, uint32_t l) const noexcept {
        assert(i < m);
        assert(j < n);
        assert(l < k);
        return i + j * m + l * m * n;
    }
    KOKKOS_INLINE_FUNCTION size_t operator()(const ippl::Vector<uint32_t, 3>& acc)const noexcept{
        return this->operator()(acc[0], acc[1], acc[2]);
    }
    KOKKOS_INLINE_FUNCTION ippl::Vector<uint32_t, 3> decompose(size_t i) const noexcept {
        assert(i < size());
        ippl::Vector<uint32_t, 3> ret;
        ret[2] = i / (size_t(m) * size_t(n));
        size_t rem = i % (size_t(m) * size_t(n));
        ret[0] = rem % m;
        ret[1] = rem / m;
        return ret;
    }
    KOKKOS_INLINE_FUNCTION size_t size() const noexcept {
        return m * n * k;
    }
    index_range<3> getRange()const noexcept{
        return index_range<3>{.begin{1,1,1}, .end{m - 1, n - 1, k - 1}};
    }
    index_range<3> getFullRange()const noexcept{
        return index_range<3>{.begin{0, 0, 0}, .end{m, n, k}};
    }
    KOKKOS_INLINE_FUNCTION uint32_t nghost()const noexcept{
        return 1;
    }
};

template<typename T>
struct mesh{
    ippl::Vector<T, 3> origin ;
    ippl::Vector<T, 3> spacing;
    ippl::Vector<T, 3> inverse_spacing;
    template<std::integral O>
    KOKKOS_INLINE_FUNCTION ippl::Vector<T, 3> positionOf(const ippl::Vector<O, 3>& p)const noexcept{
        //TODO: nghost
        return (p - ippl::Vector<O, 3>(1)).template cast<T>() * spacing + origin;
    }
    KOKKOS_INLINE_FUNCTION pear<ippl::Vector<int, 3>, ippl::Vector<T, 3>> gridCoordinatesOfWithoutOffset(ippl::Vector<T, 3> pos)const noexcept{
        pear<ippl::Vector<int, 3>, ippl::Vector<T, 3>> ret;
        ippl::Vector<T, 3> relpos = pos - origin;
        ippl::Vector<T, 3> gridpos = relpos / spacing;
        ippl::Vector<int, 3> ipos;
        ippl::Vector<T, 3> fracpos = gridpos.decompose(&ipos);

        //TODO: NGHOST!!!!!!!
        ipos += ippl::Vector<int, 3>(1);
        ret.first = ipos;
        ret.second = fracpos;
        return ret;
    }
    KOKKOS_INLINE_FUNCTION T volume()const noexcept{
        return spacing[0] * spacing[1] * spacing[2];
    }
    KOKKOS_INLINE_FUNCTION pear<ippl::Vector<int, 3>, ippl::Vector<T, 3>> gridCoordinatesOf(ippl::Vector<T, 3> pos)const noexcept{
        //return pear<ippl::Vector<int, 3>, ippl::Vector<T, 3>>{ippl::Vector<int, 3>{5,5,5}, ippl::Vector<T, 3>{0,0,0}};
        //printf("%.10e, %.10e, %.10e\n", (inverse_spacing * spacing)[0], (inverse_spacing * spacing)[1], (inverse_spacing * spacing)[2]);
        pear<ippl::Vector<int, 3>, ippl::Vector<T, 3>> ret;
        //pos -= spacing * T(0.5);
        ippl::Vector<T, 3> relpos = pos - origin;
        ippl::Vector<T, 3> gridpos = relpos * inverse_spacing;
        ippl::Vector<int, 3> ipos;
        ipos = gridpos.template cast<int>();
        ippl::Vector<T, 3> fracpos = gridpos - ipos.cast<T>();//.decompose(&ipos);

        //TODO: NGHOST!!!!!!!
        ipos += ippl::Vector<int, 3>(1);
        ret.first = ipos;
        ret.second = fracpos;
        return ret;
    }
};
template<typename T, bool fundamental> //For true
struct extract_value_type_impl{
    using type = T;   
};
template<typename T>
struct extract_value_type_impl<T, false>{ //For false
    using type = extract_value_type_impl<typename T::value_type, std::is_fundamental_v<typename T::value_type>>::type;
};
template<typename T>
struct extract_value_type{
    using type = extract_value_type_impl<T, std::is_fundamental_v<T>>::type;
};
template<typename T>
using extract_value_t = typename extract_value_type<T>::type;
template <typename T>
struct grid_base{
    using imap_type = ghosted_indexmap;
    imap_type m_imap;
    using mesh_type = mesh<extract_value_t<T>>;
    mesh_type m_mesh;
    T *__restrict__ data;
    index_range<3> getRange()const noexcept{
        return m_imap.getRange();
    }
    index_range<3> getFullRange()const noexcept{
        return m_imap.getFullRange();
    }
    KOKKOS_INLINE_FUNCTION uint32_t nghost()const noexcept{
        return m_imap.nghost();
    }
};
template <typename T, space sp>
struct grid : grid_base<T>{};
template<typename T>
struct grid<T, device> : grid_base<T>{
    using imap_type = typename grid_base<T>::imap_type;
    using super = grid_base<T>;
    using mesh_type = super::mesh_type;
    using value_type = T;
    constexpr grid() = default;
    constexpr grid(imap_type im, T* __restrict__ d) : super{im, mesh_type{}, d}{}
    grid(uint32_t m, uint32_t n, uint32_t k) : super{.m_imap = imap_type(m, n, k)} {
        grid_base<T>::data = cuda_malloc_helper<T>(grid_base<T>::m_imap.size());
    }
    grid(uint32_t m, uint32_t n, uint32_t k, mesh_type mesh) : super{.m_imap = imap_type(m, n, k), .m_mesh = mesh} {
        grid_base<T>::data = cuda_malloc_helper<T>(grid_base<T>::m_imap.size());
    }
    void setZero(){
        auto dis = *this;
        parallel_for_cuda([dis]KOKKOS_FUNCTION(size_t i, size_t j, size_t k)mutable{
            dis(i, j, k).fill(0);
        }, super::getFullRange());
        cudaMemset(super::data, 0, super::m_imap.size() * sizeof(T));
    }
    KOKKOS_INLINE_FUNCTION T &operator()(uint32_t i, uint32_t j, uint32_t l) noexcept {
        return super::data[super::m_imap(i, j, l)];
    }

    KOKKOS_INLINE_FUNCTION const T &operator()(uint32_t i, uint32_t j, uint32_t l) const noexcept {
        return super::data[super::m_imap(i, j, l)];
    }
    KOKKOS_INLINE_FUNCTION size_t imap(uint32_t i, uint32_t j, uint32_t l)const noexcept{
        return super::m_imap(i, j, l);
    }
    KOKKOS_INLINE_FUNCTION size_t imap(ippl::Vector<uint32_t, 3> in)const noexcept{
        return super::m_imap(in[0], in[1], in[2]);
    }
    KOKKOS_INLINE_FUNCTION T &operator[](uint32_t i) noexcept {
        assert(i < super::m_imap.size());
        return super::data[i];
    }
    KOKKOS_INLINE_FUNCTION const T &operator[](uint32_t i) const noexcept {
        assert(i < super::m_imap.size());
        return super::data[i];
    }

    grid<T, host> hostCopy() const {
        grid<T, host> ret(this->m_imap, cuda_malloc_host_helper<T>(super::m_imap.size()));
        ret.m_mesh = super::m_mesh;
        cudaMemcpy(ret.data, super::data, sizeof(T) * super::m_imap.size(), cudaMemcpyDeviceToHost);
        return ret;
    }
    void destroy(){
        cudaFree(super::data);
    }
    T sum(){
        return sum(this->getRange());
    }
    T sum(index_range<3> range) const{
        const T* __restrict__ data = this->data;
        index_range<3> interior = range;
        const auto mimap = this->m_imap;
        T sum = thrust::transform_reduce(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(interior.volume()),
            [data, interior, mimap]KOKKOS_FUNCTION(size_t x) -> T{
                //return T(1);
                auto first_idx = interior.decompose(x);
                T ret = data[mimap(first_idx[0], first_idx[1], first_idx[2])];
                //if(ret.squaredNorm() > 0)
                //    printf("%f\n", ret.squaredNorm());
                return data[mimap(first_idx[0], first_idx[1], first_idx[2])];
            },
            T(0),
            []KOKKOS_FUNCTION(T x, T y){
                return x + y;
        });
        return sum;
    }
    /**
     * @brief Unfortunate function, the templated version of this is buggy
     * 
     * @return extract_value_t<T> (scalar) the volumetric integral of 1/2 * (E^2 + B^2)
     */
    extract_value_t<T> volumeIntegral_energy(){
        const T* __restrict__ data = this->data;
        index_range<3> interior = this->getRange();
        using skalar = extract_value_t<T>;
        const auto mimap = this->m_imap;
        skalar sum = thrust::transform_reduce(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(interior.volume()),
            [data, interior, mimap]KOKKOS_FUNCTION(size_t x) -> skalar{
                //return T(1);
                auto first_idx = interior.decompose(x);
                T ret = data[mimap(first_idx[0], first_idx[1], first_idx[2])];
                skalar retsk = ret[0].squaredNorm() + ret[1].squaredNorm();
                //if(retsk > 0)
                //    printf("%f\n", retsk);
                return ret[0].squaredNorm() + ret[1].squaredNorm();
            },
            skalar(0),
            []KOKKOS_FUNCTION(skalar x, skalar y){
                return x + y;
        });
        return sum * super::m_mesh.volume();
    }
    template<typename callable>
    auto volumeIntegral(callable c) -> std::invoke_result_t<callable, T> const{
        const T* __restrict__ data = this->data;
        index_range<3> interior = this->getRange();
        using result_type = std::remove_all_extents_t<decltype(c(data[0]))>;
        const auto mimap = this->m_imap;
        result_type sum = thrust::transform_reduce(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(interior.volume()),
            [data, interior, mimap, c]KOKKOS_FUNCTION(size_t x) -> result_type{
                //return T(1);
                auto first_idx = interior.decompose(x);
                T ret = data[mimap(first_idx[0], first_idx[1], first_idx[2])];
                //if(ret.squaredNorm() > 0)
                //    printf("%f\n", ret.squaredNorm());
                return c(data[mimap(first_idx[0], first_idx[1], first_idx[2])]);
            },
            result_type (0),
            []KOKKOS_FUNCTION(result_type x, result_type y){
                return x + y;
        });
        return sum * super::m_mesh.volume();
    }
    T volumeIntegral() const{
        const T* __restrict__ data = this->data;
        index_range<3> interior = this->getRange();
        const auto mimap = this->m_imap;
        T sum = thrust::transform_reduce(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(interior.volume()),
            [data, interior, mimap]KOKKOS_FUNCTION(size_t x) -> T{
                //return T(1);
                auto first_idx = interior.decompose(x);
                T ret = data[mimap(first_idx[0], first_idx[1], first_idx[2])];
                //if(ret.squaredNorm() > 0)
                //    printf("%f\n", ret.squaredNorm());
                return data[mimap(first_idx[0], first_idx[1], first_idx[2])];
            },
            T(0),
            []KOKKOS_FUNCTION(T x, T y){
                return x + y;
        });
        return sum * super::m_mesh.volume();
    }
};
template<typename T>
struct grid<T, host> : grid_base<T>{
    using imap_type = typename grid_base<T>::imap_type;
    using super = grid_base<T>;
    using mesh_type = super::mesh_type;
    using value_type = T;

    KOKKOS_INLINE_FUNCTION constexpr grid(imap_type im, T* __restrict__ d) : super{im, mesh_type{}, d}{}
    grid(uint32_t m, uint32_t n, uint32_t k) : super{.m_imap = imap_type(m, n, k)} {
        grid_base<T>::data = cuda_malloc_host_helper<T>(grid_base<T>::m_imap.size());
    }
    KOKKOS_INLINE_FUNCTION T &operator()(uint32_t i, uint32_t j, uint32_t l) noexcept {
        return super::data[super::m_imap(i, j, l)];
    }

    KOKKOS_INLINE_FUNCTION const T &operator()(uint32_t i, uint32_t j, uint32_t l) const noexcept {
        return super::data[super::m_imap(i, j, l)];
    }
    KOKKOS_INLINE_FUNCTION size_t imap(uint32_t i, uint32_t j, uint32_t l)const noexcept{
        return super::m_imap(i, j, l);
    }
    KOKKOS_INLINE_FUNCTION T &operator[](uint32_t i) noexcept {
        return super::data[i];
    }
    KOKKOS_INLINE_FUNCTION const T &operator[](uint32_t i) const noexcept {
        return super::data[i];
    }
    void setZero(){
        std::memset(super::data, 0, super::m_imap.size());
    }

    grid<T, device> deviceCopy() const {
        grid<T, device> ret(this->m_imap, cuda_malloc_helper<T>(super::m_imap.size()));
        cudaMemcpy(ret.data, super::data, sizeof(T) * super::m_imap.size(), cudaMemcpyHostToDevice);
        return ret;
    }
    void destroy(){
        cudaFreeHost(super::data);
    }
};
template<typename applyable, unsigned i, unsigned N>
auto& apply_impl(applyable& v, const ippl::Vector<uint32_t, N>& arg){
    
}
template<typename applyable, unsigned N>
KOKKOS_INLINE_FUNCTION auto& apply(applyable& v, const ippl::Vector<uint32_t, N>& arg){
    if constexpr(N == 1){
        return v(arg[0]);
    }
    if constexpr(N == 2){
        return v(arg[0], arg[1]);
    }
    if constexpr(N == 3){
        return v(arg[0], arg[1], arg[2]);
    }
    if constexpr(N == 4){
        return v(arg[0], arg[1], arg[2], arg[3]);
    }
    if constexpr(N == 5){
        return v(arg[0], arg[1], arg[2], arg[3], arg[4]);
    }
    //return typename applyable::value_type{};
}
template<typename T, space sp>
KOKKOS_INLINE_FUNCTION T second_derivative(grid<T, sp> g, ippl::Vector<uint32_t, 3> pos, uint32_t dir){
    const ippl::Vector<uint32_t, 3> ex{g.m_imap.m, g.m_imap.n, g.m_imap.k};
    extract_value_t<T> scal = g.m_mesh.inverse_spacing[dir] * g.m_mesh.inverse_spacing[dir];
    T ret;
    ret.fill(0);
    if(pos[dir] == 0){
        pos[dir] += 2;
        ret += apply(g, pos);
        pos[dir] -= 1;
        ret -= apply(g, pos) * extract_value_t<T>(2);
        pos[dir] -= 1;
        ret += apply(g, pos);
    }
    else if(pos[dir] == ex[dir] - 1){
        ret += apply(g, pos);
        pos[dir] -= 1;
        ret -= apply(g, pos) * extract_value_t<T>(2);
        pos[dir] -= 1;
        ret += apply(g, pos);
    }
    else{
        pos[dir] += 1;
        ret += apply(g, pos);
        pos[dir] -= 1;
        ret -= apply(g, pos) * extract_value_t<T>(2);
        pos[dir] -= 1;
        ret += apply(g, pos);
    }
    return ret * scal;
}

/**
 * @brief Lorentz boost along a single axis, computationally very efficient
 * 
 * @tparam T scalar type (e.g. float or double)
 * @tparam axis Boost axis (0=x, 1=y, 2=z, etc.) (default 2, corresponding to z)
 */
template<typename T, unsigned axis = 2>
struct UniaxialLorentzframe{
    constexpr static T c = 1.0;
    using scalar = T;
    using Vector3 = ippl::Vector<T, 3>;
    scalar beta_m;
    scalar gammaBeta_m;
    scalar gamma_m;
    KOKKOS_INLINE_FUNCTION UniaxialLorentzframe(const scalar gammaBeta){
        gammaBeta_m = gammaBeta;
        beta_m = gammaBeta / sqrt(1 + gammaBeta * (gammaBeta));
        gamma_m = sqrt(1 + gammaBeta * (gammaBeta));
    }
    KOKKOS_INLINE_FUNCTION void primedToUnprimed(Vector3& arg, scalar time)const noexcept{
        arg[axis] = gamma_m * (arg[axis] + beta_m * time); 
    }
    KOKKOS_INLINE_FUNCTION pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_EB(const pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& unprimedEB)const noexcept{
        
        pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
        //ret.first = unprimedEB.first;
        //ret.first[0] -= unprimedEB.second[1];
        //ret.first[1] += unprimedEB.second[0];
        //ret.second = unprimedEB.second;
        ippl::Vector<scalar, 3> betavec{0, 0, beta_m};
        ret.first  = ippl::Vector<T, 3>(unprimedEB.first  + betavec.cross(unprimedEB.second)) * gamma_m;// - (vnorm * (gamma_m - 1) * (unprimedEB.first.dot(vnorm)));
        ret.second = ippl::Vector<T, 3>(unprimedEB.second - betavec.cross(unprimedEB.first )) * gamma_m;// - (vnorm * (gamma_m - 1) * (unprimedEB.second.dot(vnorm)));
        ret.first[axis] -= (gamma_m - 1) * unprimedEB.first[axis];
        ret.second[axis] -= (gamma_m - 1) * unprimedEB.second[axis];
        return ret;
    }
};
#endif
