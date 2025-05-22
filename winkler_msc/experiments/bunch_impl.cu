#include "bunch.cu"
#include "grid.cu"
template<typename scalar, space sp>
pear<ippl::Vector<scalar, 6>, matrix<scalar, 6, 6>> particles<scalar, sp>::covariance_matrix()const noexcept{
    matrix<scalar, 6, 6> ret(0);
    //using accmat = matrix<acc_scalar, 6, 6>;
    using vec6 = ippl::Vector<scalar, 6>;
    //using accvec = ippl::Vector<acc_scalar, 6>;
    const ippl::Vector<scalar, 3>* positions = this->positions;
    const ippl::Vector<scalar, 3>* gammaBeta = this->gammaBeta;
    /*accvec sum = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count_m),
        [positions, gammaBeta]KOKKOS_FUNCTION(size_t x) -> accvec{
            //return T(1);
            vec3 p  = positions[x];
            vec3 gb = gammaBeta[x];
            return vec6{p[0], p[1], p[2], gb[0], gb[1], gb[2]}.template cast<acc_scalar>();
        },
        accvec(0),
        []KOKKOS_FUNCTION(accvec x, accvec y){
            return x + y;
    });
    const accvec mean = sum / acc_scalar(count_m);
    accmat covs = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count_m),
        [positions, gammaBeta, mean]KOKKOS_FUNCTION(size_t x) -> accmat{
            //return T(1);
            vec3 p  = positions[x];
            vec3 gb = gammaBeta[x];
            accvec central = vec6{p[0], p[1], p[2], gb[0], gb[1], gb[2]}.template cast<acc_scalar>();
            central -= mean;
            accmat ret;
            for(unsigned i = 0;i < 6;i++){
                for(unsigned j = 0;j < 6;j++){
                    ret.data[j][i] = central[i] * central[j];
                }
            }
            return ret;
        },
        accmat(0),
        []KOKKOS_FUNCTION(accmat x, accmat y){
            return x + y;
    });*/
    Dataset<scalar, 6> mean_and_cov = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count_m),
        [positions, gammaBeta]KOKKOS_FUNCTION(size_t x){
            vec3 p  = positions[x];
            vec3 gb = gammaBeta[x];
            return Dataset<scalar, 6>(ippl::Vector<scalar, 6>{p[0], p[1], p[2], gb[0], gb[1], gb[2]}, matrix<scalar, 6, 6>(scalar(0)), 1);
        },
        Dataset<scalar, 6>(ippl::Vector<scalar, 6>(scalar(0)), matrix<scalar, 6, 6>(scalar(0)), 0),
        []KOKKOS_FUNCTION(Dataset<scalar, 6> x, Dataset<scalar, 6> y){
            return x + y;
    });
    //const accmat cov = covs / acc_scalar(count_m);
    //const matrix<scalar, 6, 6> cov2 = mean_and_cov.covariance;
    //std::cout << "Alternative:\n" << cov2 << "\n\n";
    return pear{mean_and_cov.mean, mean_and_cov.covariance};
}


template<typename scalar, space sp>
std::complex<double> particles<scalar, sp>::bunching_factor(scalar lambda_u, scalar frame_gamma){
    using vec3 = ippl::Vector<scalar, 3>;
    const vec3* __restrict__ pn = this->positions;
    //Accumulate a Vector2 that represents a complex
    ippl::Vector<double, 2> sum = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count_m),
        [pn, frame_gamma, lambda_u]KOKKOS_FUNCTION(size_t x) -> ippl::Vector<double, 2>{
            
            scalar z = pn[x][2];
            scalar arg = z / (lambda_u);
            return ippl::Vector<double, 2>{double(cos(arg)), double(sin(arg))};
        },
        ippl::Vector<double, 2>(0),
        []KOKKOS_FUNCTION(ippl::Vector<double, 2> x, ippl::Vector<double, 2> y){
            return x + y;
    });
    std::complex<double> ret(sum[0], sum[1]);
    return ret / double(count_m);
}

template<typename scalar, space sp>
ippl::Vector<scalar, 3> particles<scalar, sp>::average_velocity(scalar dt){
    const scalar idt = scalar(1) / dt;
    const vec3* __restrict__ pn = this->positions;
    const vec3* __restrict__ pnm1 = this->previous_positions;
    vec3 sum = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count_m),
        [pn, pnm1, idt]KOKKOS_FUNCTION(size_t x) -> vec3{
            //return T(1);
            return (pn[x] - pnm1[x]) * idt;
        },
        vec3(0),
        []KOKKOS_FUNCTION(vec3 x, vec3 y){
            return x + y;
    });
    return sum / count_m;
}
template<typename scalar, space sp>
ippl::Vector<double, 3> particles<scalar, sp>::zvariance(){
    using vec3 = ippl::Vector<scalar, 3>;
    const vec3* __restrict__ pn = this->positions;
    using dvec3 = ippl::Vector<double, 3>;
    dvec3 sum = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count_m),
        [pn]KOKKOS_FUNCTION(size_t x) -> dvec3{
            return pn[x].template cast<double>();
        },
        dvec3(0),
        []KOKKOS_FUNCTION(dvec3 x, dvec3 y){
            return x + y;
    });
    const dvec3 avg = sum / double(count_m);
    dvec3 varsum = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count_m),
        [avg, pn]KOKKOS_FUNCTION(size_t x) -> dvec3{
            return (pn[x].template cast<double>() - avg) * (pn[x].template cast<double>() - avg);
        },
        dvec3(0),
        []KOKKOS_FUNCTION(dvec3 x, dvec3 y){
            return x + y;
    });
    return varsum / double(count_m);
}
template struct particles<float, device>;
template struct particles<double, device>;