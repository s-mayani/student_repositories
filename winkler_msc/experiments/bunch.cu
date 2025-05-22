#ifndef BUNCH_CUH
#define BUNCH_CUH
#include "common.cu"
#include <complex>
#include "cfg.hpp"
#include "grid.cu"
template <typename T, unsigned N>
struct Dataset {
    ippl::Vector<T, N> mean;
    matrix<T, N, N> covariance;
    size_t weight;
    Dataset() = default;
    KOKKOS_INLINE_FUNCTION Dataset(const ippl::Vector<T, N>& m, const matrix<T, N, N>& cov, size_t w)
        : mean(m), covariance(cov), weight(w) {}

    /**
     * Overloaded + operator to combine two datasets.
     * 
     * @param other The dataset to be combined with.
     * @return Dataset containing the combined mean, covariance, and weight.
     */
    KOKKOS_FUNCTION __attribute__((noinline)) Dataset operator+(const Dataset& other) const {
        size_t total_weight = weight + other.weight;

        ippl::Vector<T, N> combined_mean = (mean * weight + other.mean * other.weight) / total_weight;

        // Calculate mean deviations
        ippl::Vector<T, N> mean_dev1 = mean - combined_mean;
        ippl::Vector<T, N> mean_dev2 = other.mean - combined_mean;

        // Calculate outer products of mean deviations
        matrix<T, N, N> outer_prod_mean_dev1 = outer_product(mean_dev1, mean_dev1);
        matrix<T, N, N> outer_prod_mean_dev2 = outer_product(mean_dev2, mean_dev2);

        matrix<T, N, N> combined_covariance = ((covariance * weight + other.covariance * other.weight) / total_weight) +
                                               ((outer_prod_mean_dev1 * weight +
                                                 outer_prod_mean_dev2 * other.weight) / total_weight);

        return Dataset(combined_mean, combined_covariance, total_weight);
    }
};
using acc_scalar = double;
template<typename _scalar, space sp>
struct particles{
    using scalar = _scalar;
    using vec3 = ippl::Vector<scalar, 3>;
    vec3* __restrict__ positions;
    vec3* __restrict__ previous_positions;
    vec3* __restrict__ gammaBeta;
    size_t count_m;
    scalar charge_per_particle;
    scalar mass_per_particle;
    particles() : positions(nullptr), previous_positions(nullptr), gammaBeta(nullptr), count_m(0){}
    particles(size_t count, scalar cpp = electron_charge_in_unit_charges, scalar mpp = electron_mass_in_unit_masses) : count_m(count), charge_per_particle(cpp), mass_per_particle(mpp){
        if constexpr(sp == space::device){
            positions = cuda_malloc_helper<ippl::Vector<scalar, 3>>(count);
            previous_positions = cuda_malloc_helper<ippl::Vector<scalar, 3>>(count);
            gammaBeta = cuda_malloc_helper<ippl::Vector<scalar, 3>>(count);
        }
        else if constexpr(sp == space::host){
            positions = cuda_malloc_host_helper<ippl::Vector<scalar, 3>>(count);
            previous_positions = cuda_malloc_host_helper<ippl::Vector<scalar, 3>>(count);
            gammaBeta = cuda_malloc_host_helper<ippl::Vector<scalar, 3>>(count);
        }
        else{
            
        }
    }
    KOKKOS_INLINE_FUNCTION index_range<1> getRange()const noexcept{
        return index_range<1>{.begin = ippl::Vector<uint32_t, 1>{0},
                              .end = ippl::Vector<uint32_t, 1>{(uint32_t)count_m}};
    }
    void resize(size_t newsize){
        particles<scalar, sp> niu(newsize);
        size_t copysize = std::min(count_m, newsize);
        if(copysize > 0){
            assert(positions);
            assert(previous_positions);
            assert(gammaBeta);
            assert(niu.positions);
            assert(niu.previous_positions);
            assert(niu.gammaBeta);
            if constexpr(sp == space::host){
                cudaMemcpy(niu.positions, positions, copysize * sizeof(vec3), cudaMemcpyDeviceToDevice);
                cudaMemcpy(niu.previous_positions, previous_positions, copysize * sizeof(vec3), cudaMemcpyDeviceToDevice);
                cudaMemcpy(niu.gammaBeta, gammaBeta, copysize * sizeof(vec3), cudaMemcpyDeviceToDevice);
            }
            else{
                cudaMemcpy(niu.positions, positions, copysize * sizeof(vec3), cudaMemcpyHostToHost);
                cudaMemcpy(niu.previous_positions, previous_positions, copysize * sizeof(vec3), cudaMemcpyHostToHost);
                cudaMemcpy(niu.gammaBeta, gammaBeta, copysize * sizeof(vec3), cudaMemcpyHostToHost);
            }
        }
        std::swap(positions, niu.positions);
        std::swap(previous_positions, niu.previous_positions);
        std::swap(gammaBeta, niu.gammaBeta);
        std::swap(count_m, niu.count_m);
        niu.destroy();
        assert(count_m == newsize);
    }
    void destroy(){
        if constexpr(sp == space::device){
            if(positions)cudaFree(positions);
            if(previous_positions)cudaFree(previous_positions);
            if(gammaBeta)cudaFree(gammaBeta);
        }
        else if constexpr(sp == space::host){
            if(positions)cudaFreeHost(positions);
            if(previous_positions)cudaFreeHost(previous_positions);
            if(gammaBeta)cudaFreeHost(gammaBeta);
        }
        positions = nullptr;
        previous_positions = nullptr;
        gammaBeta = nullptr;
    }

    template<typename spatial_eb_function, typename should_experience_gathered_field>
    void update(grid<ippl::Vector<ippl::Vector<scalar, 3>, 2>, device> EB_grid, const scalar bunch_dt, const scalar time_arg, LorentzFrame<scalar> lf, spatial_eb_function undulator_field, should_experience_gathered_field activation_func, const uint32_t stepcount){
        matrix<scalar, 4, 4> bunch_to_lab = lf.primedToUnprimed();
        assert(lf.gammaBeta_m[0] == 0);
        assert(lf.gammaBeta_m[1] == 0);
        const UniaxialLorentzframe<scalar, 2> ulb(lf.gammaBeta_m[2]);
        //std::cout << bunch_to_lab << "\n";
        const scalar charge = charge_per_particle;
        const scalar mass   = mass_per_particle;
        //std::cout << "Charge to mass ratio: " << charge * bunch_dt / mass << "\n";
        for_each([=]KOKKOS_FUNCTION(vec3& prev, vec3& r, vec3& gammabeta){
            //#pragma unroll 1
            for(uint32_t step = 0;step < stepcount;step++){
                ippl::Vector<ippl::Vector<scalar, 3>, 2> EB = gather(EB_grid, r);
                //ippl::Vector<scalar, 3> labpos = (bunch_to_lab * prepend_t(r, time)).template tail<3>();
                scalar time = time_arg + step * bunch_dt;
                ippl::Vector<scalar, 3> labpos = r;
                ulb.primedToUnprimed(labpos, time);
                if(!activation_func(labpos)){
                    EB.fill(0);
                }
                pear<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> EB_undulator_frame = undulator_field(labpos);
                //printf("Undulator field Lab: %.3e, %.3e, %.3e\n", EB_undulator_frame.second[0], EB_undulator_frame.second[1], EB_undulator_frame.second[2]);
                pear<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> EB_undulator_bunch = ulb.transform_EB(EB_undulator_frame);
                //printf("Undulator field bunch: %.3e, %.3e, %.3e\n", EB_undulator_bunch.second[0], EB_undulator_bunch.second[1], EB_undulator_bunch.second[2]);
                //printf("%.4e %.4e\n", time, EB_undulator_bunch.second[1]);
                //printf("Gathered E: %f, %f, %f\n", EB[0][0], EB[0][1], EB[0][2]);
                //printf("Gathered B: %f, %f, %f\n", EB[1][0], EB[1][1], EB[1][2]);
                //printf("Bunchpos: %f %f %f\n", r[0], r[1], r[2]);
                //printf("Labpos: %f %f %f %f\n", labpos[0], labpos[1], labpos[2], labpos[3]);
                {
                    //vec3 beta = gammabeta / sqrt(scalar(1) + gammabeta.squaredNorm());
                    //vec3 gatherforce = EB[0] * charge + beta.cross(EB[1]) * charge;
                    //vec3 undulator_force = EB_undulator_bunch.first * charge + beta.cross(EB_undulator_bunch.first) * charge;
                    //scalar ratio = sqrt(gatherforce.squaredNorm()) / sqrt(undulator_force.squaredNorm());
                    //if(!isinf(ratio) && !isnan(ratio) && ratio < 1)
                    //    printf("Ratio: %.3e\n", ratio);
                }
                EB[0] += EB_undulator_bunch.first;
                EB[1] += EB_undulator_bunch.second;
                //EB[0] *= scalar(-1);
                //EB[1] *= scalar(-1);
                //printf("Undulator E: %.3e, %.3e, %.3e\n", EB_undulator_bunch.first[0], EB_undulator_bunch.first[1], EB_undulator_bunch.first[2]);
                //printf("Undulator B: %.3e, %.3e, %.3e\n", EB_undulator_bunch.second[0], EB_undulator_bunch.second[1], EB_undulator_bunch.second[2]);

                //EB[0].fill(0);
                //EB[1].fill(0);
                //EB[1][1] = 1.0;
                assert_isreal((gammabeta[0]));
                assert_isreal((gammabeta[1]));
                assert_isreal((gammabeta[2]));
                const ippl::Vector<scalar, 3> pgammabeta = gammabeta;

                const ippl::Vector<scalar, 3> t1 = pgammabeta + charge * bunch_dt * EB[0] / (scalar(2) * mass);
                const scalar alpha = charge * bunch_dt / (scalar(2) * mass * sqrt(1 + t1.dot(t1)));
                const ippl::Vector<scalar, 3> t2 = t1 + alpha * t1.cross(EB[1]);
                const ippl::Vector<scalar, 3> t3 = t1 + t2.cross(scalar(2) * alpha * (EB[1] / (1.0  + alpha * alpha * (EB[1].dot(EB[1])))));
                const ippl::Vector<scalar, 3> ngammabeta = t3 + charge * bunch_dt * EB[0] / (scalar(2) * mass);
                //prev = r;
                //printf("Before: %f, %f, %f\n", r[0], r[1], r[2]);
                assert_isreal((ngammabeta[0]));
                assert_isreal((ngammabeta[1]));
                assert_isreal((ngammabeta[2]));
                assert_isreal((r[0]));
                assert_isreal((r[1]));
                assert_isreal((r[2]));
                r = r + bunch_dt * ngammabeta / (sqrt(scalar(1.0) + (ngammabeta.dot(ngammabeta))));
                assert_isreal((r[0]));
                assert_isreal((r[1]));
                assert_isreal((r[2]));

                //printf("R After: %.3e, %.3e, %.3e\n", r[0], r[1], r[2]);

                //printf("Before: %.3e, %.3e, %.3e\n", gammabeta[0], gammabeta[1], gammabeta[2]);
                gammabeta = ngammabeta;
            }
            //printf("After : %.3e, %.3e, %.3e\n", gammabeta[0], gammabeta[1], gammabeta[2]);
        });
    }
    
    pear<ippl::Vector<scalar, 6>, matrix<scalar, 6, 6>> covariance_matrix()const noexcept;
    template<typename functor>
    void for_each_const(functor&& callable)const{
        if(count_m == 0)return;
        const ippl::Vector<scalar, 3>* positions = this->positions;
        const ippl::Vector<scalar, 3>* prepositions = this->previous_positions;
        const ippl::Vector<scalar, 3>* gammaBeta = this->gammaBeta;
        parallel_for_cuda([=]KOKKOS_FUNCTION(size_t idx){
            const ippl::Vector<scalar, 3>& pp = positions[idx];
            const ippl::Vector<scalar, 3>& prep = prepositions[idx];
            const ippl::Vector<scalar, 3>& gb = gammaBeta[idx];
            callable(prep, pp, gb);
        }, getRange());
    }

    template<typename functor>
    void for_each(functor&& callable){
        if(count_m == 0)return;
        ippl::Vector<scalar, 3>* positions = this->positions;
        ippl::Vector<scalar, 3>* prepositions = this->previous_positions;
        ippl::Vector<scalar, 3>* gammaBeta = this->gammaBeta;
        parallel_for_cuda([=]KOKKOS_FUNCTION(size_t idx){
            ippl::Vector<scalar, 3>& pp = positions[idx];
            ippl::Vector<scalar, 3>& prep = prepositions[idx];
            ippl::Vector<scalar, 3>& gb = gammaBeta[idx];
            callable(prep, pp, gb);
        }, getRange());
    }
    template<typename scalar>
    void scatterCurrent(grid<ippl::Vector<scalar, 4>, sp> g, scalar dt){
        const ippl::Vector<scalar, 3>* positions = this->positions;
        const ippl::Vector<scalar, 3>* prepositions = this->previous_positions;
        //const ippl::Vector<scalar, 3>* gammaBeta = this->gammaBeta;

        const scalar volume = g.m_mesh.spacing[0] * g.m_mesh.spacing[1] * g.m_mesh.spacing[2];
        const scalar cpp = charge_per_particle;
        parallel_for_cuda([=]KOKKOS_FUNCTION(size_t idx)mutable{
            scatterLineToGrid<scalar, false>(g, prepositions[idx], positions[idx], cpp / (dt * volume));
        }, getRange());
    }
    template<typename scalar>
    void scatter(grid<ippl::Vector<scalar, 4>, sp> g, scalar dt){
        
        const vec3* positions = this->positions;
        //ippl::Vector<scalar, 3>* gammaBeta = this->gammaBeta;
        
        const scalar cpp = this->charge_per_particle;
        const scalar volume = g.m_mesh.spacing[0] * g.m_mesh.spacing[1] * g.m_mesh.spacing[2];
        assert_isreal(volume);
        assert_isreal((scalar(1) / volume));
        //std::cout << "cpp: " << cpp << "\n";
        //std::cout << "volume: " << volume << "\n";
        assert_isreal((cpp / volume));
        parallel_for_cuda([=]KOKKOS_FUNCTION(size_t idx)mutable{
            scatterToGrid(g, positions[idx], cpp / volume);
        }, getRange());
        //std::cout << "Ransch: " << getRange().end[0] << "\n";
    }
    template<bool full = true>
    void update_hostcopy(particles<scalar, host>& ret)const noexcept{
        //TODO update count
        if(count_m != ret.count_m){
            ret.resize(count_m);
        }
        
        cudaMemcpy(ret.positions, positions, count_m * sizeof(vec3), cudaMemcpyDeviceToHost);
        if constexpr(full){
            cudaMemcpy(ret.previous_positions, previous_positions, count_m * sizeof(vec3), cudaMemcpyDeviceToHost);
            cudaMemcpy(ret.gammaBeta, gammaBeta, count_m * sizeof(vec3), cudaMemcpyDeviceToHost);
        }
    }
    template<bool full = true>
        requires(sp == device)
    void update_from_hostcopy(const particles<scalar, host>& ref)noexcept{
        if(count_m != ref.count_m){
            resize(ref.count_m);
        }


        cudaMemcpy(positions, ref.positions, count_m * sizeof(vec3), cudaMemcpyHostToDevice);
        if constexpr(full){
            cudaMemcpy(previous_positions, ref.previous_positions, count_m * sizeof(vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(gammaBeta, ref.gammaBeta, count_m * sizeof(vec3), cudaMemcpyHostToDevice);
        }
    }
    particles<scalar, host> hostCopy()const noexcept{
        particles<scalar, host> ret;
        if(count_m == 0)return ret;
        ret.count_m = count_m;
        ret.positions = cuda_malloc_host_helper<vec3>(count_m);
        cudaMemcpy(ret.positions, positions, count_m * sizeof(vec3), cudaMemcpyDeviceToHost);
        
        ret.previous_positions = cuda_malloc_host_helper<vec3>(count_m);
        cudaMemcpy(ret.previous_positions, previous_positions, count_m * sizeof(vec3), cudaMemcpyDeviceToHost);
        
        ret.gammaBeta = cuda_malloc_host_helper<vec3>(count_m);
        cudaMemcpy(ret.gammaBeta, gammaBeta, count_m * sizeof(vec3), cudaMemcpyDeviceToHost);
        return ret;
    }
    /**
     * @brief Computes the bunching factor for this bunch with corrected radiation period lambda_s
     * 
     * @param dt 
     * @param lambda_s = lambda_r * beta / beta_0 = lambda_u / (2 * (gamma_0^2)) * beta / beta_0
     * @return double 
     */
    std::complex<double> bunching_factor(scalar lambda_u, scalar frame_gamma);
    vec3 average_velocity(scalar dt);
    ippl::Vector<double, 3> zvariance();
};
#endif