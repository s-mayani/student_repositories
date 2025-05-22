#ifndef CFG_HPP
#define CFG_HPP
#include <cstdint>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <unordered_map>
#define nokokkos
#include "/run/media/manuel/docpart/gitclones/ippl_seeding_bugfix/ippl_oo/src/Utility/unit.hpp"
using std::uint32_t;
#ifndef IPPL_Vector_H
#ifndef kf
#define kf
#endif
constexpr double sqrt_4pi = 3.54490770181103205459;
constexpr double alpha_scaling_factor = 1e30;
constexpr double unit_length_in_meters = 1.616255e-35 * alpha_scaling_factor;
constexpr double unit_charge_in_electron_charges = 11.70623710394218618969 / sqrt_4pi;
constexpr double unit_time_in_seconds = 5.391247e-44 * alpha_scaling_factor;
constexpr double electron_mass_in_kg = 9.1093837015e-31;
constexpr double unit_mass_in_kg = 2.176434e-8 / alpha_scaling_factor;
constexpr double unit_energy_in_joule = unit_mass_in_kg * unit_length_in_meters * unit_length_in_meters / (unit_time_in_seconds * unit_time_in_seconds);
constexpr double kg_in_unit_masses = 1.0 / unit_mass_in_kg;
constexpr double meter_in_unit_lengths = 1.0 / unit_length_in_meters;
constexpr double electron_charge_in_coulombs = 1.602176634e-19;
constexpr double coulomb_in_electron_charges = 1.0 / electron_charge_in_coulombs;

constexpr double electron_charge_in_unit_charges = 1.0 / unit_charge_in_electron_charges;
constexpr double second_in_unit_times = 1.0 / unit_time_in_seconds;
constexpr double electron_mass_in_unit_masses = electron_mass_in_kg * kg_in_unit_masses;
constexpr double unit_force_in_newtons = unit_mass_in_kg * unit_length_in_meters / (unit_time_in_seconds * unit_time_in_seconds);

constexpr double coulomb_in_unit_charges = coulomb_in_electron_charges * electron_charge_in_unit_charges;
constexpr double unit_voltage_in_volts = unit_energy_in_joule * coulomb_in_unit_charges;
constexpr double unit_charges_in_coulomb = 1.0 / coulomb_in_unit_charges;
constexpr double unit_current_in_amperes = unit_charges_in_coulomb / unit_time_in_seconds;
constexpr double ampere_in_unit_currents = 1.0 / unit_current_in_amperes;
constexpr double unit_current_length_in_ampere_meters = unit_current_in_amperes * unit_length_in_meters;
constexpr double unit_magnetic_fluxdensity_in_tesla = unit_voltage_in_volts * unit_time_in_seconds / (unit_length_in_meters * unit_length_in_meters);
constexpr double unit_electric_fieldstrength_in_voltpermeters = (unit_voltage_in_volts / unit_length_in_meters);
constexpr double voltpermeter_in_unit_fieldstrengths = 1.0 / unit_electric_fieldstrength_in_voltpermeters;
constexpr double unit_powerdensity_in_watt_per_square_meter = 1.389e122 / (alpha_scaling_factor * alpha_scaling_factor * alpha_scaling_factor * alpha_scaling_factor);
constexpr double volts_in_unit_voltages = 1.0 / unit_voltage_in_volts;
constexpr double epsilon0_in_si = unit_current_in_amperes * unit_time_in_seconds / (unit_voltage_in_volts * unit_length_in_meters);
constexpr double mu0_in_si = unit_force_in_newtons / (unit_current_in_amperes * unit_current_in_amperes);
constexpr double G = unit_length_in_meters * unit_length_in_meters * unit_length_in_meters / (unit_mass_in_kg * unit_time_in_seconds * unit_time_in_seconds);
constexpr double verification_gravity = unit_mass_in_kg * unit_mass_in_kg / (unit_length_in_meters * unit_length_in_meters) * G;
constexpr double verification_coulomb = (unit_charges_in_coulomb * unit_charges_in_coulomb / (unit_length_in_meters * unit_length_in_meters) * (1.0 / (epsilon0_in_si))) / unit_force_in_newtons;
template<typename T, typename R>
struct pear{
    T first;
    R second;
};
namespace ippl {
template <typename T, unsigned Dim>
struct Vector {
    using value_type = T;

    constexpr static unsigned dim = Dim;
    T data[Dim];
    kf constexpr Vector(const std::initializer_list<T> &list) {
        // PAssert(list.size() == Dim);
        unsigned int i = 0;
        for (auto &l : list) {
            data[i] = l;
            ++i;
        }
    }
    Vector() = default;
    constexpr kf Vector(T v){
        fill(v);
    }
    kf value_type dot(const Vector<T, Dim>& v)const noexcept{
        value_type ret = 0;
        for(unsigned i = 0;i < dim;i++){
            ret += data[i] * v[i];
        }
        return ret;
    }
    template<unsigned N>
    kf Vector<T, N> tail()const noexcept{
        Vector<T, N> ret;
        static_assert(N <= Dim, "N must be smaller than Dim");
        constexpr unsigned diff = Dim - N;
        for(unsigned i = 0;i < N;i++){
            ret[i] = (*this)[i + diff];
        }
        return ret;
    }
    template<unsigned N>
    kf Vector<T, N> head()const noexcept{
        Vector<T, N> ret;
        static_assert(N <= Dim, "N must be smaller than Dim");
        for(unsigned i = 0;i < N;i++){
            ret[i] = (*this)[i];
        }
        return ret;
    }
    kf value_type squaredNorm()const noexcept{
        return dot(*this);
    }
    kf value_type norm(){
        #ifndef __CUDA_ARCH__
        using std::sqrt;
        #endif
        return sqrt(squaredNorm());
    }
    kf Vector<T, Dim> normalized()const noexcept{
        return *this / norm();
    }
    kf value_type sum()const noexcept{
        value_type ret = 0;
        for(unsigned i = 0;i < dim;i++){
            ret += data[i];
        }
        return ret;
    }
    kf value_type average()const noexcept{
        value_type ret = 0;
        for(unsigned i = 0;i < dim;i++){
            ret += data[i];
        }
        return ret / dim;
    }
    kf Vector<T, 3> cross(const Vector<T, Dim>& v) const noexcept requires(Dim == 3){
        Vector<T, 3> ret(0);
        ret[0] = data[1] * v[2] - data[2] * v[1];
        ret[1] = data[2] * v[0] - data[0] * v[2];
        ret[2] = data[0] * v[1] - data[1] * v[0];
        return ret;
    }
    kf bool operator==(const Vector<T, dim>& o) const noexcept{
        for(unsigned i = 0;i < dim;i++){
            if(data[i] != o[i])return false;
        }
        return true;
    }
    kf value_type &operator[](unsigned int i) noexcept {
        assert(i < dim);
        return data[i];
    }
    kf T* begin()noexcept{
        return data;
    }
    kf T* end()noexcept{
        return data + dim;
    }
    kf const T* begin()const noexcept{
        return data;
    }
    kf const T* end()const noexcept{
        return data + dim;
    }
    kf constexpr void fill(value_type x){
        for (unsigned i = 0; i < dim; i++) {
            data[i] = value_type(x);
        }
    }
    kf const value_type &operator[](unsigned int i) const noexcept {
        assert(i < dim);
        return data[i];
    }

    kf value_type &operator()(unsigned int i) noexcept {
        assert(i < dim);
        return data[i];
    }

    kf const value_type &operator()(unsigned int i) const noexcept {
        assert(i < dim);
        return data[i];
    }
    kf Vector operator-()const noexcept{
        Vector ret;
        for (unsigned i = 0; i < dim; i++) {
            ret[i] = -(*this)[i];
        }
        return ret;
    }
    kf Vector<T, dim> decompose(Vector<int, dim>* integral){
        #ifndef __CUDA_ARCH__
        using std::modf;
        #endif
        Vector<T, dim> ret;
        for(unsigned i = 0;i < dim;i++){
            if constexpr(std::is_same_v<T, float>){
                float tmp;
                ret[i] = modff((*this)[i], &tmp);
                (*integral)[i] = (int)tmp;
            }
            else if constexpr(std::is_same_v<T, double>){
                double tmp;
                ret[i] = modf((*this)[i], &tmp);
                (*integral)[i] = (int)tmp;
            }
        }
        return ret;
    }
    template<typename O>
    constexpr kf Vector<O, dim> cast()const noexcept{
        Vector<O, dim> ret;
        for (unsigned i = 0; i < dim; i++) {
            ret.data[i] = (O)(data[i]);
        }
        return ret;
    }
    kf Vector operator*(const value_type &o) const noexcept {
        Vector ret;
        for (unsigned i = 0; i < dim; i++) {
            ret[i] = (*this)[i] * o;
        }
        return ret;
    }
#define defop_kf(OP)                                        \
    constexpr kf Vector<T, Dim> operator OP(const Vector<T, Dim> &o) const noexcept { \
        Vector<T, Dim> ret;                                         \
        for (unsigned i = 0; i < dim; i++) {                \
            ret.data[i] = data[i] OP o.data[i];                    \
        }                                                   \
        return ret;                                         \
    }
    defop_kf(+)
    defop_kf(-)
    defop_kf(*)
    defop_kf(/)
    #define def_aop_kf(OP)                                                  \
    kf Vector<T, Dim>& operator OP(const Vector<T, Dim> &o) noexcept {      \
        Vector<T, Dim> ret;                                                 \
        for (unsigned i = 0; i < dim; i++) {                                \
            (*this)[i] OP o[i];                                             \
        }                                                                   \
        return *this;                                                       \
    }
    def_aop_kf(+=)
    def_aop_kf(-=)
    def_aop_kf(*=)
    def_aop_kf(/=)
    template<typename stream_t>
    friend stream_t& operator<<(stream_t& str, const Vector<T, Dim>& v){
        //tr << "{";
        for(unsigned i = 0; i < Dim;i++){
            str << v[i];
            if(i < Dim - 1)str << ", ";
        }
        //str << "}";
        return str;
    }
    //defop_kf(*)
    //defop_kf(/)
};
} // namespace ippl

template<typename T, unsigned N>
kf ippl::Vector<T, N> operator *(const T &o, const ippl::Vector<T, N>& v) noexcept {
    ippl::Vector<T, N> ret;
    for (unsigned i = 0; i < N; i++) {
        ret[i] = v[i] * o;
    }
    return ret;
}
template<typename T, int m, int n>
struct matrix{
    /**
     * @brief Column major
     * 
     */
    ippl::Vector<ippl::Vector<T, m>, n> data;
    constexpr static bool squareMatrix = (m == n && m > 0);

    constexpr kf matrix(T diag)
    requires(m == n){
        for(unsigned i = 0;i < n;i++){
            for(unsigned j = 0;j < n;j++){
                data[i][j] = diag * T(i == j);
            }
        }
    }
    constexpr matrix() = default;
    kf constexpr static matrix zero(){
        matrix<T, m, n> ret;
        for(unsigned i = 0;i < n;i++){
            for(unsigned j = 0;j < n;j++){
                ret.data[i][j] = 0;
            }
        }
        return ret;
    };
    kf void setZero(){
        for(unsigned i = 0;i < n;i++){
            for(unsigned j = 0;j < n;j++){
                data[i][j] = 0;
            }
        }
    }

    kf T operator()(int i, int j)const noexcept{
        return data[j][i];
    }
    kf T& operator()(int i, int j)noexcept{
        return data[j][i];
    }
    template<typename O>
    kf matrix<O, m, n> cast()const noexcept{
        matrix<O, m, n> ret;
        for(unsigned i = 0;i < n;i++){
            ret.data[i] = data[i].template cast<O>();
        }
        return ret;
    }
    kf matrix<T, m, n> operator+(const matrix<T, m, n>& other) const {
        matrix<T, m, n> result;
        for (int i = 0; i < n; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Implement matrix subtraction
    kf matrix<T, m, n> operator-(const matrix<T, m, n>& other) const {
        matrix<T, m, n> result;
        for (int i = 0; i < n; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
    kf matrix<T, m, n> operator*(const T& factor) const {
        matrix<T, m, n> result;
        for (int i = 0; i < n; ++i) {
            result.data[i] = data[i] * factor;
        }
        return result;
    }
    kf matrix<T, m, n> operator/(const T& divisor) const {
        matrix<T, m, n> result;
        for (int i = 0; i < n; ++i) {
            result.data[i] = data[i] / divisor;
        }
        return result;
    }

    // Implement matrix-vector multiplication
    template<unsigned int other_m>
    kf ippl::Vector<T, m> operator*(const ippl::Vector<T, other_m>& vec) const {
        static_assert((int)other_m == n);
        ippl::Vector<T, m> result(0);
        for (int i = 0; i < n; ++i) {
            result += vec[i] * data[i];
        }
        return result;
    }
    template<int otherm, int othern>
        requires(n == otherm)
    kf matrix<T, m, othern> operator*(const matrix<T, otherm, othern>& otherMat) const noexcept {
        matrix<T, m, othern> ret(0);
        for(int i = 0;i < m;i++){
            for(int j = 0;j < othern;j++){
                for(int k = 0;k < n;k++){
                    ret(i, j) += (*this)(i, k) * otherMat(k, j);
                }
            }
        }
        return ret;
    }
    kf void addCol(int i, int j, T alpha = 1.0){
        data[j] += data[i] * alpha;
    }
    kf matrix<T, m, n> inverse()const noexcept
        requires (squareMatrix){
        constexpr int N = m;
        
        matrix<T, m, n> ret(1.0);
        matrix<T, m, n> dis(*this);

        for(int i = 0;i < N;i++){
            for(int j = i + 1;j < N;j++){
                T alpha = -dis(i, j) / dis(i, i);
                dis.addCol(i, j, alpha);
                dis(i, j) = 0;
                ret.addCol(i, j, alpha);
            }
        }
        for(int i = N - 1;i >= 0;i--){
            for(int j = i - 1;j >= 0;j--){
                T alpha = -dis(i, j) / dis(i, i);
                dis.addCol(i, j, alpha);
                dis(i, j) = 0;
                ret.addCol(i, j, alpha);
            }
        }
        for(int i = 0;i < N;i++){
            T d = dis(i, i);
            T oneod = T(1) / d;
            dis.data[i] *= oneod;
            ret.data[i] *= oneod;
        }

        return ret;
    }
    
    template<typename stream_t>
    friend stream_t& operator<<(stream_t& str, const matrix<T, m, n>& mat){
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                str << mat.data[j][i] << " ";
            }
            str << "\n";
        }
        return str;
    }
};
template<typename T, unsigned N, unsigned M>
kf matrix<T, N, N> outer_product(const ippl::Vector<T, N>& l, const ippl::Vector<T, M>& r){
    matrix<T, N, M> ret;
    for(unsigned i = 0;i < N;i++){
        for(unsigned j = 0;j < M;j++){
            ret.data[j][i] = l[i] * r[j];
        }
    }
    return ret;
}
#endif
struct config {
    using scalar = double;
    
    //using length_unit = funits::length<scalar, funits::planck_base>;
    //using duration_unit = funits::time<scalar, funits::planck_base>;
    // GRID PARAMETERS
    ippl::Vector<uint32_t, 3> resolution;

    ippl::Vector<scalar, 3> extents;
    scalar total_time;
    scalar timestep_ratio;

    scalar length_scale_in_jobfile, temporal_scale_in_jobfile;

    // All in unit_charge, or unit_mass
    scalar charge, mass;

    uint64_t num_particles;
    bool space_charge;

    // BUNCH PARAMETERS
    ippl::Vector<scalar, 3> mean_position;
    ippl::Vector<scalar, 3> sigma_position;
    ippl::Vector<scalar, 3> position_truncations;
    ippl::Vector<scalar, 3> sigma_momentum;
    scalar bunch_gamma;

    scalar undulator_K;
    scalar undulator_period;
    scalar undulator_length;

    uint32_t output_rhythm;
    std::unordered_map<std::string, double> experiment_options;
};
template<typename T>
struct LorentzFrame{
    constexpr static T c = 1.0;
    using scalar = T;
    using Vector3 = ippl::Vector<T, 3>;
    ippl::Vector<T, 3> beta_m;
    ippl::Vector<T, 3> gammaBeta_m;
    T gamma_m;
    kf LorentzFrame(const ippl::Vector<T, 3>& gammaBeta){
        beta_m = gammaBeta / sqrt(1 + gammaBeta.dot(gammaBeta));
        gamma_m = sqrt(1 + gammaBeta.dot(gammaBeta));
        gammaBeta_m = gammaBeta;
    }
    template<char axis>
    static LorentzFrame uniaxialGamma(T gamma){
        static_assert(axis == 'x' || axis == 'y' || axis == 'z', "Only xyz axis suproted");
        assert(gamma >= 1.0 && "Gamma must be >= 1");
        //using Kokkos::sqrt;
        
        T beta = gamma == 1 ? T(0) : sqrt(gamma * gamma - 1) / gamma;
        Vector3 arg{0,0,0};
        arg[axis - 'x'] = gamma * beta;
        return LorentzFrame<T>(arg);
    }
    kf matrix<T, 4, 4> unprimedToPrimed()const noexcept{
        T betaMagsq = beta_m.dot(beta_m);
        //using Kokkos::abs;
        if(abs(betaMagsq) < 1e-10){
            return matrix<T, 4, 4>(T(1));
        }
        ippl::Vector<T, 3> betaSquared = beta_m * beta_m;

        matrix<T, 4, 4> ret;

        ret.data[0] = ippl::Vector<T, 4>{ gamma_m, -gammaBeta_m[0], -gammaBeta_m[1], -gammaBeta_m[2]};
        ret.data[1] = ippl::Vector<T, 4>{-gammaBeta_m[0], 1 + (gamma_m - 1) * betaSquared[0] / betaMagsq, (gamma_m - 1) * beta_m[0] * beta_m[1] / betaMagsq, (gamma_m - 1) * beta_m[0] * beta_m[2] / betaMagsq};
        ret.data[2] = ippl::Vector<T, 4>{-gammaBeta_m[1], (gamma_m - 1) * beta_m[0] * beta_m[1] / betaMagsq, 1 + (gamma_m - 1) * betaSquared[1] / betaMagsq, (gamma_m - 1) * beta_m[1] * beta_m[2] / betaMagsq};
        ret.data[3] = ippl::Vector<T, 4>{-gammaBeta_m[2], (gamma_m - 1) * beta_m[0] * beta_m[2] / betaMagsq, (gamma_m - 1) * beta_m[1] * beta_m[2] / betaMagsq, 1 + (gamma_m - 1) * betaSquared[2] / betaMagsq};

        return ret;
    }
    kf matrix<T, 4, 4> primedToUnprimed()const noexcept{
        return unprimedToPrimed().inverse();
    }

    kf Vector3 transformV(const Vector3& unprimedV)const noexcept{
        T factor = T(1.0) / (1.0 - unprimedV.dot(beta_m));
        Vector3 ret = unprimedV * scalar(1.0 / gamma_m);
        ret -= beta_m;
        ret += beta_m * (unprimedV.dot(beta_m) * (gamma_m / (gamma_m + 1)));
        return ret * factor;
    }


    kf Vector3 transformGammabeta(const Vector3& gammabeta)const noexcept{
        //using Kokkos::sqrt;
        T gamma = sqrt(T(1) + gammabeta.dot(gammabeta));
        Vector3 beta = gammabeta;
        beta /= gamma;
        Vector3 betatrf = transformV(beta);
        betatrf *= sqrt(1 - betatrf.dot(betatrf));
        return betatrf;
    }

    kf pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_EB(const pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& unprimedEB)const noexcept{
        //using Kokkos::sqrt;
        pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
        Vector3 vnorm = beta_m * (1.0 / sqrt(beta_m.dot(beta_m)));
        //std::cout << "FDTDSolver::289:" << dot_prod(vnorm, vnorm) << "\n";
        ret.first  = (ippl::Vector<T, 3>(unprimedEB.first  + beta_m.cross(unprimedEB.second)) * gamma_m) - (vnorm * (gamma_m - 1) * (unprimedEB.first.dot(vnorm)));
        ret.second = (ippl::Vector<T, 3>(unprimedEB.second - beta_m.cross(unprimedEB.first )) * gamma_m) - (vnorm * (gamma_m - 1) * (unprimedEB.second.dot(vnorm)));
        return ret;
    }
    kf pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>> inverse_transform_EB(const pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& primedEB)const noexcept{
        ippl::Vector<T, 3> mgb(gammaBeta_m * -1.0);
        return LorentzFrame<T>(mgb).transform_EB(primedEB);
    }
    //pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_inverse_EB(const pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& primedEB)const noexcept{
    //    using Kokkos::sqrt;
    //    pear<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
    //    Vector3 vnorm = beta_m * (1.0 / sqrt(dot_prod(beta_m, beta_m)));
    //    ret.first  = (primedEB.first - cross_prod(beta_m, primedEB.second))*gamma - (gamma_m - 1) * (dot_prod(primedEB.first, vnorm) * vnorm);
    //    ret.second = (primedEB.second + cross_prod(beta_m, primedEB.first))*gamma - (gamma_m - 1) * (dot_prod(primedEB.second, vnorm) * vnorm);
    //    return ret;
    //}
};
config read_config(const char *filepath);
#endif