#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <cassert>
#include <fstream>

constexpr double alpha_scaling_factor = 1;
constexpr double unit_length_in_meters = 1.616255e-35 * alpha_scaling_factor;
constexpr double unit_charge_in_electron_charges = 11.71;
constexpr double unit_time_in_seconds = 5.391247e-44 * alpha_scaling_factor;
constexpr double electron_mass_in_kg = 9.1093837015e-31;
constexpr double unit_mass_in_kg = 2.176434e-8 / alpha_scaling_factor;

constexpr double kg_in_unit_masses = 1.0 / unit_mass_in_kg;
constexpr double meter_in_unit_lengths = 1.0 / unit_length_in_meters;
constexpr double electron_charge_in_unit_charges = 1.0 / unit_charge_in_electron_charges;
constexpr double second_in_unit_times = 1.0 / unit_time_in_seconds;

constexpr double electron_mass_in_unit_masses = electron_mass_in_kg * kg_in_unit_masses;
namespace ippl{
    template<typename T, unsigned N>
    using Vector = Eigen::Matrix<T, N, 1>;
    template<typename T, typename U>
    using pear = std::pair<T, U>;
}
template<typename scalar>
struct static_undulator{
    scalar k_u;
    scalar undulator_parameter;
    scalar B_magnitude;
    scalar length;
    scalar distance_to_entry;
    constexpr static_undulator(scalar lambda_u_, scalar undulator_parameter_, scalar length_, scalar distance_to_entry_)
     : k_u((2.0 * M_PI) / lambda_u_), undulator_parameter(undulator_parameter_), B_magnitude((2 * M_PI * electron_mass_in_unit_masses * undulator_parameter) / (electron_charge_in_unit_charges * lambda_u_)), length(length_), distance_to_entry(distance_to_entry_){
        
    }
    constexpr ippl::pear<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> operator()(const ippl::Vector<scalar, 3>& position_in_lab_frame)const noexcept{
        ippl::pear<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> ret;
        if(position_in_lab_frame[2] < distance_to_entry){
            //assert(false);
            scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
            assert(z_in_undulator < 0);
            scalar scal = exp(-((k_u * z_in_undulator) * (k_u * z_in_undulator) * 0.5));
            //printf("Discard: %.4e\n", scal);
            ret.second[0] = 0;
            ret.second[1] = B_magnitude * cosh(k_u * position_in_lab_frame[1]) * z_in_undulator * k_u * scal;
            ret.second[2] = B_magnitude * sinh(k_u * position_in_lab_frame[1]) * scal;
        }
        else if(position_in_lab_frame[2] > distance_to_entry && position_in_lab_frame[2] < distance_to_entry + length){
            scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
            assert(z_in_undulator >= 0);
            ret.second[0] = 0;
            ret.second[1] = B_magnitude * cosh(k_u * position_in_lab_frame[1]) * sin(k_u * z_in_undulator);
            ret.second[2] = B_magnitude * sinh(k_u * position_in_lab_frame[1]) * cos(k_u * z_in_undulator);
        }
        return ret;
        
    }
};
using Eigen::Vector3d;
constexpr double charge = electron_charge_in_unit_charges;
constexpr double mass = electron_mass_in_unit_masses;
constexpr double micro = 1e-6;
constexpr double gamma_to_beta(double gamma){
    return std::sqrt(gamma * gamma - 1) / gamma;
}
constexpr double beta_to_gamma(double beta){
    return 1.0 / std::sqrt(1 - beta * beta);
}
constexpr double gammabeta_to_gamma(double gammabeta){
    return std::sqrt(1 + gammabeta * gammabeta);
}
Vector3d gammabeta_to_beta(Vector3d gammabeta){
    return gammabeta / std::sqrt(1.0 + gammabeta.squaredNorm());
}
/*int main(){
    using scalar = double;
    [[maybe_unused]] constexpr scalar c = 1.0;
    [[maybe_unused]] constexpr scalar lambda_u = 3e4 * micro * meter_in_unit_lengths;
    [[maybe_unused]] constexpr scalar length = 5.0 * meter_in_unit_lengths;
    [[maybe_unused]] constexpr scalar bunch_dt = length / 2e7;
    [[maybe_unused]] constexpr scalar k_u = 2.0 * M_PI / lambda_u;
    [[maybe_unused]] constexpr scalar distance_to_entry = 2.0 * lambda_u;
    [[maybe_unused]] constexpr scalar K = 1.417;
    [[maybe_unused]] constexpr scalar B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K) / (electron_charge_in_unit_charges * lambda_u);

    std::cout << length / lambda_u << "\n";
    std::cout << B_magnitude << "\n";
    std::cout << B_magnitude * charge * (lambda_u) / mass << "\n";

    return 0;
}*/
double do_run(double gamma, double K, bool verbose = false){
    using scalar = double;
    
    Vector3d pos(0.0, 1000.0 * micro * meter_in_unit_lengths, 0.0);
    Vector3d gammabeta(0, 0, gamma_to_beta(gamma) * gamma);
    const scalar lambda_u = 3e4 * (micro) * meter_in_unit_lengths;
    const scalar length = 5.0 * meter_in_unit_lengths;
    const double bunch_dt = length / (2.0 * (gamma * gamma + 100 * gamma + 1000));
    const scalar k_u = 2 * M_PI / lambda_u;
    const scalar distance_to_entry = 5.0 * lambda_u;
    const scalar B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K) / (electron_charge_in_unit_charges * lambda_u);
    static_undulator<scalar> undulator_field(lambda_u, K, length, distance_to_entry);
    
    uint64_t average_beta_count = 0;
    __float128 average_beta = 0.0;
    for(uint64_t step = 0;;step++){
        using std::sqrt;
        const ippl::Vector<scalar, 3> pgammabeta = gammabeta;
        std::pair<Vector3d, Vector3d> ufield = undulator_field(pos);
        //std::pair<Vector3d, Vector3d> ufield{Vector3d(0,0,0), Vector3d(0,0,0)};
        Vector3d E = ufield.first;
        Vector3d B = ufield.second;
        
        const ippl::Vector<scalar, 3> t1 = pgammabeta - charge * bunch_dt * E / (scalar(2) * mass);
        const scalar alpha = -charge * bunch_dt / (scalar(2) * mass * sqrt(1 + t1.dot(t1)));
        const ippl::Vector<scalar, 3> t2 = t1 + alpha * t1.cross(B);
        const ippl::Vector<scalar, 3> t3 = t1 + t2.cross(scalar(2) * alpha * (B / (1.0 + alpha * alpha * (B.dot(B)))));
        const ippl::Vector<scalar, 3> ngammabeta = t3 - charge * bunch_dt * E / (scalar(2) * mass);
        assert((ngammabeta / (sqrt(scalar(1.0) + (ngammabeta.dot(ngammabeta))))).squaredNorm() < 1);
        pos = pos + bunch_dt * ngammabeta / (sqrt(scalar(1.0) + (ngammabeta.dot(ngammabeta))));
        if((step & ((1 << 20) - 1)) == 0){
            std::cerr << step << " timesteps\n";
        }
        if((step & 1) == 0 && verbose)
        {
            //std::cout << gammabeta_to_beta(ngammabeta).z() << "\n";
            //std::cout << "B: " << B.cross(gammabeta).transpose() << "\n";           
            std::cout <<
             pos.transpose() * unit_length_in_meters << " " << 
             beta_to_gamma((ngammabeta / (sqrt(scalar(1.0) + (ngammabeta.dot(ngammabeta))))).z()) << "\n";
            //std::cout << gammabeta - ngammabeta << "\n";
            //std::cout << (ngammabeta / std::sqrt(1 + gammabeta.squaredNorm())).transpose() << "\n";
        }
        gammabeta = ngammabeta;
        if(pos.hasNaN() || pos.z() > distance_to_entry + length + lambda_u)break;
        if(pos.z() >= distance_to_entry){
            average_beta_count++;
            average_beta += __float128((ngammabeta / (sqrt(scalar(1.0) + (ngammabeta.dot(ngammabeta))))).z());
        }
    }
    std::cerr << "Average gamma in undulator: " << beta_to_gamma(double(average_beta / __float128(average_beta_count))) << "\n";
    return beta_to_gamma(double(average_beta / __float128(average_beta_count)));
}
int main(){
    std::cout.precision(10);
    std::ofstream boris_gammas("boris_gammas.txt");
    for(double K = 0.1; K < 7.9; K += 0.05){
        boris_gammas << K << " ";
        for(double gamma : {20.0, 100.0, 1000.0}){
            double avg_gamma = do_run(gamma, K);
            boris_gammas << avg_gamma << " ";
        }
        boris_gammas << "\n";
    }
    double avg_gamma = do_run(100.41, 1.417);
    std::cout << "Fel-IR: " << avg_gamma << "\n";
}